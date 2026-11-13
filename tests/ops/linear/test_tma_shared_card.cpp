// Two tensor-parallel ranks on one card (--devices 0,0) run their shares on two streams at once.
//
// A consumer warp of a TMA-pipelined kernel used to release its stage while its own ldmatrix reads
// of that stage were still pending, so the producer's refill could overwrite bytes the warp had
// not read yet. It showed under a concurrent kernel on the same card, as wrong values in one warp's
// rows (cta_mbarrier_release_stage). This harness runs each share alone for a reference, then
// interleaves it with a partner share on a second stream without synchronizing, and compares every
// concurrent output with its reference bit for bit. On an RTX 5090 the FP8 shares reach the A8
// route's 32x64, 64x64, 64x128 and 128x64 tiles, where the race showed most; the NVFP4 share runs
// the A4 TMA route, which has the same release.

#include "core/arena.h"
#include "core/device_buffer.h"
#include "core/device_launch.h"
#include "core/qtype.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
namespace qw = quantized_weight;

constexpr std::int32_t kK      = 5120;
constexpr std::int32_t kTokens = 1024;

struct Share {
    QType qtype;
    std::int32_t rows;
    std::uint32_t seed;
};

struct Operands {
    DeviceBuffer weight_bytes;
    Weight weight{};
    DeviceBuffer x, y;
    WorkspaceArena workspace;
    ops::LinearPolicy policy;
    std::size_t outputs;

    Operands(const Share& share, std::size_t capacity)
        : workspace(capacity),
          policy(share.qtype == QType::NVFP4 ? ops::LinearPolicy::AllowA4
                                             : ops::LinearPolicy::AllowA8),
          outputs(static_cast<std::size_t>(share.rows) * kTokens) {
        qw::PatternedWeightOptions options;
        if (share.qtype == QType::NVFP4) {
            options.weight_scale_divisor = 1.0F;
            options.input_scale_divisor  = 6.0F;
        }
        const auto packed = qw::make_patterned_weight(share.qtype, share.rows, kK, share.seed, options);
        weight_bytes      = DeviceBuffer(packed.payload.size());
        weight_bytes.copy_from_host(packed.payload.data(), packed.payload.size());
        weight = packed.device_weight(weight_bytes.p);
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(kK) * kTokens);
        for (std::size_t i = 0; i < bits.size(); ++i) {
            const std::uint64_t mixed = qw::detail::mix64((std::uint64_t{share.seed} << 40) | i);
            bits[i] = f32_to_bf16(static_cast<float>(static_cast<double>(mixed & 0xffffU) / 65536.0 - 0.5));
        }
        x = DeviceBuffer(bits.size() * 2);
        x.copy_from_host(bits.data(), x.bytes);
        y = DeviceBuffer(outputs * 2);
    }

    void run(cudaStream_t stream) {
        workspace.reset();
        Tensor input(x.p, DType::BF16, {kK, kTokens});
        Tensor output(y.p, DType::BF16, {weight.n, kTokens});
        ops::linear(input, weight, output, policy, workspace, stream);
    }

    std::vector<std::uint16_t> read() const {
        std::vector<std::uint16_t> bits(outputs);
        y.copy_to_host(bits.data(), outputs * 2);
        return bits;
    }
};

std::size_t capacity(const Share& share) {
    return std::max<std::size_t>(
        ops::linear_workspace_capacity_bytes(share.qtype, share.rows, kK,
                                             share.qtype == QType::NVFP4
                                                 ? ops::LinearPolicy::AllowA4
                                                 : ops::LinearPolicy::AllowA8,
                                             kTokens, kTokens),
        256);
}

// Runs `share` and `partner` alone, then `kRounds` times four interleaved calls of each on their own
// streams, and counts the rounds in which either final output differs from its reference.
int run_pair(const Share& share, const Share& partner) {
    constexpr int kRounds = 20;
    Operands a(share, capacity(share)), b(partner, capacity(partner));
    cudaStream_t stream_a = nullptr, stream_b = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking), "stream a");
    cuda_check(cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking), "stream b");
    a.run(stream_a);
    cuda_check(cudaStreamSynchronize(stream_a), "reference a");
    b.run(stream_b);
    cuda_check(cudaStreamSynchronize(stream_b), "reference b");
    const auto reference_a = a.read(), reference_b = b.read();
    int differing_rounds    = 0;
    std::size_t worst       = 0;
    for (int round = 0; round < kRounds; ++round) {
        cuda_check(cudaMemset(a.y.p, 0xff, a.y.bytes), "poison a");
        cuda_check(cudaMemset(b.y.p, 0xff, b.y.bytes), "poison b");
        cuda_check(cudaDeviceSynchronize(), "poison");
        for (int call = 0; call < 4; ++call) {
            a.run(stream_a);
            b.run(stream_b);
        }
        cuda_check(cudaDeviceSynchronize(), "concurrent calls");
        const auto got_a = a.read(), got_b = b.read();
        std::size_t differing = 0;
        for (std::size_t i = 0; i < got_a.size(); ++i) differing += got_a[i] != reference_a[i];
        for (std::size_t i = 0; i < got_b.size(); ++i) differing += got_b[i] != reference_b[i];
        differing_rounds += differing != 0;
        worst = std::max(worst, differing);
    }
    cuda_check(cudaStreamDestroy(stream_a), "destroy stream a");
    cuda_check(cudaStreamDestroy(stream_b), "destroy stream b");
    const auto name = [](const Share& s) { return s.qtype == QType::NVFP4 ? "NVFP4" : "FP8"; };
    if (differing_rounds != 0) {
        std::cerr << name(share) << " [" << share.rows << "," << kK << "] beside " << name(partner)
                  << " [" << partner.rows << "," << kK << "] T=" << kTokens << ": "
                  << differing_rounds << '/' << kRounds
                  << " concurrent rounds differ from the calls alone (worst " << worst
                  << " outputs)\n";
    }
    return differing_rounds != 0 ? 1 : 0;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const QType fp8 = QType::FP8_E4M3FN_ROW_BF16;
        int failures    = 0;
        for (const std::int32_t rows : {256, 512, 704, 1024})
            failures += run_pair({fp8, rows, 6101U}, {fp8, rows, 6203U});
        failures += run_pair({fp8, 1024, 6107U}, {fp8, 3072, 6211U});
        failures += run_pair({QType::NVFP4, 8704, 6113U}, {QType::NVFP4, 8704, 6217U});
        failures += run_pair({QType::NVFP4, 8704, 6119U}, {fp8, 1024, 6221U});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " TMA shares on one card\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "TMA shares on one card: " << error.what() << '\n';
        return 1;
    }
}
