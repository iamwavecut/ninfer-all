// Runs a schedule of DSpine runtime calls (models/dspine/runtime.h) for runtime_check.py, which
// converts a small random DSpine checkpoint, writes the schedule and checks every proposal against
// the drafter's FP64 oracle:
//
//   ninfer_tests ninfer_dspine_runtime_driver --check-cuda             exit 77 without a device
//   ninfer_tests ninfer_dspine_runtime_driver ARTIFACT SCHEDULE OUTPUT
//
// SCHEDULE holds little-endian 32-bit words: the magic and version, the target's hidden width,
// vocabulary, layer count, candidate domain and the drafter's taps, the runtime options; the
// target's BF16 token rows and head, both [vocab, hidden]; each sequence's position count and FP32
// features [positions, taps, hidden]; then the calls, each a code and its operands:
//
//   1 store  seq first n tap bf16 column       tap `tap` of the sequence's positions
//                                              [first, first + n), as FP32 or as BF16 rows
//   2 commit lane position column count
//   3 rows   n column, then n x (lane position)  commit_rows
//   4 copy   from to                            copy_lane
//   5 save   lane position                      the lane's context-cache image to host memory
//   6 load   lane position                      the saved image into the lane
//   7 draft  b k, then b x (lane anchor position)
//
// OUTPUT receives each draft call's drafts (I32 [b, k], row after row), then its confidences
// (FP32 [b, k]). The target embeds by copying BF16 rows and projects by a BF16 linear head.

#include "core/arena.h"
#include "core/device.h"
#include "core/device_launch.h"
#include "core/qtype.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "cuda_availability.h"
#include "models/dspine/model.h"
#include "models/dspine/runtime.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_policy.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
namespace dspine = ninfer::models::dspine;

constexpr std::uint32_t kMagic   = 0x4e505344; // "DSPN"
constexpr std::uint32_t kVersion = 1;

class Input {
public:
    explicit Input(const char* path) : in_(path, std::ios::binary) {
        if (!in_) { throw std::runtime_error(std::string("cannot read ") + path); }
        in_.exceptions(std::ios::badbit | std::ios::failbit);
    }
    std::uint32_t word() {
        std::uint32_t value = 0;
        in_.read(reinterpret_cast<char*>(&value), sizeof(value));
        return value;
    }
    template <class T> std::vector<T> array(std::size_t count) {
        std::vector<T> values(count);
        in_.read(reinterpret_cast<char*>(values.data()),
                 static_cast<std::streamsize>(count * sizeof(T)));
        return values;
    }

private:
    std::ifstream in_;
};

std::uint16_t to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16U) & 1U); // nearest, ties to even (finite inputs)
    return static_cast<std::uint16_t>(bits >> 16U);
}

Weight bf16_matrix(const DeviceBuffer& data, std::int32_t rows, std::int32_t columns) {
    Weight weight{};
    weight.qtype           = QType::BF16;
    weight.layout          = QuantLayout::Contiguous;
    weight.payload         = data.p;
    weight.payload_bytes   = data.bytes;
    weight.qdata           = data.p;
    weight.ndim            = 2;
    weight.shape[0]        = rows;
    weight.shape[1]        = columns;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = columns;
    weight.n               = rows;
    weight.k               = columns;
    return weight;
}

std::int32_t extent(std::uint32_t value) {
    if (value == 0 || value > 1U << 30U) { throw std::runtime_error("schedule: extent out of range"); }
    return static_cast<std::int32_t>(value);
}

int run(const char* artifact, const char* schedule, const char* output_path) {
    Input in(schedule);
    if (in.word() != kMagic || in.word() != kVersion) {
        throw std::runtime_error("schedule: not a DSpine runtime schedule of this version");
    }
    const std::uint32_t hidden = in.word(), vocab = in.word(), layers = in.word();
    const std::uint32_t domain = in.word(), taps = in.word();
    dspine::RuntimeOptions options;
    options.sequences       = in.word();
    options.kept_states     = in.word();
    options.max_context     = in.word();
    options.draft_tokens    = in.word();
    options.draft_rows      = in.word();
    options.commit_rows     = in.word();
    options.feature_columns = in.word();
    options.piece_columns   = in.word();
    const std::size_t cells = std::size_t(vocab) * hidden;
    const auto table_rows   = in.array<std::uint16_t>(cells);
    const auto head_rows    = in.array<std::uint16_t>(cells);
    std::vector<std::vector<float>> features(options.sequences);
    for (auto& sequence : features) {
        sequence = in.array<float>(std::size_t(in.word()) * taps * hidden);
    }

    DeviceContext device(0);
    const cudaStream_t stream = device.stream;
    const auto model          = dspine::load_model(artifact, device);
    const dspine::Config& c   = model->config();
    if (c.hidden_size != hidden || c.vocab_size != vocab || c.taps() != taps ||
        c.num_target_layers != layers) {
        throw std::runtime_error("schedule: the artifact's geometry differs from the schedule's");
    }
    DeviceBuffer table(cells * 2), head(cells * 2);
    table.copy_from_host(table_rows.data(), cells * 2);
    head.copy_from_host(head_rows.data(), cells * 2);
    const Weight table_weight = bf16_matrix(table, extent(vocab), extent(hidden));
    const Weight head_weight  = bf16_matrix(head, extent(vocab), extent(hidden));
    options.head_bytes        = ops::linear_workspace_capacity_bytes(
        QType::BF16, extent(vocab), extent(hidden), ops::LinearPolicy::A16Only, 1,
        extent(options.draft_tokens * options.draft_rows));
    dspine::Target target{
        .hidden = hidden,
        .vocab  = vocab,
        .layers = layers,
        .domain = static_cast<std::int32_t>(domain),
        .logits = DType::BF16,
        .embed  = [table_weight](const Tensor& ids, Tensor& rows,
                                cudaStream_t s) { ops::embedding(ids, table_weight, rows, s); },
        .head   = [head_weight](const Tensor& x, Tensor& logits, WorkspaceArena& work,
                              cudaStream_t s) {
            ops::linear(x, head_weight, logits, ops::LinearPolicy::A16Only, work, s);
        }};
    dspine::Runtime runtime(*model, std::move(target), options);
    runtime.prepare(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::ofstream out(output_path, std::ios::binary);
    if (!out) { throw std::runtime_error(std::string("cannot write ") + output_path); }
    out.exceptions(std::ios::badbit | std::ios::failbit);
    std::unique_ptr<PinnedHostBuffer> image;
    std::uint64_t image_size = 0;
    const std::uint32_t calls = in.word();
    for (std::uint32_t call = 0; call < calls; ++call) {
        const std::uint32_t code = in.word();
        switch (code) {
        case 1: {
            const std::uint32_t seq = in.word(), first = in.word(), n = in.word();
            const std::uint32_t tap = in.word(), as_bf16 = in.word(), column = in.word();
            const auto& source = features.at(seq);
            if ((std::size_t(first) + n) * taps * hidden > source.size() || tap >= taps) {
                throw std::runtime_error("schedule: a store outside the sequence's features");
            }
            std::vector<float> rows(std::size_t(n) * hidden);
            for (std::uint32_t p = 0; p < n; ++p) {
                std::memcpy(rows.data() + std::size_t(p) * hidden,
                            source.data() + ((std::size_t(first) + p) * taps + tap) * hidden,
                            std::size_t(hidden) * sizeof(float));
            }
            DeviceBuffer staged(rows.size() * (as_bf16 != 0 ? 2 : 4));
            if (as_bf16 != 0) {
                std::vector<std::uint16_t> half(rows.size());
                for (std::size_t i = 0; i < rows.size(); ++i) { half[i] = to_bf16(rows[i]); }
                staged.copy_from_host(half.data(), half.size() * 2);
            } else {
                staged.copy_from_host(rows.data(), rows.size() * 4);
            }
            const Tensor tensor(staged.p, as_bf16 != 0 ? DType::BF16 : DType::FP32,
                                {extent(hidden), extent(n)});
            runtime.store_tap(tap, tensor, column, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream)); // before `staged` is freed
            break;
        }
        case 2: {
            const std::uint32_t lane = in.word(), position = in.word(), column = in.word();
            runtime.commit(lane, position, column, in.word(), stream);
            break;
        }
        case 3: {
            const std::uint32_t n = in.word(), column = in.word();
            std::vector<std::int32_t> lanes(n), positions(n);
            for (std::uint32_t i = 0; i < n; ++i) {
                lanes[i]     = static_cast<std::int32_t>(in.word());
                positions[i] = static_cast<std::int32_t>(in.word());
            }
            DeviceBuffer d_lanes(std::size_t(n) * 4), d_positions(std::size_t(n) * 4);
            d_lanes.copy_from_host(lanes.data(), lanes.size() * 4);
            d_positions.copy_from_host(positions.data(), positions.size() * 4);
            runtime.commit_rows(Tensor(d_lanes.p, DType::I32, {extent(n)}),
                                Tensor(d_positions.p, DType::I32, {extent(n)}), column, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream)); // before the operands are freed
            break;
        }
        case 4: {
            const std::uint32_t from = in.word();
            runtime.copy_lane(from, in.word(), stream);
            break;
        }
        case 5:
        case 6: {
            const bool save          = code == 5;
            const std::uint32_t lane = in.word(), position = in.word();
            const std::uint64_t bytes = runtime.image_bytes(position);
            if (save) {
                image      = std::make_unique<PinnedHostBuffer>(bytes);
                image_size = bytes;
            } else if (image == nullptr || bytes != image_size) {
                throw std::runtime_error("schedule: a load without a saved image of its size");
            }
            auto* host          = static_cast<std::uint8_t*>(image->data());
            std::uint64_t moved = 0;
            runtime.copy_image(lane, position, [&](auto* region, std::size_t size) {
                if (moved + size > image_size) {
                    throw std::runtime_error("image regions exceed image_bytes()");
                }
                if (save) {
                    CUDA_CHECK(cudaMemcpyAsync(host + moved, region, size, cudaMemcpyDeviceToHost,
                                               stream));
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(region, host + moved, size, cudaMemcpyHostToDevice,
                                               stream));
                }
                moved += size;
            });
            if (moved != image_size) {
                throw std::runtime_error("image regions fall short of image_bytes()");
            }
            break;
        }
        case 7: {
            const std::uint32_t b = in.word(), k = in.word();
            std::vector<std::uint32_t> lanes(b), positions(b);
            std::vector<std::int32_t> anchors(b);
            for (std::uint32_t j = 0; j < b; ++j) {
                lanes[j]     = in.word();
                anchors[j]   = static_cast<std::int32_t>(in.word());
                positions[j] = in.word();
            }
            std::vector<std::int32_t> drafts(std::size_t(b) * k);
            std::vector<float> confidence(std::size_t(b) * k);
            runtime.draft(lanes, anchors, positions, k, drafts, confidence, stream);
            out.write(reinterpret_cast<const char*>(drafts.data()),
                      static_cast<std::streamsize>(drafts.size() * 4));
            out.write(reinterpret_cast<const char*>(confidence.data()),
                      static_cast<std::streamsize>(confidence.size() * 4));
            break;
        }
        default: throw std::runtime_error("schedule: unknown call");
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::cout << "ran " << calls << " DSpine runtime calls over a " << runtime.window()
              << "-position ring, " << runtime.device_bytes() << " device bytes\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--check-cuda") {
            int count            = 0;
            const cudaError_t e = cudaGetDeviceCount(&count);
            if (e == cudaSuccess) { return count == 0 ? 77 : 0; }
            if (ninfer::test::cuda_unavailable(e)) { return 77; }
            std::cerr << "FAIL: cudaGetDeviceCount: " << cudaGetErrorString(e) << '\n';
            return 1;
        }
        if (argc != 4) {
            std::cerr << "usage: ninfer_dspine_runtime_driver ARTIFACT SCHEDULE OUTPUT "
                         "(tests/models/dspine/runtime_check.py writes them)\n";
            return 2;
        }
        return run(argv[1], argv[2], argv[3]);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
