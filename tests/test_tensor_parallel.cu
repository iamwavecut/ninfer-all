// Qualification of `TensorParallelExchange`, the all-reduce of a tensor-parallel group.
//
// The oracle is the reduction itself evaluated on the host in the same order: each rank's FP32
// partials summed in rank order, added to the BF16 residual in FP32, rounded once. Only adds
// appear, so the host evaluates the same IEEE operations and the comparison is exact. Every rank's
// replica must also equal every other rank's. Ranks share device 0, as in the stage-link test,
// where every transport runs (peer access holds within one card) and words and scatter flags fall
// back to whole reduces; with two or more devices the ranks are also dealt over them, so the
// transports cross cards, and ranks on cards of their own exchange words and scatter flags,
// including parts that end short of 16 bytes and ranks without a range. Each case names one route
// for every size: every transport the devices allow in every form, at a decode column, a verify
// window, an odd size, a size whose scatter ranges end in a partial block or leave some ranks
// without one, and a prompt chunk past PeerWrite's whole reduces. A captured sequence of exchanges
// with changed inputs between replays checks that the protocol holds inside one graph spanning
// every rank's stream and reuses a slot within it. Ranks on two and on four cards then interleave
// bands of different transports and forms over both slots, with all ranks or only rank 0
// contributing, so host and peer words and scatter flags share one epoch, and run words and
// scatter flags across the wrap of their count, widening past what the epochs before the wrap
// stored. A prompt chunk's exchanges then run in pieces on the ranks' side streams, under every
// transport and form, including PeerWrite pieces that outgrow an inbox row and read instead.
// Rounded partials have their own oracle: each partial rounded on the host first (to BF16, or to
// 8-bit blocks by the same correctly rounded divisions), then the same sum; they run whole and in
// pieces under every transport (as reduce-scatters, whole reduces past an inbox row's FP32 values,
// and whole reduces followed by a narrow reduce-scatter piece), between FP32 exchanges in the other
// slot, while the forms and sums that ignore them must give the FP32 oracle. Last, an exchange
// measures its routes and runs on what it chose.
//
// The host writes the inputs with blocking copies from pageable memory, which may return before
// the data reaches the device; the rank streams do not wait for them (they are non-blocking), so
// every device is synchronized before the exchange runs.

#include "core/device.h"
#include "core/tensor_parallel.h"
#include "cuda_availability.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::TensorParallelForm;
using ninfer::TensorParallelPartials;
using ninfer::TensorParallelRoute;
using ninfer::TensorParallelSum;
using ninfer::TensorParallelTransport;
using ninfer::test::cuda_unavailable;

int expect(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << "expectation failed: " << label << '\n';
    return 1;
}

std::uint16_t bits(__nv_bfloat16 value) {
    std::uint16_t out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

struct Case {
    std::size_t ranks;
    std::int64_t values;
    int devices; // ranks are dealt over devices [0, devices)
    TensorParallelRoute route;
};

// The route an exchange of `bytes` whose slot is `bytes` takes when its one band names `asked`:
// words only between ranks on their own cards, up to 32 KiB and not by reading; scatter flags only
// between ranks on their own cards, up to 1 MiB; two ranks storing an FP32 sum reduce whole;
// PeerWrite reduces a partial that fills its slot whole up to 1 MiB and reads past it.
TensorParallelRoute expected_route(TensorParallelRoute asked, std::size_t bytes, std::size_t ranks,
                                   bool own_cards, TensorParallelSum sum) {
    TensorParallelRoute route = asked;
    if (route.form == TensorParallelForm::Words &&
        (route.transport == TensorParallelTransport::PeerRead || !own_cards ||
         bytes > (32U << 10U))) {
        route.form = TensorParallelForm::Whole;
    }
    if (route.form == TensorParallelForm::ScatterFlags && (!own_cards || bytes > (1U << 20U))) {
        route.form = TensorParallelForm::Whole;
    }
    if ((route.form == TensorParallelForm::Scatter ||
         route.form == TensorParallelForm::ScatterFlags) &&
        sum == TensorParallelSum::StoreFp32 && ranks == 2) {
        route.form = TensorParallelForm::Whole;
    }
    if (route.form == TensorParallelForm::Whole &&
        route.transport == TensorParallelTransport::PeerWrite && bytes > (1U << 20U)) {
        route.transport = TensorParallelTransport::PeerRead;
    }
    return route;
}

// One exchange on every rank's stream: ranks [0, contributors) publish their partials of `values`,
// the others nothing, and every rank reduces into `out[r]` and completes it.
template <class Out>
void exchange_all(ninfer::TensorParallelExchange& exchange, ninfer::DeviceContext& ctx,
                  std::size_t slot, const std::vector<Out*>& out, std::int64_t values,
                  std::size_t contributors, TensorParallelSum sum = TensorParallelSum::AddBf16) {
    const auto bytes = static_cast<std::size_t>(values) * sizeof(float);
    for (std::size_t r = 0; r < out.size(); ++r) {
        exchange.publish(r, slot, r < contributors ? bytes : 0, out[r], ctx.rank(r).stream, sum);
    }
    for (std::size_t r = 0; r < out.size(); ++r) {
        exchange.reduce(r, slot, out[r], values, contributors, ctx.rank(r).stream, sum);
    }
    for (std::size_t r = 0; r < out.size(); ++r) {
        exchange.complete(r, slot, out[r], values, ctx.rank(r).stream, sum);
    }
}

// The replica every rank must hold: the residual plus the rank-order sum of the partials.
std::vector<std::uint16_t> oracle(const std::vector<__nv_bfloat16>& residual,
                                  const std::vector<std::vector<float>>& partials) {
    std::vector<std::uint16_t> out(residual.size());
    for (std::size_t i = 0; i < residual.size(); ++i) {
        float sum = partials[0][i];
        for (std::size_t r = 1; r < partials.size(); ++r) { sum += partials[r][i]; }
        out[i] = bits(__float2bfloat16(__bfloat162float(residual[i]) + sum));
    }
    return out;
}

int compare_replicas(const std::vector<__nv_bfloat16*>& replica, std::int64_t values,
                     const std::vector<std::uint16_t>& want, const char* label) {
    int failures = 0;
    for (std::size_t r = 0; r < replica.size(); ++r) {
        std::vector<std::uint16_t> got(want.size());
        CUDA_CHECK(cudaMemcpy(got.data(), replica[r], values * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < want.size(); ++i) { wrong += got[i] != want[i]; }
        if (wrong != 0) {
            std::cerr << "rank " << r << " of " << replica.size() << ", " << values << " values: "
                      << wrong << " differ\n";
        }
        failures += expect(wrong == 0, label);
    }
    return failures;
}

// Rank r on device r % devices, and whether every rank has a card of its own.
std::vector<int> deal(std::size_t ranks, int devices, bool& own_cards) {
    std::vector<int> ids(ranks);
    for (std::size_t r = 0; r < ranks; ++r) { ids[r] = static_cast<int>(r) % devices; }
    own_cards = ranks <= static_cast<std::size_t>(devices);
    return ids;
}

int run_case(const Case& c, std::mt19937& rng) {
    bool own_cards = false;
    const std::vector<int> ids = deal(c.ranks, c.devices, own_cards);
    ninfer::DeviceContext ctx(std::span<const int>(ids.data(), ids.size()));
    if (c.route.transport != TensorParallelTransport::Host && !ctx.peer_access()) { return 0; }
    const auto settle = [&] {
        for (int device = 0; device < c.devices; ++device) {
            CUDA_CHECK(cudaSetDevice(device));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaSetDevice(0));
    };
    const std::size_t bytes = static_cast<std::size_t>(c.values) * sizeof(float);
    ninfer::TensorParallelExchange exchange(ctx, {.slot_bytes = bytes,
                                                  .routes     = {{.bytes = bytes, .route = c.route}}});
    int failures = 0;
    for (const TensorParallelSum sum :
         {TensorParallelSum::AddBf16, TensorParallelSum::StoreBf16, TensorParallelSum::StoreFp32}) {
        failures += expect(exchange.route(bytes, sum) ==
                               expected_route(c.route, bytes, c.ranks, own_cards, sum),
                           "a named route holds where the size, cards and sum allow it");
    }
    failures += expect(exchange.timings().empty(), "named routes are not measured");

    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16*> replica(c.ranks);
    for (std::size_t r = 0; r < c.ranks; ++r) {
        CUDA_CHECK(cudaSetDevice(ids[r]));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&replica[r]),
                              c.values * sizeof(__nv_bfloat16)));
    }
    CUDA_CHECK(cudaSetDevice(0));
    const auto residual_of = [&](float scale) {
        std::vector<__nv_bfloat16> residual(static_cast<std::size_t>(c.values));
        for (auto& v : residual) { v = __float2bfloat16(scale * normal(rng)); }
        return residual;
    };
    const auto partials_of = [&](float scale) {
        std::vector<std::vector<float>> partials(c.ranks,
                                                 std::vector<float>(static_cast<std::size_t>(c.values)));
        for (auto& p : partials) {
            for (auto& v : p) { v = scale * normal(rng); }
        }
        return partials;
    };
    const auto load = [&](const std::vector<__nv_bfloat16>& residual,
                          const std::vector<std::vector<float>>& partials, std::size_t slot) {
        for (std::size_t r = 0; r < c.ranks; ++r) {
            CUDA_CHECK(cudaMemcpy(replica[r], residual.data(), c.values * sizeof(__nv_bfloat16),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(exchange.partial(r, slot), partials[r].data(), bytes,
                                  cudaMemcpyHostToDevice));
        }
        settle();
    };

    // Three exchanges in a row, so both slots are used and the first is reused.
    for (int round = 0; round < 3; ++round) {
        const std::size_t slot = static_cast<std::size_t>(round) % 2;
        const auto residual    = residual_of(4.0f);
        const auto partials    = partials_of(1.0f + static_cast<float>(round));
        load(residual, partials, slot);
        exchange_all(exchange, ctx, slot, replica, c.values, c.ranks);
        ctx.synchronize();
        failures += compare_replicas(replica, c.values, oracle(residual, partials),
                                     "every replica equals the rank-order sum");
    }

    // A projection only rank 0 runs: the others publish nothing and every replica adds rank 0's.
    {
        const auto residual = residual_of(4.0f);
        const auto partials = partials_of(1.0f);
        load(residual, partials, 1);
        exchange_all(exchange, ctx, 1, replica, c.values, 1);
        ctx.synchronize();
        const std::vector<std::vector<float>> first(partials.begin(), partials.begin() + 1);
        failures += compare_replicas(replica, c.values, oracle(residual, first),
                                     "one contributor reaches every replica");
    }

    // A block output: the sum stored in place of the output, rounded once to BF16 or kept in FP32.
    {
        const auto residual = residual_of(4.0f);
        const auto partials = partials_of(1.0f);
        load(residual, partials, 0);
        exchange_all(exchange, ctx, 0, replica, c.values, c.ranks, TensorParallelSum::StoreBf16);
        ctx.synchronize();
        const std::vector<__nv_bfloat16> zero(residual.size(), __float2bfloat16(0.0f));
        failures += compare_replicas(replica, c.values, oracle(zero, partials),
                                     "a stored BF16 sum is the rank-order sum rounded once");

        std::vector<float*> sums(c.ranks);
        for (std::size_t r = 0; r < c.ranks; ++r) {
            CUDA_CHECK(cudaSetDevice(ids[r]));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sums[r]), bytes));
        }
        CUDA_CHECK(cudaSetDevice(0));
        load(residual, partials, 1);
        exchange_all(exchange, ctx, 1, sums, c.values, c.ranks, TensorParallelSum::StoreFp32);
        ctx.synchronize();
        std::vector<float> want(static_cast<std::size_t>(c.values));
        for (std::size_t i = 0; i < want.size(); ++i) {
            float sum = partials[0][i];
            for (std::size_t r = 1; r < c.ranks; ++r) { sum += partials[r][i]; }
            want[i] = sum;
        }
        for (std::size_t r = 0; r < c.ranks; ++r) {
            std::vector<float> got(want.size());
            CUDA_CHECK(cudaMemcpy(got.data(), sums[r], bytes, cudaMemcpyDeviceToHost));
            failures += expect(std::memcmp(got.data(), want.data(), bytes) == 0,
                               "a stored FP32 sum is the rank-order sum");
            CUDA_CHECK(cudaSetDevice(ids[r]));
            CUDA_CHECK(cudaFree(sums[r]));
        }
        CUDA_CHECK(cudaSetDevice(0));
    }

    // A captured pass: three exchanges back to back on every rank's stream, joined into rank 0's;
    // the third reuses the first one's slot and partials.
    {
        cudaStream_t origin = ctx.rank(0).stream;
        std::vector<cudaEvent_t> fork(c.ranks), join(c.ranks);
        // An event is recorded on a stream of its own device.
        for (std::size_t r = 0; r < c.ranks; ++r) {
            CUDA_CHECK(cudaSetDevice(ids[r]));
            CUDA_CHECK(cudaEventCreateWithFlags(&fork[r], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&join[r], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaSetDevice(0));
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
        CUDA_CHECK(cudaEventRecord(fork[0], origin));
        for (std::size_t r = 1; r < c.ranks; ++r) {
            CUDA_CHECK(cudaStreamWaitEvent(ctx.rank(r).stream, fork[0], 0));
        }
        for (std::size_t slot : {0U, 1U, 0U}) {
            exchange_all(exchange, ctx, slot, replica, c.values, c.ranks);
        }
        for (std::size_t r = 1; r < c.ranks; ++r) {
            CUDA_CHECK(cudaEventRecord(join[r], ctx.rank(r).stream));
            CUDA_CHECK(cudaStreamWaitEvent(origin, join[r], 0));
        }
        CUDA_CHECK(cudaStreamEndCapture(origin, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            const auto residual = residual_of(1.0f);
            const auto first    = partials_of(1.0f);
            const auto second   = partials_of(1.0f);
            load(residual, first, 0);
            for (std::size_t r = 0; r < c.ranks; ++r) {
                CUDA_CHECK(cudaMemcpy(exchange.partial(r, 1), second[r].data(), bytes,
                                      cudaMemcpyHostToDevice));
            }
            settle();
            CUDA_CHECK(cudaGraphLaunch(executable, origin));
            CUDA_CHECK(cudaStreamSynchronize(origin));
            std::vector<__nv_bfloat16> after(residual);
            for (const auto* partials : {&first, &second, &first}) {
                const auto next = oracle(after, *partials);
                std::memcpy(after.data(), next.data(), next.size() * sizeof(std::uint16_t));
            }
            std::vector<std::uint16_t> want(after.size());
            std::memcpy(want.data(), after.data(), want.size() * sizeof(std::uint16_t));
            failures +=
                compare_replicas(replica, c.values, want, "a captured pass replays every exchange");
        }
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
        for (std::size_t r = 0; r < c.ranks; ++r) {
            CUDA_CHECK(cudaEventDestroy(fork[r]));
            CUDA_CHECK(cudaEventDestroy(join[r]));
        }
    }

    try {
        exchange.publish(0, 0, bytes + sizeof(float), replica[0], ctx.rank(0).stream);
        failures += expect(false, "an oversized partial is refused");
    } catch (const std::invalid_argument&) {}
    for (std::size_t r = 0; r < c.ranks; ++r) {
        CUDA_CHECK(cudaSetDevice(ids[r]));
        CUDA_CHECK(cudaFree(replica[r]));
    }
    CUDA_CHECK(cudaSetDevice(0));
    return failures;
}

// Exchanges of the sizes in `sizes`, each with its contributor count, on `exchange` over `ctx`'s
// ranks (rank r on device r), against the oracle.
int run_sequence(ninfer::TensorParallelExchange& exchange, ninfer::DeviceContext& ctx,
                 std::span<const std::int64_t> sizes, std::span<const std::size_t> contributors,
                 std::mt19937& rng, const char* label) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const std::size_t ranks = ctx.size();
    std::int64_t widest     = 0;
    for (const std::int64_t values : sizes) { widest = std::max(widest, values); }
    std::vector<__nv_bfloat16*> replica(ranks);
    for (std::size_t r = 0; r < ranks; ++r) {
        CUDA_CHECK(cudaSetDevice(ctx.rank(r).device));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&replica[r]), widest * sizeof(__nv_bfloat16)));
    }
    int failures = 0;
    for (std::size_t step = 0; step < sizes.size(); ++step) {
        const std::int64_t values = sizes[step];
        const std::size_t slot    = step % 2;
        std::vector<__nv_bfloat16> residual(static_cast<std::size_t>(values));
        for (auto& v : residual) { v = __float2bfloat16(4.0f * normal(rng)); }
        std::vector<std::vector<float>> partials(ranks, std::vector<float>(residual.size()));
        for (auto& p : partials) {
            for (auto& v : p) { v = normal(rng); }
        }
        for (std::size_t r = 0; r < ranks; ++r) {
            CUDA_CHECK(cudaSetDevice(ctx.rank(r).device));
            CUDA_CHECK(cudaMemcpy(replica[r], residual.data(), values * sizeof(__nv_bfloat16),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(exchange.partial(r, slot), partials[r].data(),
                                  values * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        exchange_all(exchange, ctx, slot, replica, values, contributors[step]);
        ctx.synchronize();
        partials.resize(contributors[step]);
        failures += compare_replicas(replica, values, oracle(residual, partials), label);
    }
    for (std::size_t r = 0; r < ranks; ++r) {
        CUDA_CHECK(cudaSetDevice(ctx.rank(r).device));
        CUDA_CHECK(cudaFree(replica[r]));
    }
    CUDA_CHECK(cudaSetDevice(0));
    return failures;
}

// A chunk's exchanges in pieces of `widths` values each (see Pieces in core/tensor_parallel.h),
// every piece handed from the rank's stream to its side stream and back by the piece events, as a
// prompt pass does: over both slots and the first again, with every rank contributing and then rank
// 0 alone, under every transport the cards allow in both forms and with a band of scatter flags,
// which a first piece small enough takes and the later pieces reduce whole.
int run_pieces(std::size_t ranks, int devices, std::span<const std::int64_t> widths,
               std::mt19937& rng, const char* label) {
    bool own_cards             = false;
    const std::vector<int> ids = deal(ranks, devices, own_cards);
    ninfer::DeviceContext ctx(std::span<const int>(ids.data(), ids.size()));
    std::int64_t values = 0;
    for (const std::int64_t width : widths) { values += width; }
    const std::size_t bytes = static_cast<std::size_t>(values) * sizeof(float);
    std::vector<TensorParallelTransport> transports{TensorParallelTransport::Host};
    if (ctx.peer_access()) {
        transports.push_back(TensorParallelTransport::PeerRead);
        transports.push_back(TensorParallelTransport::PeerWrite);
    }
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16*> replica(ranks);
    for (std::size_t r = 0; r < ranks; ++r) {
        CUDA_CHECK(cudaSetDevice(ids[r]));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&replica[r]),
                              values * sizeof(__nv_bfloat16)));
    }
    int failures = 0;
    for (const TensorParallelTransport transport : transports) {
        for (const TensorParallelForm form :
             {TensorParallelForm::Whole, TensorParallelForm::Scatter,
              TensorParallelForm::ScatterFlags}) {
            ninfer::TensorParallelExchange exchange(
                ctx, {.slot_bytes = bytes,
                      .routes     = {{.bytes = bytes, .route = {transport, form}}}});
            const std::size_t contributors[] = {ranks, ranks, ranks, 1};
            for (std::size_t round = 0; round < std::size(contributors); ++round) {
                const std::size_t slot = round % 2;
                std::vector<__nv_bfloat16> residual(static_cast<std::size_t>(values));
                for (auto& v : residual) { v = __float2bfloat16(4.0f * normal(rng)); }
                std::vector<std::vector<float>> partials(ranks,
                                                         std::vector<float>(residual.size()));
                for (auto& p : partials) {
                    for (auto& v : p) { v = normal(rng); }
                }
                for (std::size_t r = 0; r < ranks; ++r) {
                    CUDA_CHECK(cudaSetDevice(ids[r]));
                    CUDA_CHECK(cudaMemcpy(replica[r], residual.data(),
                                          values * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(exchange.partial(r, slot), partials[r].data(), bytes,
                                          cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                std::int64_t offset = 0;
                for (std::size_t index = 0; index < widths.size(); ++index) {
                    const std::int64_t count = widths[index];
                    const ninfer::TensorParallelPiece piece{.index = index, .offset = offset};
                    for (std::size_t r = 0; r < ranks; ++r) {
                        CUDA_CHECK(cudaSetDevice(ids[r]));
                        const cudaStream_t side = exchange.side_stream(r);
                        CUDA_CHECK(cudaEventRecord(exchange.piece_ready(r, index),
                                                   ctx.rank(r).stream));
                        CUDA_CHECK(cudaStreamWaitEvent(side, exchange.piece_ready(r, index), 0));
                        exchange.publish(r, slot,
                                         r < contributors[round]
                                             ? static_cast<std::size_t>(count) * sizeof(float)
                                             : 0,
                                         replica[r] + offset, side, TensorParallelSum::AddBf16,
                                         piece);
                    }
                    for (std::size_t r = 0; r < ranks; ++r) {
                        exchange.reduce(r, slot, replica[r] + offset, count, contributors[round],
                                        exchange.side_stream(r), TensorParallelSum::AddBf16,
                                        piece);
                    }
                    for (std::size_t r = 0; r < ranks; ++r) {
                        CUDA_CHECK(cudaSetDevice(ids[r]));
                        exchange.complete(r, slot, replica[r] + offset, count,
                                          exchange.side_stream(r), TensorParallelSum::AddBf16,
                                          piece);
                        CUDA_CHECK(cudaEventRecord(exchange.piece_done(r, index),
                                                   exchange.side_stream(r)));
                    }
                    offset += count;
                }
                for (std::size_t r = 0; r < ranks; ++r) {
                    for (std::size_t index = 0; index < widths.size(); ++index) {
                        CUDA_CHECK(cudaStreamWaitEvent(ctx.rank(r).stream,
                                                       exchange.piece_done(r, index), 0));
                    }
                }
                ctx.synchronize();
                partials.resize(contributors[round]);
                failures += compare_replicas(replica, values, oracle(residual, partials), label);
            }
        }
    }
    for (std::size_t r = 0; r < ranks; ++r) {
        CUDA_CHECK(cudaSetDevice(ids[r]));
        CUDA_CHECK(cudaFree(replica[r]));
    }
    CUDA_CHECK(cudaSetDevice(0));
    return failures;
}

// Each partial rounded and back, as a rounded exchange moves it: to BF16, or to 8-bit blocks (each
// 32 values from the start: scale = largest magnitude / 127, code = value / scale rounded to the
// nearest integer, ties to even; value = code * scale).
std::vector<std::vector<float>> rounded(std::vector<std::vector<float>> partials,
                                        TensorParallelPartials how = TensorParallelPartials::Bf16) {
    for (auto& partial : partials) {
        if (how == TensorParallelPartials::Bf16) {
            for (auto& v : partial) { v = __bfloat162float(__float2bfloat16(v)); }
            continue;
        }
        for (std::size_t block = 0; block < partial.size(); block += 32) {
            const std::size_t end = std::min(partial.size(), block + 32);
            float largest         = 0.0f;
            for (std::size_t i = block; i < end; ++i) {
                largest = std::max(largest, std::fabs(partial[i]));
            }
            const float scale = largest / 127.0f;
            for (std::size_t i = block; i < end; ++i) {
                // An integer code: a value that rounds to a zero code decodes to +0, never -0.
                const int code = largest == 0.0f
                                     ? 0
                                     : static_cast<int>(std::clamp(
                                           std::nearbyint(partial[i] / scale), -127.0f, 127.0f));
                partial[i]     = static_cast<float>(code) * scale;
            }
        }
    }
    return partials;
}

// Rounded partials, `ranks` ranks dealt over `devices`: for each transport the cards allow, form
// and size, a BF16 exchange, an FP32 one in the other slot and an 8-bit one, against the rounded
// oracle where the route rounds (a whole reduce or a reduce-scatter) and the FP32 one where it does
// not (words, scatter flags), and FP32 sums of BF16 and of 8-bit partials; then a chunk in pieces
// under every transport, both ways. An exchange built without rounded copies refuses rounded
// partials on a route that would round them.
int run_rounded(std::size_t ranks, int devices, std::mt19937& rng) {
    bool own_cards             = false;
    const std::vector<int> ids = deal(ranks, devices, own_cards);
    ninfer::DeviceContext ctx(std::span<const int>(ids.data(), ids.size()));
    const std::int64_t sizes[] = {1031, 5120 * 4, 5120 * 64};
    const std::size_t slot_bytes = static_cast<std::size_t>(5120 * 64) * sizeof(float);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16*> replica(ranks);
    std::vector<float*> sums(ranks);
    for (std::size_t r = 0; r < ranks; ++r) {
        CUDA_CHECK(cudaSetDevice(ids[r]));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&replica[r]), slot_bytes / 2));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sums[r]), slot_bytes));
    }
    CUDA_CHECK(cudaSetDevice(0));
    const auto load = [&](ninfer::TensorParallelExchange& exchange, std::size_t slot,
                          std::int64_t values, std::vector<__nv_bfloat16>& residual,
                          std::vector<std::vector<float>>& partials) {
        residual.assign(static_cast<std::size_t>(values), __float2bfloat16(0.0f));
        for (auto& v : residual) { v = __float2bfloat16(4.0f * normal(rng)); }
        partials.assign(ranks, std::vector<float>(residual.size()));
        for (auto& p : partials) {
            for (auto& v : p) { v = normal(rng); }
        }
        for (std::size_t r = 0; r < ranks; ++r) {
            CUDA_CHECK(cudaSetDevice(ids[r]));
            CUDA_CHECK(cudaMemcpy(replica[r], residual.data(), values * sizeof(__nv_bfloat16),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(exchange.partial(r, slot), partials[r].data(),
                                  values * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaSetDevice(0));
    };
    const auto exchange_rounded = [&](ninfer::TensorParallelExchange& exchange, std::size_t slot,
                                      auto out, std::int64_t values, TensorParallelSum sum,
                                      TensorParallelPartials partials,
                                      ninfer::TensorParallelPiece piece = {}) {
        const auto bytes = static_cast<std::size_t>(values) * sizeof(float);
        for (std::size_t r = 0; r < ranks; ++r) {
            exchange.publish(r, slot, bytes, out[r] + piece.offset, ctx.rank(r).stream, sum, piece,
                             partials);
        }
        for (std::size_t r = 0; r < ranks; ++r) {
            exchange.reduce(r, slot, out[r] + piece.offset, values, ranks, ctx.rank(r).stream, sum,
                            piece, partials);
        }
        for (std::size_t r = 0; r < ranks; ++r) {
            exchange.complete(r, slot, out[r] + piece.offset, values, ctx.rank(r).stream, sum,
                              piece, partials);
        }
        ctx.synchronize();
    };
    int failures = 0;
    std::vector<TensorParallelTransport> transports{TensorParallelTransport::Host};
    if (ctx.peer_access()) {
        transports.push_back(TensorParallelTransport::PeerRead);
        transports.push_back(TensorParallelTransport::PeerWrite);
    }
    for (const TensorParallelTransport transport : transports) {
        for (const TensorParallelForm form :
             {TensorParallelForm::Whole, TensorParallelForm::Scatter, TensorParallelForm::Words,
              TensorParallelForm::ScatterFlags}) {
            ninfer::TensorParallelExchange exchange(
                ctx, {.slot_bytes       = slot_bytes,
                      .routes           = {{.bytes = slot_bytes, .route = {transport, form}}},
                      .rounded_partials = TensorParallelPartials::Bf16});
            for (const std::int64_t values : sizes) {
                const std::size_t bytes         = static_cast<std::size_t>(values) * sizeof(float);
                const TensorParallelRoute route = expected_route(
                    {transport, form}, bytes, ranks, own_cards, TensorParallelSum::AddBf16);
                const bool rounds                    = route.form == TensorParallelForm::Whole ||
                                                       route.form == TensorParallelForm::Scatter;
                const TensorParallelPartials asked[] = {TensorParallelPartials::Bf16,
                                                        TensorParallelPartials::Fp32,
                                                        TensorParallelPartials::Int8};
                for (std::size_t round = 0; round < std::size(asked); ++round) {
                    const std::size_t slot = round % 2;
                    std::vector<__nv_bfloat16> residual;
                    std::vector<std::vector<float>> partials;
                    load(exchange, slot, values, residual, partials);
                    exchange_rounded(exchange, slot, replica, values, TensorParallelSum::AddBf16,
                                     asked[round]);
                    const bool rounded_here =
                        rounds && asked[round] != TensorParallelPartials::Fp32;
                    failures += compare_replicas(
                        replica, values,
                        oracle(residual, rounded_here ? rounded(partials, asked[round]) : partials),
                        rounded_here ? "rounded partials match the rounded rank-order sum"
                                     : "partials the route keeps in FP32 match the rank-order sum");
                }
                // An FP32 sum of rounded partials, stored in FP32 (its route may differ from a BF16
                // sum's: two ranks reduce it whole).
                for (const TensorParallelPartials how :
                     {TensorParallelPartials::Bf16, TensorParallelPartials::Int8}) {
                    std::vector<__nv_bfloat16> residual;
                    std::vector<std::vector<float>> partials;
                    load(exchange, 0, values, residual, partials);
                    exchange_rounded(exchange, 0, sums, values, TensorParallelSum::StoreFp32, how);
                    const TensorParallelRoute fp32_route = expected_route(
                        {transport, form}, bytes, ranks, own_cards, TensorParallelSum::StoreFp32);
                    const bool fp32_rounds = fp32_route.form == TensorParallelForm::Whole ||
                                             fp32_route.form == TensorParallelForm::Scatter;
                    const auto moved       = fp32_rounds ? rounded(partials, how) : partials;
                    std::vector<float> want(residual.size());
                    for (std::size_t i = 0; i < want.size(); ++i) {
                        float sum = moved[0][i];
                        for (std::size_t r = 1; r < ranks; ++r) { sum += moved[r][i]; }
                        want[i] = sum;
                    }
                    for (std::size_t r = 0; r < ranks; ++r) {
                        std::vector<float> got(want.size());
                        CUDA_CHECK(cudaMemcpy(got.data(), sums[r], bytes, cudaMemcpyDeviceToHost));
                        const bool same = std::memcmp(got.data(), want.data(), bytes) == 0;
                        failures += expect(
                            same, fp32_rounds ? "an FP32 sum of rounded partials matches"
                                              : "an FP32 sum the route keeps unrounded matches");
                        if (!same) {
                            std::size_t i = 0;
                            while (std::memcmp(&got[i], &want[i], sizeof(float)) == 0) { ++i; }
                            std::cerr << "  transport " << static_cast<int>(transport) << " form "
                                      << static_cast<int>(form) << " expected route "
                                      << static_cast<int>(fp32_route.transport) << '/'
                                      << static_cast<int>(fp32_route.form) << " ranks " << ranks
                                      << " values " << values << " partials "
                                      << static_cast<int>(how) << " rank " << r << ": value " << i
                                      << " is " << got[i] << ", not " << want[i]
                                      << " (unrounded sum " <<
                                [&] {
                                    float sum = partials[0][i];
                                    for (std::size_t k = 1; k < ranks; ++k) {
                                        sum += partials[k][i];
                                    }
                                    return sum;
                                }() << ")\n";
                        }
                    }
                }
            }
        }
    }
    // A chunk in pieces, rounded: four pieces of one slot as a reduce-scatter; whole, the last
    // piece past an inbox row's FP32 values but within its rounded bytes; and whole while the pieces
    // are wide, then a narrow last piece whose band names a reduce-scatter.
    struct PieceCase {
        TensorParallelForm wide;
        TensorParallelForm narrow;
        std::array<std::int64_t, 4> widths;
    };
    const PieceCase piece_cases[] = {
        {TensorParallelForm::Scatter, TensorParallelForm::Scatter,
         {5120 * 16, 5120 * 16, 5120 * 16, 5120 * 16}},
        {TensorParallelForm::Whole, TensorParallelForm::Whole,
         {5120 * 16, 5120 * 16, 5120 * 16, 5120 * 16}},
        {TensorParallelForm::Whole, TensorParallelForm::Scatter,
         {5120 * 20, 5120 * 20, 5120 * 20, 5120 * 4}},
    };
    for (const TensorParallelTransport transport : transports) {
        for (const PieceCase& pieces : piece_cases) {
            for (const TensorParallelPartials how :
                 {TensorParallelPartials::Bf16, TensorParallelPartials::Int8}) {
                const std::size_t narrow_bytes =
                    static_cast<std::size_t>(pieces.widths.back()) * sizeof(float);
                ninfer::TensorParallelExchange exchange(
                    ctx, {.slot_bytes       = slot_bytes,
                          .routes           = {{.bytes = narrow_bytes,
                                                .route = {transport, pieces.narrow}},
                                               {.bytes = slot_bytes,
                                                .route = {transport, pieces.wide}}},
                          .rounded_partials = how});
                const std::int64_t values = 5120 * 64;
                std::vector<__nv_bfloat16> residual;
                std::vector<std::vector<float>> partials;
                load(exchange, 1, values, residual, partials);
                std::int64_t offset = 0;
                for (std::size_t index = 0; index < pieces.widths.size(); ++index) {
                    exchange_rounded(exchange, 1, replica, pieces.widths[index],
                                     TensorParallelSum::AddBf16, how,
                                     ninfer::TensorParallelPiece{.index = index, .offset = offset});
                    offset += pieces.widths[index];
                }
                failures +=
                    compare_replicas(replica, values, oracle(residual, rounded(partials, how)),
                                     "rounded pieces match the rounded rank-order sum");
            }
        }
    }
    // An exchange that measures its routes with rounded partials too: its rounded bands cover the
    // slot, and rounded partials on the routes they chose match the rounded rank-order sum.
    for (const TensorParallelPartials how :
         {TensorParallelPartials::Bf16, TensorParallelPartials::Int8}) {
        ninfer::TensorParallelExchange exchange(
            ctx, {.slot_bytes = slot_bytes, .rounded_partials = how});
        failures += expect(!exchange.rounded_bands().empty() &&
                               exchange.rounded_bands().back().bytes == slot_bytes,
                           "measured rounded bands cover the slot");
        std::cout << exchange.describe() << '\n';
        for (const std::int64_t values : sizes) {
            const auto bytes = static_cast<std::size_t>(values) * sizeof(float);
            const TensorParallelRoute route =
                exchange.route(bytes, TensorParallelSum::AddBf16, how);
            const bool rounds = route.form == TensorParallelForm::Whole ||
                                route.form == TensorParallelForm::Scatter;
            std::vector<__nv_bfloat16> residual;
            std::vector<std::vector<float>> partials;
            load(exchange, 0, values, residual, partials);
            exchange_rounded(exchange, 0, replica, values, TensorParallelSum::AddBf16, how);
            failures += compare_replicas(
                replica, values, oracle(residual, rounds ? rounded(partials, how) : partials),
                "rounded partials on measured routes match the rounded rank-order sum");
        }
    }
    // Without rounded copies a route that would round refuses.
    {
        ninfer::TensorParallelExchange exchange(
            ctx, {.slot_bytes = slot_bytes,
                  .routes     = {{.bytes = slot_bytes,
                                  .route = {TensorParallelTransport::Host,
                                            TensorParallelForm::Whole}}}});
        bool refused = false;
        try {
            exchange.publish(0, 0, slot_bytes, replica[0], ctx.rank(0).stream,
                             TensorParallelSum::AddBf16, {}, TensorParallelPartials::Bf16);
        } catch (const std::logic_error&) { refused = true; }
        failures += expect(refused, "rounded partials need an exchange with rounded copies");
    }
    for (std::size_t r = 0; r < ranks; ++r) {
        CUDA_CHECK(cudaSetDevice(ids[r]));
        CUDA_CHECK(cudaFree(replica[r]));
        CUDA_CHECK(cudaFree(sums[r]));
    }
    CUDA_CHECK(cudaSetDevice(0));
    return failures;
}

// `ranks` ranks on as many cards: bands of different transports and forms interleaved over both
// slots. Host words and scatter flags, and with peer access peer words and scatter flags, take
// turns on one epoch.
int run_mixed(int ranks, std::mt19937& rng) {
    std::vector<int> ids(static_cast<std::size_t>(ranks));
    for (int r = 0; r < ranks; ++r) { ids[static_cast<std::size_t>(r)] = r; }
    ninfer::DeviceContext ctx(std::span<const int>(ids.data(), ids.size()));
    constexpr std::size_t kSlot = 5120 * 4 * sizeof(float);
    using Band                  = ninfer::TensorParallelBand;
    std::vector<Band> routes;
    if (ctx.peer_access()) {
        routes = {
            {16U << 10U, {TensorParallelTransport::Host, TensorParallelForm::Words}},
            {24U << 10U, {TensorParallelTransport::PeerWrite, TensorParallelForm::Words}},
            {32U << 10U, {TensorParallelTransport::PeerRead, TensorParallelForm::ScatterFlags}},
            {40U << 10U, {TensorParallelTransport::Host, TensorParallelForm::ScatterFlags}},
            {48U << 10U, {TensorParallelTransport::PeerWrite, TensorParallelForm::ScatterFlags}},
            {60U << 10U, {TensorParallelTransport::Host, TensorParallelForm::Scatter}},
            {kSlot, {TensorParallelTransport::PeerWrite, TensorParallelForm::Scatter}},
        };
    } else {
        routes = {
            {16U << 10U, {TensorParallelTransport::Host, TensorParallelForm::Words}},
            {32U << 10U, {TensorParallelTransport::Host, TensorParallelForm::ScatterFlags}},
            {40U << 10U, {TensorParallelTransport::Host, TensorParallelForm::Whole}},
            {60U << 10U, {TensorParallelTransport::Host, TensorParallelForm::ScatterFlags}},
            {kSlot, {TensorParallelTransport::Host, TensorParallelForm::Scatter}},
        };
        try {
            ninfer::TensorParallelExchange refused(
                ctx, {.slot_bytes = kSlot, .transport = TensorParallelTransport::PeerWrite});
            return expect(false, "a peer transport without peer access is refused");
        } catch (const std::invalid_argument&) {}
    }
    ninfer::TensorParallelExchange exchange(ctx, {.slot_bytes = kSlot, .routes = routes});
    int failures = 0;
    for (const Band& band : routes) {
        failures += expect(exchange.route(band.bytes) == band.route,
                           "every band's route holds at its size");
    }
    const std::int64_t sizes[]       = {5120,  20480, 1031,  10240, 15360, 20480, 5120,
                                        10240, 1031,  15360, 7168,  12288, 7168,  12288};
    const auto all             = static_cast<std::size_t>(ranks);
    const std::size_t contributors[] = {all, all, 1, 1, all, 1, all, all, 1, all, all, 1, 1, all};
    failures += run_sequence(exchange, ctx, sizes, contributors, rng,
                             "interleaved routes match the rank-order sum");
    return failures;
}

// `ranks` ranks on as many cards exchanging words or scatter flags across the wrap of their count,
// under every such route the cards allow: narrow exchanges up to the wrap, then wider ones, whose
// words and flags past the narrow width no epoch before the wrap stored.
int run_wrap(int ranks, std::mt19937& rng) {
    std::vector<int> ids(static_cast<std::size_t>(ranks));
    for (int r = 0; r < ranks; ++r) { ids[static_cast<std::size_t>(r)] = r; }
    ninfer::DeviceContext ctx(std::span<const int>(ids.data(), ids.size()));
    constexpr std::size_t kSlot = 5120 * sizeof(float);
    std::vector<TensorParallelRoute> routes{
        {TensorParallelTransport::Host, TensorParallelForm::Words},
        {TensorParallelTransport::Host, TensorParallelForm::ScatterFlags}};
    if (ctx.peer_access()) {
        routes.push_back({TensorParallelTransport::PeerWrite, TensorParallelForm::Words});
        routes.push_back({TensorParallelTransport::PeerRead, TensorParallelForm::ScatterFlags});
        routes.push_back({TensorParallelTransport::PeerWrite, TensorParallelForm::ScatterFlags});
    }
    int failures = 0;
    for (const TensorParallelRoute route : routes) {
        ninfer::TensorParallelExchange exchange(
            ctx, {.slot_bytes  = kSlot,
                  .routes      = {{.bytes = kSlot, .route = route}},
                  .words_count = 0xFFFFFFFFU - 3U});
        failures += expect(exchange.route(kSlot) == route,
                           "ranks on their own cards exchange words");
        const std::int64_t sizes[] = {100, 100, 100, 5120, 5120, 100, 5120, 5120};
        const auto all             = static_cast<std::size_t>(ranks);
        const std::size_t contributors[] = {all, all, all, all, 1, all, all, all};
        failures += run_sequence(exchange, ctx, sizes, contributors, rng,
                                 "words across the wrap of their count match the rank-order sum");
    }
    return failures;
}

// An exchange that measures its routes: its bands cover the slot with routes the devices allow, and
// every size runs on what it chose. A forced transport keeps every band to it.
int run_measured(int devices, std::mt19937& rng) {
    bool own_cards = false;
    const std::vector<int> ids = deal(2, devices, own_cards);
    ninfer::DeviceContext ctx(std::span<const int>(ids.data(), ids.size()));
    constexpr std::size_t kSlot = 5120 * 64 * sizeof(float);
    int failures                = 0;
    std::vector<TensorParallelTransport> transports{TensorParallelTransport::Auto,
                                                    TensorParallelTransport::Host};
    if (ctx.peer_access()) { transports.push_back(TensorParallelTransport::PeerWrite); }
    for (const TensorParallelTransport transport : transports) {
        ninfer::TensorParallelExchange exchange(ctx, {.slot_bytes = kSlot, .transport = transport});
        const auto& bands = exchange.bands();
        failures += expect(!bands.empty() && bands.back().bytes == kSlot &&
                               !exchange.whole_bands().empty() && !exchange.timings().empty(),
                           "a measured exchange times its routes and covers its slot");
        for (const auto& band : bands) {
            failures += expect(transport == TensorParallelTransport::Auto ||
                                   band.route.transport == transport,
                               "a forced transport keeps every band");
            failures += expect(band.route.transport == TensorParallelTransport::Host ||
                                   ctx.peer_access(),
                               "a peer route only with peer access");
            failures += expect(band.route.form != TensorParallelForm::Words ||
                                   (own_cards && band.bytes <= (32U << 10U)),
                               "words only between cards, up to their size");
            failures += expect(band.route.form != TensorParallelForm::ScatterFlags ||
                                   (own_cards && band.bytes <= (1U << 20U)),
                               "scatter flags only between cards, up to their size");
        }
        for (const auto& band : exchange.whole_bands()) {
            failures += expect(band.route.form == TensorParallelForm::Whole,
                               "the gather bands reduce whole");
        }
        failures += expect(!exchange.describe().empty(), "a measured exchange describes itself");
        std::cout << exchange.describe() << '\n';
        const std::int64_t sizes[]       = {1031, 5120, 20480, 81920, 327680, 5120, 327680};
        const std::size_t contributors[] = {2, 2, 2, 1, 2, 1, 2};
        failures += run_sequence(exchange, ctx, sizes, contributors, rng,
                                 "measured routes match the rank-order sum");
    }
    return failures;
}

// Cards of one model divide work by one memory bandwidth: a share that followed the noise between
// their measurements would move from start to start and change the sums the ranks split.
int run_bandwidth(int devices) {
    std::vector<int> ids(static_cast<std::size_t>(devices));
    for (int r = 0; r < devices; ++r) { ids[static_cast<std::size_t>(r)] = r; }
    ninfer::DeviceContext ctx(std::span<const int>(ids.data(), ids.size()));
    const std::vector<double> speed = ninfer::rank_copy_bandwidth(ctx);
    int failures = expect(speed.size() == ids.size(), "every rank has a copy bandwidth");
    for (std::size_t a = 0; a < speed.size(); ++a) {
        failures += expect(std::isfinite(speed[a]) && speed[a] > 0.0,
                           "a copy bandwidth is positive and finite");
        for (std::size_t b = a + 1; b < speed.size(); ++b) {
            const cudaDeviceProp& x = ctx.rank(a).props;
            const cudaDeviceProp& y = ctx.rank(b).props;
            const bool one_model = std::strncmp(x.name, y.name, sizeof(x.name)) == 0 &&
                                   x.totalGlobalMem == y.totalGlobalMem &&
                                   x.multiProcessorCount == y.multiProcessorCount &&
                                   x.major == y.major && x.minor == y.minor;
            if (one_model) {
                failures += expect(speed[a] == speed[b], "cards of one model have one bandwidth");
            }
        }
    }
    return failures;
}

int run() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);
    CUDA_CHECK(cudaSetDevice(0));
    std::mt19937 rng(20261025);
    int failures = 0;
    // A decode column of the 27B residual, a verify window, an odd size, ranges of one partial
    // block and ranks without one, and a prompt chunk; words at the sizes they take.
    std::vector<int> device_counts{1};
    for (const int devices : {2, 4}) {
        if (count >= devices) { device_counts.push_back(devices); }
    }
    for (const int devices : device_counts) {
        for (const TensorParallelTransport transport :
             {TensorParallelTransport::Host, TensorParallelTransport::PeerRead,
              TensorParallelTransport::PeerWrite}) {
            for (const std::size_t ranks : {2U, 3U, 4U, 8U}) {
                for (const TensorParallelForm form :
                     {TensorParallelForm::Whole, TensorParallelForm::Scatter}) {
                    for (const std::int64_t values :
                         {5120LL, 5120LL * 4, 1031LL, 100LL, 5120LL * 512}) {
                        failures += run_case({ranks, values, devices, {transport, form}}, rng);
                    }
                }
                for (const std::int64_t values : {5120LL, 1031LL, 100LL}) {
                    failures += run_case(
                        {ranks, values, devices, {transport, TensorParallelForm::Words}}, rng);
                }
                // Scatter flags up to their 1 MiB and past it.
                for (const std::int64_t values : {5120LL, 5120LL * 4, 1031LL, 100LL, 5120LL * 64}) {
                    failures += run_case(
                        {ranks, values, devices, {transport, TensorParallelForm::ScatterFlags}},
                        rng);
                }
            }
        }
        // A prompt chunk of the 27B residual in pieces of 256 columns and a shorter last one, and
        // a narrower chunk whose later pieces outgrow a PeerWrite inbox row.
        const std::int64_t chunk[]  = {5120 * 256, 5120 * 256, 5120 * 256, 5120 * 232};
        const std::int64_t narrow[] = {131072, 131072, 131072, 131072};
        for (const std::size_t ranks : {2U, 4U}) {
            failures += run_pieces(ranks, devices, chunk, rng,
                                   "a chunk exchanged in pieces matches the rank-order sum");
            failures += run_pieces(ranks, devices, narrow, rng,
                                   "pieces past an inbox row match the rank-order sum");
            failures += run_rounded(ranks, devices, rng);
        }
        failures += run_measured(devices, rng);
    }
    for (const int devices : device_counts) {
        if (devices < 2) { continue; }
        failures += run_bandwidth(devices);
        failures += run_mixed(devices, rng);
        failures += run_wrap(devices, rng);
    }
    if (failures != 0) { return 1; }
    std::cout << "tensor-parallel exchange holds\n";
    return 0;
}

} // namespace

int run_tensor_parallel_test() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        return 1;
    }
}
