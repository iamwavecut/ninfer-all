// Times TensorParallelExchange, the all-reduce of tensor parallelism, at the partial sizes of
// decode steps, verifications and prompt chunks. For every rank count the device list allows (2, 4,
// 8 and the whole list), it times every route the devices allow (host memory, and with peer access
// between every pair of cards peer reads and peer writes; words, whole reduces, reduce-scatters and
// scatter flags) at every size, then builds an exchange that measures its own routes as the engine
// does at startup and prints what it measured, the bands it chose and the time of each size on
// them.
//
// Each measurement captures `--exchanges` back-to-back exchanges of every rank's partial on every
// rank's stream into one graph, forked from rank 0's stream and joined back as a pass is, replays
// it `--replays` times and divides the time on rank 0's stream by the exchanges replayed. Partials
// hold ones and replicas zeros: the time does not depend on the values, which
// tests/test_tensor_parallel qualifies. A route a size cannot take (words past 32 KiB, scatter
// flags past 1 MiB or by peer access, either between ranks on one card, a whole PeerWrite reduce
// past its inbox rows, a reduce-scatter of two ranks storing FP32 sums) shows "-".
//
// `--sum` picks what a reduce does with the sum: adds it to a BF16 replica (`add`, the Qwen3.5-core
// residual), or stores it as BF16 (`bf16`) or FP32 (`fp32`, Qwen3.8-Flash-Next's routed experts).
// `--partials` moves the partials as computed (`fp32`) or rounded (`bf16`, `int8`), as
// `--tensor-parallel-prompt-partials` does a prompt chunk's; words and scatter flags keep FP32.
//
// Run: ninfer-tp-exchange-bench --devices 0,1,2,3 [--hidden 5120] [--sum add|bf16|fp32]
//          [--partials fp32|bf16|int8] [--columns 1,2,4,8,13,16,32,64,128,256,512,1024,2048]
//          [--exchanges 16] [--replays 10]
#include "core/device.h"
#include "core/tensor_parallel.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

using ninfer::TensorParallelForm;
using ninfer::TensorParallelRoute;
using ninfer::TensorParallelTransport;

void ck(cudaError_t err, const char* what) {
    if (err == cudaSuccess) { return; }
    std::fprintf(stderr, "%s: %s: %s\n", what, cudaGetErrorName(err), cudaGetErrorString(err));
    std::exit(2);
}

std::vector<long long> parse_list(const std::string& text) {
    std::vector<long long> out;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, ',')) { out.push_back(std::stoll(item)); }
    return out;
}

struct Options {
    std::vector<int> devices;
    long long hidden              = 5120;
    ninfer::TensorParallelSum sum = ninfer::TensorParallelSum::AddBf16;
    ninfer::TensorParallelPartials partials = ninfer::TensorParallelPartials::Fp32;
    std::vector<long long> columns{1, 2, 4, 8, 13, 16, 32, 64, 128, 256, 512, 1024, 2048};
    int exchanges = 16;
    int replays   = 10;
};

// Bytes of one output value.
std::size_t output_bytes(ninfer::TensorParallelSum sum) {
    return sum == ninfer::TensorParallelSum::StoreFp32 ? sizeof(float) : 2;
}

std::string route_name(TensorParallelRoute route) {
    return std::string(ninfer::tensor_parallel_transport_name(route.transport)) + " " +
           ninfer::tensor_parallel_form_name(route.form);
}

// Each rank's replica of the widest partial, with the events that fork and join the ranks' streams.
struct RankBuffers {
    std::vector<void*> replica;
    std::vector<cudaEvent_t> fork;
    std::vector<cudaEvent_t> join;
};

RankBuffers allocate(const ninfer::DeviceContext& ctx, std::size_t replica_bytes) {
    RankBuffers out;
    const std::size_t n = ctx.size();
    out.replica.resize(n);
    out.fork.resize(n);
    out.join.resize(n);
    for (std::size_t r = 0; r < n; ++r) {
        ninfer::RankBinding bind(ctx, r);
        ck(cudaMalloc(&out.replica[r], replica_bytes), "replica");
        ck(cudaMemset(out.replica[r], 0, replica_bytes), "replica zero");
        ck(cudaEventCreateWithFlags(&out.fork[r], cudaEventDisableTiming), "event");
        ck(cudaEventCreateWithFlags(&out.join[r], cudaEventDisableTiming), "event");
    }
    return out;
}

void release(const ninfer::DeviceContext& ctx, RankBuffers& buffers) {
    for (std::size_t r = 0; r < ctx.size(); ++r) {
        ninfer::RankBinding bind(ctx, r);
        (void)cudaFree(buffers.replica[r]);
        (void)cudaEventDestroy(buffers.fork[r]);
        (void)cudaEventDestroy(buffers.join[r]);
    }
}

// Milliseconds of `launch` repeated `repeats` times on rank 0's stream, after as many unmeasured
// runs: after one alone the first small exchange of a new exchange measured up to 1.2 times its
// later runs on a 2x RTX A4000 box.
template <class Launch>
double time_on_origin(const ninfer::DeviceContext& ctx, int repeats, const Launch& launch) {
    ninfer::RankBinding bind(ctx, 0);
    const cudaStream_t origin = ctx.rank(0).stream;
    cudaEvent_t start         = nullptr;
    cudaEvent_t stop          = nullptr;
    ck(cudaEventCreate(&start), "timing event");
    ck(cudaEventCreate(&stop), "timing event");
    for (int i = 0; i < repeats; ++i) { launch(); }
    ck(cudaStreamSynchronize(origin), "warm-up");
    ck(cudaEventRecord(start, origin), "timing record");
    for (int i = 0; i < repeats; ++i) { launch(); }
    ck(cudaEventRecord(stop, origin), "timing record");
    ck(cudaEventSynchronize(stop), "timing synchronize");
    float ms = 0.0F;
    ck(cudaEventElapsedTime(&ms, start, stop), "timing elapsed");
    (void)cudaEventDestroy(start);
    (void)cudaEventDestroy(stop);
    return ms;
}

// Microseconds per exchange of `values` per partial, every rank contributing, in a captured graph.
double time_exchanges(ninfer::TensorParallelExchange& exchange, const ninfer::DeviceContext& ctx,
                      RankBuffers& buffers, std::int64_t values, const Options& options) {
    const std::size_t n       = ctx.size();
    const cudaStream_t origin = ctx.rank(0).stream;
    const auto bytes          = static_cast<std::size_t>(values) * sizeof(float);
    ninfer::RankBinding bind(ctx, 0);
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    ck(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal), "capture");
    ck(cudaEventRecord(buffers.fork[0], origin), "fork record");
    for (std::size_t r = 1; r < n; ++r) {
        ck(cudaStreamWaitEvent(ctx.rank(r).stream, buffers.fork[0], 0), "fork wait");
    }
    for (int e = 0; e < options.exchanges; ++e) {
        const auto slot = static_cast<std::size_t>(e) % ninfer::TensorParallelExchange::kSlots;
        for (std::size_t r = 0; r < n; ++r) {
            exchange.publish(r, slot, bytes, buffers.replica[r], ctx.rank(r).stream, options.sum,
                             {}, options.partials);
        }
        for (std::size_t r = 0; r < n; ++r) {
            exchange.reduce(r, slot, buffers.replica[r], values, n, ctx.rank(r).stream, options.sum,
                            {}, options.partials);
        }
        for (std::size_t r = 0; r < n; ++r) {
            exchange.complete(r, slot, buffers.replica[r], values, ctx.rank(r).stream, options.sum,
                              {}, options.partials);
        }
    }
    for (std::size_t r = 1; r < n; ++r) {
        ck(cudaEventRecord(buffers.join[r], ctx.rank(r).stream), "join record");
        ck(cudaStreamWaitEvent(origin, buffers.join[r], 0), "join wait");
    }
    ck(cudaStreamEndCapture(origin, &graph), "end capture");
    ck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate");
    const double ms = time_on_origin(
        ctx, options.replays, [&] { ck(cudaGraphLaunch(executable, origin), "graph launch"); });
    (void)cudaGraphExecDestroy(executable);
    (void)cudaGraphDestroy(graph);
    return ms * 1000.0 / (static_cast<double>(options.replays) * options.exchanges);
}

// Ones in every partial, so an 8-bit rounding divides as it does real values.
void fill_partials(ninfer::TensorParallelExchange& exchange, const ninfer::DeviceContext& ctx,
                   std::size_t slot_bytes) {
    const std::vector<float> ones(slot_bytes / sizeof(float), 1.0F);
    for (std::size_t r = 0; r < ctx.size(); ++r) {
        ninfer::RankBinding bind(ctx, r);
        for (std::size_t slot = 0; slot < ninfer::TensorParallelExchange::kSlots; ++slot) {
            ck(cudaMemcpy(exchange.partial(r, slot), ones.data(), slot_bytes,
                          cudaMemcpyHostToDevice),
               "partial fill");
        }
    }
}

void run_group(const std::vector<int>& ids, const Options& options) {
    ninfer::DeviceContext ctx(std::span<const int>(ids.data(), ids.size()));
    const std::size_t n    = ctx.size();
    const long long widest = *std::max_element(options.columns.begin(), options.columns.end());
    const std::size_t slot_bytes = static_cast<std::size_t>(widest * options.hidden) * 4;
    RankBuffers buffers          = allocate(ctx, slot_bytes / 4 * output_bytes(options.sum));
    std::string list;
    for (const int id : ids) { list += (list.empty() ? "" : ",") + std::to_string(id); }
    const char* sum = options.sum == ninfer::TensorParallelSum::AddBf16     ? "add"
                      : options.sum == ninfer::TensorParallelSum::StoreBf16 ? "bf16"
                                                                            : "fp32";
    const char* partials = options.partials == ninfer::TensorParallelPartials::Bf16   ? "bf16"
                           : options.partials == ninfer::TensorParallelPartials::Int8 ? "int8"
                                                                                      : "fp32";
    std::printf("\ndevices %s, %zu ranks, %s peer access, hidden %lld, sum %s, partials %s\n",
                list.c_str(), n, ctx.peer_access() ? "with" : "without", options.hidden, sum,
                partials);

    std::vector<TensorParallelTransport> transports{TensorParallelTransport::Host};
    if (ctx.peer_access()) {
        transports.push_back(TensorParallelTransport::PeerRead);
        transports.push_back(TensorParallelTransport::PeerWrite);
    }
    // Microseconds of every route at every size; negative where the size cannot take it.
    std::vector<TensorParallelRoute> routes;
    std::vector<std::vector<double>> times;
    for (const TensorParallelTransport transport : transports) {
        for (const TensorParallelForm form :
             {TensorParallelForm::Words, TensorParallelForm::Whole, TensorParallelForm::Scatter,
              TensorParallelForm::ScatterFlags}) {
            const TensorParallelRoute route{transport, form};
            ninfer::TensorParallelExchange exchange(
                ctx, {.slot_bytes       = slot_bytes,
                      .routes           = {{.bytes = slot_bytes, .route = route}},
                      .rounded_partials = options.partials});
            fill_partials(exchange, ctx, slot_bytes);
            std::vector<double> row;
            bool any = false;
            for (const long long columns : options.columns) {
                const auto values = static_cast<std::int64_t>(columns * options.hidden);
                const auto bytes  = static_cast<std::size_t>(values) * sizeof(float);
                if (exchange.route(bytes, options.sum, options.partials) != route) {
                    row.push_back(-1.0);
                    continue;
                }
                row.push_back(time_exchanges(exchange, ctx, buffers, values, options));
                any = true;
            }
            if (any) {
                routes.push_back(route);
                times.push_back(row);
            }
        }
    }

    ninfer::TensorParallelExchange automatic(
        ctx, {.slot_bytes = slot_bytes, .rounded_partials = options.partials});
    fill_partials(automatic, ctx, slot_bytes);
    std::printf("%s\n", automatic.describe().c_str());
    // What the exchange measured at startup, by probe size, in the columns of the table below.
    std::printf("%18s", "startup probe");
    for (const TensorParallelRoute& route : routes) {
        std::printf(" %18s", route_name(route).c_str());
    }
    std::printf("\n");
    const auto& timings = automatic.timings();
    for (std::size_t i = 0; i < timings.size();) {
        const std::size_t bytes = timings[i].bytes;
        std::printf("%17.0fK", static_cast<double>(bytes) / 1024.0);
        for (const TensorParallelRoute& route : routes) {
            const auto found =
                std::find_if(timings.begin() + static_cast<std::ptrdiff_t>(i), timings.end(),
                             [&](const auto& band) {
                                 return band.bytes == bytes && band.route == route;
                             });
            if (found == timings.end()) {
                std::printf(" %18s", "-");
            } else {
                std::printf(" %18.1f", found->microseconds);
            }
        }
        std::printf("\n");
        while (i < timings.size() && timings[i].bytes == bytes) { ++i; }
    }
    std::printf("%8s %9s", "columns", "partial");
    for (const TensorParallelRoute& route : routes) {
        std::printf(" %18s", route_name(route).c_str());
    }
    std::printf(" %20s %10s %8s\n", "auto", "auto_us", "vs_best");
    for (std::size_t i = 0; i < options.columns.size(); ++i) {
        const auto values = static_cast<std::int64_t>(options.columns[i] * options.hidden);
        const auto bytes  = static_cast<std::size_t>(values) * sizeof(float);
        std::printf("%8lld %8.0fK", options.columns[i], static_cast<double>(bytes) / 1024.0);
        double best = -1.0;
        for (const auto& row : times) {
            if (row[i] < 0.0) {
                std::printf(" %18s", "-");
                continue;
            }
            std::printf(" %18.1f", row[i]);
            best = best < 0.0 ? row[i] : std::min(best, row[i]);
        }
        const double chosen = time_exchanges(automatic, ctx, buffers, values, options);
        std::printf(" %20s %10.1f %8.3f\n",
                    route_name(automatic.route(bytes, options.sum, options.partials)).c_str(),
                    chosen, chosen / best);
    }
    release(ctx, buffers);
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_value  = i + 1 < argc;
        if (arg == "--devices" && has_value) {
            for (const long long id : parse_list(argv[++i])) {
                options.devices.push_back(static_cast<int>(id));
            }
        } else if (arg == "--hidden" && has_value) {
            options.hidden = std::stoll(argv[++i]);
        } else if (arg == "--columns" && has_value) {
            options.columns = parse_list(argv[++i]);
        } else if (arg == "--exchanges" && has_value) {
            options.exchanges = std::max(2, std::stoi(argv[++i]) / 2 * 2);
        } else if (arg == "--replays" && has_value) {
            options.replays = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--sum" && has_value && std::string(argv[i + 1]) == "add") {
            options.sum = ninfer::TensorParallelSum::AddBf16;
            ++i;
        } else if (arg == "--sum" && has_value && std::string(argv[i + 1]) == "bf16") {
            options.sum = ninfer::TensorParallelSum::StoreBf16;
            ++i;
        } else if (arg == "--sum" && has_value && std::string(argv[i + 1]) == "fp32") {
            options.sum = ninfer::TensorParallelSum::StoreFp32;
            ++i;
        } else if (arg == "--partials" && has_value && std::string(argv[i + 1]) == "fp32") {
            options.partials = ninfer::TensorParallelPartials::Fp32;
            ++i;
        } else if (arg == "--partials" && has_value && std::string(argv[i + 1]) == "bf16") {
            options.partials = ninfer::TensorParallelPartials::Bf16;
            ++i;
        } else if (arg == "--partials" && has_value && std::string(argv[i + 1]) == "int8") {
            options.partials = ninfer::TensorParallelPartials::Int8;
            ++i;
        } else {
            std::fprintf(stderr,
                         "usage: %s --devices 0,1[,...] [--hidden N] [--sum add|bf16|fp32] "
                         "[--partials fp32|bf16|int8] [--columns 1,2,...] [--exchanges N] "
                         "[--replays N]\n",
                         argv[0]);
            return 2;
        }
    }
    if (options.devices.size() < 2 || options.devices.size() > ninfer::kMaxRanks ||
        options.columns.empty() || options.hidden <= 0) {
        std::fprintf(stderr, "need 2 to %zu devices, columns and a positive hidden size\n",
                     ninfer::kMaxRanks);
        return 2;
    }
    try {
        std::vector<std::size_t> counts;
        for (const std::size_t count : {std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
            if (count <= options.devices.size()) { counts.push_back(count); }
        }
        if (counts.back() != options.devices.size()) { counts.push_back(options.devices.size()); }
        for (const std::size_t count : counts) {
            run_group(std::vector<int>(options.devices.begin(), options.devices.begin() + count),
                      options);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
