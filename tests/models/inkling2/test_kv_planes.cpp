// Inkling 2's quantized KV planes (kv_format.h, kv_planes.h) against independent oracles, for
// rk8v4 (int8 G64 keys, packed int4 G32 values) and rk4v4 (Lloyd-Max 4-bit keys):
//  - kv_store: the stored codes, decoded on the host with their stored scales, stay near the F16
//    rows stored (a coding's own error bound);
//  - attention over the planes: a full-attention visit's decode rows, verification rows, prompt
//    chunk and branches against an FP64 attention over the host-decoded rows, the slots being the
//    rows' physical positions.

#include "../../ops/kv_cache_lloyd4_oracle.h"
#include "core/device_buffer.h"
#include "models/inkling2/kernels.h"
#include "models/inkling2/kv_planes.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
namespace k  = ninfer::models::inkling2::kernels;
namespace mi = ninfer::models::inkling2;
using bf16   = __nv_bfloat16;

constexpr int kHeads = 24, kKv = 8, kD = 128, kRel = 16;
constexpr int kExtent = 1024;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

int failures = 0;

void report(bool ok, const std::string& what) {
    std::cout << (ok ? "OK   " : "FAIL ") << what << '\n';
    failures += ok ? 0 : 1;
}

std::string num(double x) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.3g", x);
    return text;
}

template <typename T>
DeviceBuffer upload(const std::vector<T>& values) {
    DeviceBuffer buffer(std::max<std::size_t>(values.size() * sizeof(T), 16));
    if (!values.empty()) { buffer.copy_from_host(values.data(), values.size() * sizeof(T)); }
    return buffer;
}

template <typename T>
std::vector<T> download(const DeviceBuffer& buffer, std::size_t count) {
    std::vector<T> values(count);
    buffer.copy_to_host(values.data(), count * sizeof(T));
    return values;
}

template <typename T>
T* as(DeviceBuffer& buffer) {
    return static_cast<T*>(buffer.p);
}

std::vector<float> normal(std::size_t n, float sigma, std::mt19937& rng) {
    std::normal_distribution<float> d(0.0f, sigma);
    std::vector<float> v(n);
    for (auto& x : v) { x = d(rng); }
    return v;
}

// The RMS of the errors over the RMS of the oracle.
struct Error {
    double num = 0, den = 0;
    void add(double got, double want) {
        num += (got - want) * (got - want);
        den += want * want;
    }
    [[nodiscard]] double rel() const { return den > 0 ? std::sqrt(num / den) : std::sqrt(num); }
};

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// The call's rows (test_kernels.cpp's RowsCase): positions, their sequences and, for a chunk, the
// first physical position and the branches.
struct RowsCase {
    std::vector<int> positions, sequences;
    bool chunk = false;
    int first  = 0, shared = 0;
    std::vector<int> starts;
};

struct RowsDevice {
    DeviceBuffer positions, sequences, starts;
    k::Rows rows;
    explicit RowsDevice(const RowsCase& c)
        : positions(upload(std::vector<std::int32_t>(c.positions.begin(), c.positions.end()))),
          sequences(upload(std::vector<std::int32_t>(c.sequences.begin(), c.sequences.end()))),
          starts(upload(std::vector<std::int32_t>(c.starts.begin(), c.starts.end()))) {
        rows.positions = as<std::int32_t>(positions);
        rows.sequences = c.sequences.empty() ? nullptr : as<std::int32_t>(sequences);
        rows.tokens    = int(c.positions.size());
        rows.chunk     = c.chunk;
        rows.first     = c.first;
        rows.shared    = c.shared;
        rows.starts    = c.starts.empty() ? nullptr : as<std::int32_t>(starts);
    }
};

int sequence_of(const RowsCase& c, int t) { return c.sequences.empty() ? 0 : c.sequences[t]; }

int physical(const RowsCase& c, int t) { return c.chunk ? c.first + t : c.positions[t]; }

double half_at(const std::uint8_t* scales, int group) {
    std::uint16_t bits;
    std::memcpy(&bits, scales + 2 * group, 2);
    __half h;
    std::memcpy(&h, &bits, 2);
    return __half2float(h);
}

// One D128 plane row's represented values from its codes and scales.
std::vector<double> decode_plane_row(mi::KvCoding coding, const std::uint8_t* codes,
                                     const std::uint8_t* scales) {
    std::vector<double> x(kD);
    for (int d = 0; d < kD; ++d) {
        switch (coding) {
        case mi::KvCoding::Int8G64:
            x[d] = double(std::int8_t(codes[d])) * half_at(scales, d / 64);
            break;
        case mi::KvCoding::Int4G32: {
            const int n = d % 2 ? codes[d / 2] >> 4 : codes[d / 2] & 15;
            x[d]        = double((n ^ 8) - 8) * half_at(scales, d / 32);
            break;
        }
        case mi::KvCoding::Lloyd4: {
            const int n = d % 2 ? codes[d / 2] >> 4 : codes[d / 2] & 15;
            x[d]        = double(test::lloyd4_code(std::uint8_t(n))) * half_at(scales, d / 64);
            break;
        }
        case mi::KvCoding::F16: break;
        }
    }
    return x;
}

// kv_store of the rows into the planes, then attention over them. The decoded rows stay near the
// stored halves (decode_bound) and the outputs match the FP64 oracle over the decoded rows
// (attention_bound, wider: the quantization error reaches the output through the softmax).
void case_run(const char* name, mi::KvCoding key, mi::KvCoding value, const RowsCase& rc,
              int splits, double decode_bound, double attention_bound, std::mt19937& rng) {
    const int t = int(rc.positions.size());
    const int sequences =
        rc.sequences.empty() ? 1 : *std::max_element(rc.sequences.begin(), rc.sequences.end()) + 1;
    int top = 0;
    for (int r = 0; r < t; ++r) { top = std::max(top, physical(rc, r)); }
    const int slots = top + 1;
    const int key_code_bytes    = mi::kv_code_bytes(key);
    const int key_scale_bytes   = mi::kv_scale_bytes(key);
    const int value_code_bytes  = mi::kv_code_bytes(value);
    const int value_scale_bytes = mi::kv_scale_bytes(value);
    // The rows kv_post would write: [sequences][heads][slots][128] halves, Gaussian.
    std::vector<__half> kc(std::size_t(sequences) * kKv * slots * kD),
        vc(std::size_t(sequences) * kKv * slots * kD);
    {
        std::normal_distribution<float> n(0.0f, 1.0f);
        for (auto& v : kc) { v = __float2half(1.5f * n(rng)); }
        for (auto& v : vc) { v = __float2half(n(rng)); }
    }
    DeviceBuffer dk = upload(kc), dv = upload(vc),
                 dkc(std::size_t(sequences) * kKv * slots * key_code_bytes),
                 dks(std::size_t(sequences) * kKv * slots * key_scale_bytes),
                 dvc(std::size_t(sequences) * kKv * slots * value_code_bytes),
                 dvs(std::size_t(sequences) * kKv * slots * value_scale_bytes);
    RowsDevice rows_dev(rc);
    const auto plane = [&](DeviceBuffer& codes, DeviceBuffer& scales, int cb, int sb,
                           mi::KvCoding coding) {
        mi::KvPlaneView view;
        view.codes      = static_cast<const std::uint8_t*>(codes.p);
        view.scales     = static_cast<const std::uint8_t*>(scales.p);
        view.code_row   = cb;
        view.scale_row  = sb;
        view.code_head  = std::int64_t(slots) * cb;
        view.scale_head = std::int64_t(slots) * sb;
        view.code_seq   = view.code_head * kKv;
        view.scale_seq  = view.scale_head * kKv;
        view.coding     = coding;
        return view;
    };
    const auto store = [&](const __half* src, DeviceBuffer& codes, DeviceBuffer& scales, int cb,
                           int sb, mi::KvCoding coding) {
        k::KvStoreArgs args;
        args.rows            = rows_dev.rows;
        args.src             = src;
        args.src_head_stride = slots;
        args.rows_count      = t;
        args.heads           = kKv;
        args.dst             = plane(codes, scales, cb, sb, coding);
        k::kv_store(args, nullptr);
    };
    store(as<__half>(dk), dkc, dks, key_code_bytes, key_scale_bytes, key);
    store(as<__half>(dv), dvc, dvs, value_code_bytes, value_scale_bytes, value);
    check(cudaDeviceSynchronize(), "kv_store");

    // The attention's other inputs, as test_kernels.cpp's attention cases take them.
    const auto q     = normal(std::size_t(t) * kHeads * kD, 0.6f, rng);
    const auto r     = normal(std::size_t(t) * kHeads * kRel, 1.0f, rng);
    const auto table = normal(std::size_t(kExtent) * kRel, 1.0f, rng);
    const std::int64_t gate_stride = kHeads * kD + 32, sink_stride = kHeads + 3;
    const auto gate  = normal(std::size_t(t) * gate_stride, 1.0f, rng);
    const auto sinks = normal(std::size_t(t) * sink_stride, 2.0f, rng);
    const float scale = 1.0f / std::sqrt(float(kD));
    DeviceBuffer dq = upload(q), dr = upload(r), dt = upload(table), dg = upload(gate),
                 ds = upload(sinks),
                 partial(k::attention_workspace_floats(t, kHeads, splits) * 4),
                 out(std::size_t(t) * kHeads * kD * 2);
    {
        k::AttentionArgs a;
        a.rows           = rows_dev.rows;
        a.q              = as<float>(dq);
        a.r              = as<float>(dr);
        a.sinks          = as<float>(ds);
        a.sink_stride    = sink_stride;
        a.table          = as<float>(dt);
        a.extent         = kExtent;
        a.k_plane        = plane(dkc, dks, key_code_bytes, key_scale_bytes, key);
        a.v_plane        = plane(dvc, dvs, value_code_bytes, value_scale_bytes, value);
        a.window         = 0;
        a.query_heads    = kHeads;
        a.kv_heads       = kKv;
        a.scale          = scale;
        a.splits         = splits;
        a.partial        = as<float>(partial);
        a.gate           = as<float>(dg);
        a.gate_stride    = gate_stride;
        a.out            = as<bf16>(out);
        k::attention(a, nullptr);
        check(cudaDeviceSynchronize(), "quantized attention");
    }
    const auto got = download<bf16>(out, std::size_t(t) * kHeads * kD);

    // The stored rows back, decoded from their downloaded codes and scales.
    const auto k_codes  = download<std::uint8_t>(dkc, std::size_t(sequences) * kKv * slots * key_code_bytes);
    const auto k_scales = download<std::uint8_t>(dks, std::size_t(sequences) * kKv * slots * key_scale_bytes);
    const auto v_codes  = download<std::uint8_t>(dvc, std::size_t(sequences) * kKv * slots * value_code_bytes);
    const auto v_scales = download<std::uint8_t>(dvs, std::size_t(sequences) * kKv * slots * value_scale_bytes);
    const auto row_at = [&](int s, int head, int slot) {
        return (std::size_t(s) * kKv + std::size_t(head)) * std::size_t(slots) + std::size_t(slot);
    };
    const auto decode = [&](bool is_key, int s, int head, int slot) {
        const std::size_t base = row_at(s, head, slot);
        if (is_key) {
            return decode_plane_row(key, k_codes.data() + base * key_code_bytes,
                                    k_scales.data() + base * key_scale_bytes);
        }
        return decode_plane_row(value, v_codes.data() + base * value_code_bytes,
                                v_scales.data() + base * value_scale_bytes);
    };

    // The store: each stored row's decoded values against its halves.
    Error ek, ev;
    for (int r = 0; r < t; ++r) {
        const int s = sequence_of(rc, r), slot = physical(rc, r);
        for (int h = 0; h < kKv; ++h) {
            const auto kd = decode(true, s, h, slot), vd = decode(false, s, h, slot);
            const std::size_t half_at_rc = (std::size_t(s) * kKv + std::size_t(h)) * slots * kD +
                                           std::size_t(slot) * kD;
            for (int d = 0; d < kD; ++d) {
                ek.add(kd[d], __half2float(kc[half_at_rc + std::size_t(d)]));
                ev.add(vd[d], __half2float(vc[half_at_rc + std::size_t(d)]));
            }
        }
    }
    report(ek.rel() < decode_bound && ev.rel() < decode_bound,
           std::string(name) + " kv_store round-trip: k " + num(ek.rel()) + ", v " + num(ev.rel()) +
               " (bounds " + num(decode_bound) + ")");

    // The attention against the oracle over the decoded rows.
    const int group = kHeads / kKv;
    Error eb;
    int biased = 0;
    for (int row = 0; row < t; ++row) {
        const int s = sequence_of(rc, row), p = rc.positions[row];
        // The keys the row sees: (physical position, logical position).
        std::vector<std::pair<int, int>> keys;
        if (!rc.chunk) {
            for (int j = 0; j <= p; ++j) { keys.emplace_back(j, j); }
        } else {
            for (int j = 0; j < rc.first; ++j) { keys.emplace_back(j, j); }
            for (int o = 0; o <= row; ++o) {
                if (!rc.starts.empty() && o >= rc.shared && rc.starts[o] != rc.starts[row]) {
                    continue;
                }
                keys.emplace_back(rc.first + o, rc.positions[o]);
            }
        }
        for (int h = 0; h < kHeads; ++h) {
            const int kv = h / group;
            const float* qh = q.data() + (std::size_t(row) * kHeads + h) * kD;
            const float* rh = r.data() + (std::size_t(row) * kHeads + h) * kRel;
            std::vector<double> score;
            std::vector<std::vector<double>> vals;
            const double sink = sinks[std::size_t(row) * sink_stride + h];
            double m = sink;
            for (const auto& [phys, logical] : keys) {
                const int distance = p - logical;
                if (distance < 0) { continue; }
                const auto kd = decode(true, s, kv, phys);
                double d = 0;
                for (int i = 0; i < kD; ++i) { d += double(qh[i]) * scale * kd[i]; }
                if (distance < kExtent) {
                    for (int i = 0; i < kRel; ++i) {
                        d += double(rh[i]) * table[std::size_t(distance) * kRel + i];
                    }
                    ++biased;
                }
                score.push_back(d);
                vals.emplace_back(decode(false, s, kv, phys));
                m = std::max(m, d);
            }
            double l = std::exp(sink - m);
            for (auto& v : score) { l += (v = std::exp(v - m)); }
            for (int i = 0; i < kD; ++i) {
                double o = 0;
                for (std::size_t j = 0; j < score.size(); ++j) { o += score[j] * vals[j][i]; }
                o = o / l * sigmoid(gate[std::size_t(row) * gate_stride + h * kD + i]);
                eb.add(__bfloat162float(got[std::size_t(row) * kHeads * kD + h * kD + i]), o);
            }
        }
    }
    report(eb.rel() < attention_bound,
           std::string(name) + " attention (" + std::to_string(biased) + " biased scores): " +
               num(eb.rel()) + " (bound " + num(attention_bound) + ")");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        std::mt19937 rng(75);
        RowsCase decode_rows{{0, 900, 1500}, {0, 1, 2}};
        RowsCase verify{{2000, 2001, 2002, 2003}, {0, 0, 0, 0}};
        RowsCase chunk;
        chunk.chunk = true;
        chunk.first = 600;
        for (int i = 0; i < 21; ++i) { chunk.positions.push_back(600 + i); }
        RowsCase branches;
        branches.chunk  = true;
        branches.first  = 30;
        branches.shared = 4;
        branches.positions = {30, 31, 32, 33, 34, 35, 36, 34, 35};
        branches.starts    = {0, 0, 0, 0, 4, 4, 4, 7, 7};
        for (const auto& [key, value, decode_bound, attention_bound] :
             {std::tuple{mi::KvCoding::Int8G64, mi::KvCoding::Int4G32, 6e-3, 2.5e-2},
              std::tuple{mi::KvCoding::Lloyd4, mi::KvCoding::Int4G32, 2.5e-1, 8.0e-2}}) {
            const char* name = key == mi::KvCoding::Int8G64 ? "rk8v4" : "rk4v4";
            case_run(name, key, value, decode_rows, 1, decode_bound, attention_bound, rng);
            case_run(name, key, value, decode_rows, 9, decode_bound, attention_bound, rng);
            case_run(name, key, value, verify, 4, decode_bound, attention_bound, rng);
            case_run(name, key, value, chunk, 1, decode_bound, attention_bound, rng);
            case_run(name, key, value, branches, 1, decode_bound, attention_bound, rng);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Inkling 2 KV planes\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Inkling 2 KV planes: " << error.what() << '\n';
        return 1;
    }
}
