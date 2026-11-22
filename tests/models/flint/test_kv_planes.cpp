// Flint's quantized KV planes (kv_format.h) against independent oracles, at Flint's full blocks
// (32 query heads over 4 KV heads of width 128), for int8 (Int8G64 keys and values), rk8v4
// (Int8G64 keys, packed Int4G32 values) and rk4v4 (Lloyd-Max 4-bit keys, rotated; Int4G32
// values):
//  - kv_store of a prompt chunk's rows (slot = first + row, one sequence) and of decode rows (each
//    row's own slot and sequence, in any order): the codes, decoded on the host with their stored
//    scales, stay near the F16 rows they encode (a coding's own error), or near H x / 8 for a
//    rotated plane (H the 128-point Sylvester-Hadamard matrix, in FP64), and every other slot keeps
//    its bytes;
//  - ops::kv128::rotate_queries: H q / 16 against the FP64 product;
//  - kv_load: a range of a sequence's slots back as halves, each equal to the host decode rounded
//    to FP16, and the slots outside the range untouched;
//  - decode_attention over the planes: decode rows of two sequences, a verification's consecutive
//    rows, one and many splits, against an FP64 attention over the host-decoded halves; refused
//    over a ring.

#include "../../ops/kv_cache_lloyd4_oracle.h"
#include "../../ops/kv_row128_hadamard_oracle.h"
#include "core/device_buffer.h"
#include "models/flint/kernels.h"
#include "models/flint/kv_format.h"
#include "ops/kv_cache/row128_rotate.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace ninfer;
namespace k  = ninfer::models::flint::kernels;
namespace mf = ninfer::models::flint;
using mf::KvCoding;

constexpr int kQuery = 32, kKv = 4, kD = 128;

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

// The RMS of the errors over the RMS of the oracle.
struct Error {
    double num = 0, den = 0;
    void add(double got, double want) {
        num += (got - want) * (got - want);
        den += want * want;
    }
    [[nodiscard]] double rel() const { return den > 0 ? std::sqrt(num / den) : std::sqrt(num); }
};

double half_at(const std::uint8_t* scales, int group) {
    std::uint16_t bits;
    std::memcpy(&bits, scales + 2 * group, 2);
    __half h;
    std::memcpy(&h, &bits, 2);
    return __half2float(h);
}

// One plane row's represented values from its codes and stored FP16 scales.
std::vector<double> decode_row(KvCoding coding, const std::uint8_t* codes,
                               const std::uint8_t* scales) {
    std::vector<double> x(kD);
    for (int d = 0; d < kD; ++d) {
        const int nibble = d % 2 ? codes[d / 2] >> 4 : codes[d / 2] & 15;
        switch (coding) {
        case KvCoding::Int8G64:
            x[d] = double(std::int8_t(codes[d])) * half_at(scales, d / 64);
            break;
        case KvCoding::Int4G32: x[d] = double((nibble ^ 8) - 8) * half_at(scales, d / 32); break;
        case KvCoding::Lloyd4:
            x[d] = double(test::lloyd4_code(std::uint8_t(nibble))) * half_at(scales, d / 64);
            break;
        case KvCoding::Mxfp8:
        case KvCoding::F16: throw std::logic_error("not a Flint plane coding");
        }
    }
    return x;
}

// A plane of `sequences` x kKv rows of `slots` slots, on the device, as the executor lays it out.
struct Plane {
    KvCoding coding;
    int sequences, slots;
    bool rotated;
    DeviceBuffer codes, scales;
    Plane(KvCoding c, int s, int n, bool r = false)
        : coding(c), sequences(s), slots(n), rotated(r),
          codes(std::size_t(s) * kKv * n * mf::kv_code_bytes(c)),
          scales(std::size_t(s) * kKv * n * mf::kv_scale_bytes(c)) {}
    [[nodiscard]] std::size_t rows() const { return std::size_t(sequences) * kKv * slots; }
    [[nodiscard]] mf::KvPlaneView view() {
        mf::KvPlaneView v;
        v.codes      = static_cast<std::uint8_t*>(codes.p);
        v.scales     = static_cast<std::uint8_t*>(scales.p);
        v.code_row   = mf::kv_code_bytes(coding);
        v.scale_row  = mf::kv_scale_bytes(coding);
        v.code_head  = std::int64_t(slots) * v.code_row;
        v.scale_head = std::int64_t(slots) * v.scale_row;
        v.code_seq   = v.code_head * kKv;
        v.scale_seq  = v.scale_head * kKv;
        v.coding     = coding;
        v.rotated    = rotated;
        return v;
    }
    [[nodiscard]] std::size_t row(int sequence, int head, int slot) const {
        return (std::size_t(sequence) * kKv + std::size_t(head)) * std::size_t(slots) +
               std::size_t(slot);
    }
};

// Host copies of a plane's bytes and the decode of every row.
struct HostPlane {
    std::vector<std::uint8_t> codes, scales;
    std::vector<double> values; // [row][128]
    HostPlane(const Plane& p)
        : codes(download<std::uint8_t>(p.codes, p.rows() * mf::kv_code_bytes(p.coding))),
          scales(download<std::uint8_t>(p.scales, p.rows() * mf::kv_scale_bytes(p.coding))),
          values(p.rows() * kD) {
        for (std::size_t r = 0; r < p.rows(); ++r) {
            const auto x = decode_row(p.coding, codes.data() + r * mf::kv_code_bytes(p.coding),
                                      scales.data() + r * mf::kv_scale_bytes(p.coding));
            std::copy(x.begin(), x.end(), values.begin() + std::ptrdiff_t(r * kD));
        }
    }
    // The value as the planes' readers see it: the decode rounded to FP16.
    [[nodiscard]] float half(std::size_t row, int d) const {
        return __half2float(__float2half(float(values[row * kD + std::size_t(d)])));
    }
};

std::vector<__half> gaussian_rows(std::size_t n, float sigma, std::mt19937& rng) {
    std::normal_distribution<float> d(0.0f, sigma);
    std::vector<__half> v(n * kD);
    for (auto& x : v) { x = __float2half(d(rng)); }
    return v;
}

// Every slot of sequence 0 from a prompt chunk's staging ([heads][slots][128], rows [first,
// slots) of it), and of sequence 1 from decode rows in shuffled slot order; slots [0, first) of
// sequence 0 and the slots of sequence 1 no row names keep their zero bytes.
struct Stored {
    std::vector<__half> chunk;          // [heads][slots][128]
    std::vector<__half> decode;         // [heads][rows][128], row r at slot decode_slots[r]
    std::vector<std::int32_t> decode_slots;
    int first = 0;
};

Stored store(Plane& plane, float sigma, int first, std::mt19937& rng) {
    Stored s;
    s.first = first;
    s.chunk = gaussian_rows(std::size_t(kKv) * plane.slots, sigma, rng);
    s.decode_slots.resize(std::size_t(plane.slots - 3));
    std::iota(s.decode_slots.begin(), s.decode_slots.end(), 3);
    std::shuffle(s.decode_slots.begin(), s.decode_slots.end(), rng);
    const int rows = int(s.decode_slots.size());
    s.decode = gaussian_rows(std::size_t(kKv) * rows, sigma, rng);
    check(cudaMemset(plane.codes.p, 0, plane.codes.bytes), "codes");
    check(cudaMemset(plane.scales.p, 0, plane.scales.bytes), "scales");
    DeviceBuffer chunk = upload(s.chunk), decode = upload(s.decode),
                 slots = upload(s.decode_slots),
                 sequences = upload(std::vector<std::int32_t>(std::size_t(rows), 1));
    k::KvStoreArgs a;
    a.src             = static_cast<const __half*>(chunk.p);
    a.src_head_stride = std::int64_t(plane.slots) * kD;
    a.src_first       = first;
    a.dst_first       = first;
    a.sequence        = 0;
    a.rows            = plane.slots - first;
    a.heads           = kKv;
    a.dst             = plane.view();
    k::kv_store(a, nullptr);
    a                 = {};
    a.src             = static_cast<const __half*>(decode.p);
    a.src_head_stride = std::int64_t(rows) * kD;
    a.dst_slots       = static_cast<const std::int32_t*>(slots.p);
    a.sequences       = static_cast<const std::int32_t*>(sequences.p);
    a.rows            = rows;
    a.heads           = kKv;
    a.dst             = plane.view();
    k::kv_store(a, nullptr);
    check(cudaDeviceSynchronize(), "kv_store");
    return s;
}

// The stored halves of (sequence, head, slot), or null for a slot no row wrote.
const __half* stored_row(const Stored& s, int slots, int sequence, int head, int slot) {
    if (sequence == 0) {
        return slot < s.first ? nullptr
                              : s.chunk.data() + (std::size_t(head) * slots + slot) * kD;
    }
    const auto at = std::find(s.decode_slots.begin(), s.decode_slots.end(), slot);
    if (at == s.decode_slots.end()) { return nullptr; }
    const std::size_t r = std::size_t(at - s.decode_slots.begin());
    return s.decode.data() + (std::size_t(head) * s.decode_slots.size() + r) * kD;
}

// kv_store's round trip: the decoded rows against the halves they encode (rotated: against
// H x / 8); unwritten rows zero.
void store_case(const char* name, KvCoding coding, bool rotated, double bound, std::mt19937& rng) {
    Plane plane(coding, 2, 300, rotated);
    const Stored s = store(plane, coding == KvCoding::Lloyd4 ? 1.5f : 1.0f, 40, rng);
    const HostPlane host(plane);
    Error e;
    std::size_t untouched = 0, unwritten = 0;
    const int code_bytes = mf::kv_code_bytes(coding), scale_bytes = mf::kv_scale_bytes(coding);
    for (int seq = 0; seq < 2; ++seq) {
        for (int h = 0; h < kKv; ++h) {
            for (int slot = 0; slot < plane.slots; ++slot) {
                const std::size_t row = plane.row(seq, h, slot);
                const __half* want    = stored_row(s, plane.slots, seq, h, slot);
                if (want == nullptr) {
                    ++unwritten;
                    bool zero = true;
                    for (int b = 0; b < code_bytes; ++b) {
                        zero &= host.codes[row * code_bytes + b] == 0;
                    }
                    for (int b = 0; b < scale_bytes; ++b) {
                        zero &= host.scales[row * scale_bytes + b] == 0;
                    }
                    untouched += zero ? 1 : 0;
                    continue;
                }
                std::vector<double> x(kD);
                for (int d = 0; d < kD; ++d) { x[d] = __half2float(want[d]); }
                if (rotated) {
                    x = test::hadamard128(x.data());
                    for (auto& v : x) { v *= 0.125; }
                }
                for (int d = 0; d < kD; ++d) {
                    e.add(host.values[row * kD + std::size_t(d)], x[d]);
                }
            }
        }
    }
    const std::string label = std::string(name) + (rotated ? " rotated" : "");
    report(e.rel() < bound && untouched == unwritten,
           label + " kv_store round trip " + num(e.rel()) + " (bound " + num(bound) +
               "), unwritten rows untouched " + std::to_string(untouched) + "/" +
               std::to_string(unwritten));
}

// kv_load of slots [begin, end) of each sequence: the host decode rounded to FP16, exactly; the
// other slots keep the sentinel.
void load_case(const char* name, KvCoding coding, std::mt19937& rng) {
    Plane plane(coding, 2, 300);
    (void)store(plane, 1.0f, 0, rng);
    const HostPlane host(plane);
    const int begin = 17, end = 251;
    std::size_t wrong = 0, kept = 0;
    for (int seq = 0; seq < 2; ++seq) {
        const std::uint16_t sentinel = 0x7bff; // the largest finite half
        std::vector<std::uint16_t> init(std::size_t(kKv) * plane.slots * kD, sentinel);
        DeviceBuffer dst = upload(init);
        k::KvLoadArgs a;
        a.src             = plane.view();
        a.sequence        = seq;
        a.begin           = begin;
        a.end             = end;
        a.heads           = kKv;
        a.dst             = static_cast<__half*>(dst.p);
        a.dst_head_stride = std::int64_t(plane.slots) * kD;
        k::kv_load(a, nullptr);
        check(cudaDeviceSynchronize(), "kv_load");
        const auto got = download<std::uint16_t>(dst, init.size());
        for (int h = 0; h < kKv; ++h) {
            for (int slot = 0; slot < plane.slots; ++slot) {
                for (int d = 0; d < kD; ++d) {
                    const std::uint16_t bits = got[(std::size_t(h) * plane.slots + slot) * kD + d];
                    if (slot < begin || slot >= end) {
                        kept += bits == sentinel ? 1 : 0;
                        wrong += bits == sentinel ? 0 : 1;
                        continue;
                    }
                    const __half want = __float2half(host.half(plane.row(seq, h, slot), d));
                    std::uint16_t want_bits;
                    std::memcpy(&want_bits, &want, 2);
                    wrong += bits == want_bits ? 0 : 1;
                }
            }
        }
    }
    report(wrong == 0, std::string(name) + " kv_load slots [" + std::to_string(begin) + ", " +
                           std::to_string(end) + ") of 2 sequences: " + std::to_string(wrong) +
                           " halves differ, " + std::to_string(kept) + " outside the range kept");
}

// rotate_queries against the FP64 H q / 16.
void rotate_case(std::mt19937& rng) {
    const int rows = 5;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> q(std::size_t(rows) * kQuery * kD);
    for (auto& v : q) { v = normal(rng); }
    DeviceBuffer dq = upload(q);
    ops::kv128::rotate_queries(static_cast<float*>(dq.p), std::int64_t(rows) * kQuery, nullptr);
    check(cudaDeviceSynchronize(), "rotate_queries");
    const auto got = download<float>(dq, q.size());
    Error e;
    for (std::size_t row = 0; row < std::size_t(rows) * kQuery; ++row) {
        const std::vector<double> x(q.begin() + std::ptrdiff_t(row * kD),
                                    q.begin() + std::ptrdiff_t((row + 1) * kD));
        const auto y = test::hadamard128(x.data());
        for (int d = 0; d < kD; ++d) { e.add(got[row * kD + std::size_t(d)], y[d] / 16.0); }
    }
    report(e.rel() < 1e-6, "rotate_queries H q / 16, " + std::to_string(rows * kQuery) +
                               " rows: rms " + num(e.rel()));
}

// decode_attention over the planes against an FP64 attention over the host-decoded halves.
// With rotated keys the queries go through rotate_queries first, and the oracle takes H q / 16 in
// FP64.
void attention_case(const char* name, KvCoding key, KvCoding value, bool rotated,
                    const std::vector<int>& positions, const std::vector<int>& sequences,
                    int splits, std::mt19937& rng) {
    const int b     = int(positions.size());
    const int slots = *std::max_element(positions.begin(), positions.end()) + 1;
    Plane kp(key, 2, slots, rotated), vp(value, 2, slots);
    (void)store(kp, 1.5f, 0, rng);
    (void)store(vp, 1.0f, 0, rng);
    const HostPlane kh(kp), vh(vp);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> q(std::size_t(b) * kQuery * kD), sinks(kQuery);
    for (auto& v : q) { v = 0.6f * normal(rng); }
    for (auto& v : sinks) { v = 2.0f * normal(rng); }
    const std::vector<std::int32_t> pos(positions.begin(), positions.end()),
        seqs(sequences.begin(), sequences.end());
    DeviceBuffer dq = upload(q), dsink = upload(sinks), dp = upload(pos), ds = upload(seqs),
                 partial(k::decode_attention_workspace_floats(b, kQuery, kD, splits) * 4),
                 out(std::size_t(b) * kQuery * kD * 2);
    k::DecodeAttentionArgs a;
    a.q           = static_cast<const float*>(dq.p);
    a.k_plane     = kp.view();
    a.v_plane     = vp.view();
    a.sinks       = static_cast<const float*>(dsink.p);
    a.sequences   = static_cast<const std::int32_t*>(ds.p);
    a.positions   = static_cast<const std::int32_t*>(dp.p);
    a.batch       = b;
    a.query_heads = kQuery;
    a.kv_heads    = kKv;
    a.head_dim    = kD;
    a.splits      = splits;
    a.scale       = 1.0f / std::sqrt(float(kD));
    a.partial     = static_cast<float*>(partial.p);
    a.out         = static_cast<__nv_bfloat16*>(out.p);
    if (rotated) {
        ops::kv128::rotate_queries(static_cast<float*>(dq.p), std::int64_t(b) * kQuery, nullptr);
        for (std::size_t row = 0; row < std::size_t(b) * kQuery; ++row) {
            const std::vector<double> x(q.begin() + std::ptrdiff_t(row * kD),
                                        q.begin() + std::ptrdiff_t((row + 1) * kD));
            const auto y = test::hadamard128(x.data());
            for (int d = 0; d < kD; ++d) { q[row * kD + std::size_t(d)] = float(y[d] / 16.0); }
        }
    }
    k::decode_attention(a, nullptr);
    check(cudaDeviceSynchronize(), "decode_attention");
    const auto got = download<__nv_bfloat16>(out, std::size_t(b) * kQuery * kD);
    Error e;
    for (int r = 0; r < b; ++r) {
        for (int h = 0; h < kQuery; ++h) {
            const int kv    = h / (kQuery / kKv);
            const float* qh = q.data() + (std::size_t(r) * kQuery + h) * kD;
            std::vector<double> score(std::size_t(positions[r]) + 1);
            double m = sinks[h];
            for (int j = 0; j <= positions[r]; ++j) {
                const std::size_t row = kp.row(sequences[r], kv, j);
                double s              = 0;
                for (int d = 0; d < kD; ++d) { s += double(qh[d]) * kh.half(row, d); }
                score[j] = s * a.scale;
                m        = std::max(m, score[j]);
            }
            double l = std::exp(double(sinks[h]) - m); // the sink: a key without a value
            for (auto& s : score) {
                s = std::exp(s - m);
                l += s;
            }
            for (int d = 0; d < kD; ++d) {
                double o = 0;
                for (int j = 0; j <= positions[r]; ++j) {
                    o += score[j] * vh.half(vp.row(sequences[r], kv, j), d);
                }
                e.add(__bfloat162float(got[(std::size_t(r) * kQuery + h) * kD + d]), o / l);
            }
        }
    }
    const std::string label = std::string(name) + (rotated ? " rotated" : "");
    report(e.rel() < 6e-3, label + " decode_attention rows " + std::to_string(b) + " splits " +
                               std::to_string(splits) + " pos " + std::to_string(slots - 1) +
                               " rms " + num(e.rel()));
}

void ring_refused() {
    Plane kp(KvCoding::Int8G64, 1, 8), vp(KvCoding::Int8G64, 1, 8);
    k::DecodeAttentionArgs a;
    a.k_plane     = kp.view();
    a.v_plane     = vp.view();
    a.batch       = 1;
    a.query_heads = kQuery;
    a.kv_heads    = kKv;
    a.head_dim    = kD;
    a.ring        = 768;
    a.window      = 513;
    a.splits      = 1;
    bool refused  = false;
    try {
        k::decode_attention(a, nullptr);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    report(refused, "decode_attention refuses quantized planes over a ring");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        std::mt19937 rng(1931);
        store_case("Int8G64", KvCoding::Int8G64, false, 8.0e-3, rng);
        store_case("Int4G32", KvCoding::Int4G32, false, 1.1e-1, rng);
        store_case("Lloyd4", KvCoding::Lloyd4, true, 1.2e-1, rng);
        rotate_case(rng);
        load_case("Int8G64", KvCoding::Int8G64, rng);
        load_case("Int4G32", KvCoding::Int4G32, rng);
        load_case("Lloyd4", KvCoding::Lloyd4, rng);
        for (const auto& [name, key, value, rotated] :
             {std::tuple{"int8", KvCoding::Int8G64, KvCoding::Int8G64, false},
              std::tuple{"rk8v4", KvCoding::Int8G64, KvCoding::Int4G32, false},
              std::tuple{"rk4v4", KvCoding::Lloyd4, KvCoding::Int4G32, true}}) {
            attention_case(name, key, value, rotated, {0, 900}, {0, 1}, 1, rng);
            attention_case(name, key, value, rotated, {40, 1999}, {1, 0}, 64, rng);
            attention_case(name, key, value, rotated, {1200, 1201, 1202, 1203}, {0, 0, 0, 0}, 16,
                           rng);
        }
        ring_refused();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Flint KV planes\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Flint KV planes: " << error.what() << '\n';
        return 1;
    }
}
