// Inkling 2's DSpark drafter kernels against FP64 oracles at the release drafter's shapes (hidden
// 2304, 24 query heads over 8 of width 128, a 1024-position window, a block of 8, a recurrent head
// of rank 256):
//  - attention over a uniform block: every row sees the anchor's window (or every position) and
//    the whole block, no sinks, gate or relative bias; window rings and full caches, one and many
//    splits, a block of 8 and of 3 rows; BF16 and q8_1 outputs.
//  - spark_mix: the weighted mixes of four taps (softmax rows) RMS-normed per mix, BF16 and q8_1 rows,
//    the last mix FP32 into the context ring from its first row on, rows past `end` untouched.
//  - spark_rotate: per-head RMS norm and NeoX rotation of queries in place and of keys into F16
//    caches with the values beside them (window ring and full cache), at positions up to 2^20 - 1
//    (the angle formed in FP64).
//  - spark_candidates exactly: each row's 16,384 largest head products by their BF16 truncation,
//    the lowest ids of those equal to the 16,384th, over uneven slices and with ties across the
//    boundary.
//  - spark_step: the recurrent head's joint projection, the soft-capped logits over the head's
//    table for the row's candidates, the first maximum (an exact tie included), its probability
//    over the candidates and the state update, three steps chained; silu_mul and add_rows.
// BF16 and F16 outputs are compared by RMS error relative to the oracle's.

#include "core/device_buffer.h"
#include "models/inkling2/kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace k = ninfer::models::inkling2::kernels;
using bf16  = __nv_bfloat16;

constexpr int kHidden = 2304, kHeads = 24, kKv = 8, kD = 128, kWindow = 1024, kBlock = 8;
constexpr int kRank = 256, kSlots = 32;
constexpr float kEps = 1e-6f, kTheta = 1e6f, kCap = 30.0f;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void sync(const char* what) { check(cudaDeviceSynchronize(), what); }

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

struct Error {
    double num = 0, den = 0;
    void add(double got, double want) {
        num += (got - want) * (got - want);
        den += want * want;
    }
    [[nodiscard]] double rel() const { return den > 0 ? std::sqrt(num / den) : std::sqrt(num); }
};

double q8_value(const std::vector<std::int8_t>& qs, const std::vector<half2>& ds, int k_,
                std::size_t row, int i) {
    return double(qs[row * k_ + i]) * double(__low2float(ds[row * (k_ / 32) + i / 32]));
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// ---- Attention over a uniform block -------------------------------------------------------------

void uniform_case(const std::string& name, int anchor, int rows, int ring, int window, int splits,
                  std::mt19937& rng) {
    const int sequences = 2, sequence = 1;
    const int last = anchor + rows - 1;
    const int slots = ring != 0 ? ring : last + 1;
    const std::int64_t head_stride = std::int64_t(slots) * kD, seq_stride = head_stride * kKv;
    std::vector<__half> kc(std::size_t(sequences) * seq_stride), vc(kc.size());
    {
        std::normal_distribution<float> n(0.0f, 1.0f);
        for (auto& v : kc) { v = __float2half(1.5f * n(rng)); }
        for (auto& v : vc) { v = __float2half(n(rng)); }
    }
    const auto q = normal(std::size_t(rows) * kHeads * kD, 0.6f, rng);
    std::vector<std::int32_t> positions(rows), seqs(rows, sequence);
    for (int i = 0; i < rows; ++i) { positions[i] = anchor + i; }
    const float scale = 1.0f / std::sqrt(float(kD));
    DeviceBuffer dk = upload(kc), dv = upload(vc), dq = upload(q), dp = upload(positions),
                 dsq = upload(seqs), partial(k::attention_workspace_floats(rows, kHeads, splits) * 4),
                 out(std::size_t(rows) * kHeads * kD * 2),
                 q8(std::size_t(rows) * kHeads * kD + std::size_t(rows) * (kHeads * kD / 32) * 4);
    k::AttentionArgs a;
    a.rows.positions = as<std::int32_t>(dp);
    a.rows.sequences = as<std::int32_t>(dsq);
    a.rows.tokens    = rows;
    a.q              = as<float>(dq);
    a.k_cache        = as<__half>(dk);
    a.v_cache        = as<__half>(dv);
    a.seq_stride     = seq_stride;
    a.head_stride    = head_stride;
    a.ring           = ring;
    a.window         = window;
    a.query_heads    = kHeads;
    a.kv_heads       = kKv;
    a.scale          = scale;
    a.splits         = splits;
    a.uniform        = true;
    a.partial        = as<float>(partial);
    a.out            = as<bf16>(out);
    k::attention(a, nullptr);
    sync("uniform attention");
    const auto got = download<bf16>(out, std::size_t(rows) * kHeads * kD);
    a.out      = nullptr;
    a.out_q.qs = as<std::int8_t>(q8);
    a.out_q.ds = reinterpret_cast<half2*>(as<std::uint8_t>(q8) + std::size_t(rows) * kHeads * kD);
    k::attention(a, nullptr);
    sync("uniform attention q8");
    const int width = kHeads * kD;
    std::vector<std::int8_t> got_q(std::size_t(rows) * width);
    std::vector<half2> got_d(std::size_t(rows) * (width / 32));
    check(cudaMemcpy(got_q.data(), a.out_q.qs, got_q.size(), cudaMemcpyDeviceToHost), "q8 qs");
    check(cudaMemcpy(got_d.data(), a.out_q.ds, got_d.size() * 4, cudaMemcpyDeviceToHost), "q8 ds");
    const int lo = window != 0 ? std::max(0, last - window + 1) : 0;
    const int group = kHeads / kKv;
    Error eb, eq;
    for (int row = 0; row < rows; ++row) {
        for (int h = 0; h < kHeads; ++h) {
            const int kv = h / group;
            const float* qh = q.data() + (std::size_t(row) * kHeads + h) * kD;
            std::vector<double> score;
            std::vector<std::int64_t> at;
            double m = -INFINITY;
            for (int p = lo; p <= last; ++p) {
                const int slot = ring != 0 ? p % ring : p;
                const std::int64_t base = sequence * seq_stride + kv * head_stride + std::int64_t(slot) * kD;
                double d = 0;
                for (int i = 0; i < kD; ++i) { d += double(qh[i]) * scale * __half2float(kc[base + i]); }
                score.push_back(d);
                at.push_back(base);
                m = std::max(m, d);
            }
            double l = 0;
            for (auto& v : score) { l += (v = std::exp(v - m)); }
            for (int i = 0; i < kD; ++i) {
                double o = 0;
                for (std::size_t j = 0; j < score.size(); ++j) { o += score[j] * __half2float(vc[at[j] + i]); }
                o /= l;
                const std::size_t idx = std::size_t(row) * width + h * kD + i;
                eb.add(__bfloat162float(got[idx]), o);
                eq.add(double(got_q[idx]) * __low2float(got_d[idx / 32]), o);
            }
        }
    }
    report(eb.rel() < 6e-3 && eq.rel() < 1.5e-2,
           "uniform attention " + name + ", " + std::to_string(rows) + " rows, " +
               std::to_string(last + 1 - lo) + " keys, splits " + std::to_string(splits) +
               ": bf16 " + num(eb.rel()) + ", q8_1 " + num(eq.rel()));
}

// ---- Feature mixes ------------------------------------------------------------------------------

void mix_case(const std::string& name, int tokens, bool q8, int end, int ring_begin, bool chunk,
              std::mt19937& rng) {
    constexpr int kTapCount = 4, kContexts = 3;
    std::vector<std::vector<float>> taps;
    for (int j = 0; j < kTapCount; ++j) { taps.push_back(normal(std::size_t(tokens) * kHidden, 1.0f + j, rng)); }
    // Softmax rows, as llama.cpp's converter stores the release's fusion weights.
    auto mix = normal((kContexts + 1) * kTapCount, 1.5f, rng);
    for (int m = 0; m <= kContexts; ++m) {
        double total = 0;
        for (int j = 0; j < kTapCount; ++j) { total += std::exp(double(mix[m * kTapCount + j])); }
        for (int j = 0; j < kTapCount; ++j) {
            mix[m * kTapCount + j] = float(std::exp(double(mix[m * kTapCount + j])) / total);
        }
    }
    std::vector<std::vector<float>> norms;
    for (int m = 0; m <= kContexts; ++m) {
        auto w = normal(kHidden, 0.3f, rng);
        for (auto& v : w) { v += 1.0f; }
        norms.push_back(w);
    }
    const int sequences = 3;
    std::vector<std::int32_t> positions(tokens), seqs(tokens);
    for (int t = 0; t < tokens; ++t) {
        positions[t] = chunk ? 500 + t : 37 * t + 3;
        seqs[t]      = chunk ? 2 : t % sequences;
    }
    std::vector<DeviceBuffer> d_taps, d_norms;
    for (const auto& t : taps) { d_taps.push_back(upload(t)); }
    for (const auto& n : norms) { d_norms.push_back(upload(n)); }
    DeviceBuffer df = upload(mix), dp = upload(positions), ds = upload(seqs);
    const std::size_t q8_bytes = std::size_t(tokens) * kHidden + std::size_t(tokens) * (kHidden / 32) * 4;
    const std::size_t q8_stride = (q8_bytes + 255) / 256 * 256;
    DeviceBuffer out(std::size_t(kContexts) * tokens * kHidden * 2), outq(kContexts * q8_stride);
    const std::int64_t ring_stride = std::int64_t(kSlots) * kHidden;
    std::vector<float> ring_init(std::size_t(sequences) * ring_stride, 1234.0f);
    DeviceBuffer ring = upload(ring_init);
    k::SparkMixArgs a;
    a.rows.positions = as<std::int32_t>(dp);
    a.rows.sequences = chunk ? nullptr : as<std::int32_t>(ds);
    a.rows.sequence  = 2;
    a.rows.tokens    = tokens;
    a.rows.chunk     = chunk;
    a.rows.first     = chunk ? 500 : 0;
    for (int j = 0; j < kTapCount; ++j) { a.taps[j] = as<float>(d_taps[j]); }
    a.tap_count = kTapCount;
    a.mix       = as<float>(df);
    for (int m = 0; m <= kContexts; ++m) { a.norm[m] = as<float>(d_norms[m]); }
    a.context_count = kContexts;
    a.hidden        = kHidden;
    a.begin         = 0;
    a.end           = end;
    a.ring_begin    = ring_begin;
    a.eps           = kEps;
    if (q8) {
        a.out_q.qs   = as<std::int8_t>(outq);
        a.out_q.ds   = reinterpret_cast<half2*>(as<std::uint8_t>(outq) + std::size_t(tokens) * kHidden);
        a.q8_stride  = std::int64_t(q8_stride);
    } else {
        a.out = as<bf16>(out);
    }
    a.ring       = as<float>(ring);
    a.seq_stride = ring_stride;
    a.slots      = kSlots;
    k::spark_mix(a, nullptr);
    sync("spark_mix");
    const auto got_ring = download<float>(ring, ring_init.size());
    std::vector<bf16> got_b;
    std::vector<std::uint8_t> got_q;
    if (q8) {
        got_q.resize(kContexts * q8_stride);
        check(cudaMemcpy(got_q.data(), outq.p, got_q.size(), cudaMemcpyDeviceToHost), "mix q8");
    } else {
        got_b = download<bf16>(out, std::size_t(kContexts) * tokens * kHidden);
    }
    Error ec, er;
    bool untouched = true;
    std::vector<bool> written(ring_init.size(), false);
    for (int t = 0; t < end; ++t) {
        for (int m = 0; m <= kContexts; ++m) {
            std::vector<double> x(kHidden, 0.0);
            for (int j = 0; j < kTapCount; ++j) {
                const double w = mix[m * kTapCount + j];
                for (int c = 0; c < kHidden; ++c) { x[c] += w * taps[j][std::size_t(t) * kHidden + c]; }
            }
            double ss = 0;
            for (const double v : x) { ss += v * v; }
            const double inv = 1.0 / std::sqrt(ss / kHidden + kEps);
            for (int c = 0; c < kHidden; ++c) {
                const double want = x[c] * inv * norms[m][c];
                if (m < kContexts) {
                    double g = 0;
                    if (q8) {
                        const auto* base = got_q.data() + std::size_t(m) * q8_stride;
                        const auto* qs   = reinterpret_cast<const std::int8_t*>(base);
                        const auto* dsc  = reinterpret_cast<const half2*>(base + std::size_t(tokens) * kHidden);
                        g = double(qs[std::size_t(t) * kHidden + c]) *
                            __low2float(dsc[std::size_t(t) * (kHidden / 32) + c / 32]);
                    } else {
                        g = __bfloat162float(got_b[(std::size_t(m) * tokens + t) * kHidden + c]);
                    }
                    ec.add(g, want);
                } else if (t >= ring_begin) {
                    const int s = chunk ? 2 : seqs[t];
                    const std::size_t at = std::size_t(s) * ring_stride +
                                           std::size_t(positions[t] % kSlots) * kHidden + c;
                    er.add(got_ring[at], want);
                    written[at] = true;
                }
            }
        }
    }
    for (std::size_t i = 0; i < got_ring.size(); ++i) {
        if (!written[i] && got_ring[i] != 1234.0f) { untouched = false; }
    }
    report(ec.rel() < (q8 ? 1.2e-2 : 4e-3) && er.rel() < 1e-5 && untouched,
           "spark_mix " + name + ": contexts " + num(ec.rel()) + ", ring " + num(er.rel()) +
               (untouched ? "" : ", other ring slots changed"));
}

// ---- Norms and rotations ------------------------------------------------------------------------

void rotate_reference(const float* x, const std::vector<float>& norm, int position, double* out) {
    double ss = 0;
    for (int d = 0; d < kD; ++d) { ss += double(x[d]) * x[d]; }
    const double inv = 1.0 / std::sqrt(ss / kD + kEps);
    double y[kD];
    for (int d = 0; d < kD; ++d) { y[d] = x[d] * inv * norm[d]; }
    for (int i = 0; i < kD / 2; ++i) {
        const double angle = double(position) * std::pow(double(kTheta), -2.0 * i / kD);
        const double c = std::cos(angle), s = std::sin(angle);
        out[i]          = y[i] * c - y[i + kD / 2] * s;
        out[i + kD / 2] = y[i + kD / 2] * c + y[i] * s;
    }
}

void rotate_case(std::mt19937& rng) {
    const int t = 8;
    const std::vector<std::int32_t> positions{0, 1, 77, 1023, 1024, 50000, 333, 1048575};
    const std::vector<std::int32_t> seqs{0, 1, 1, 0, 1, 0, 1, 1};
    auto norm = normal(kD, 0.2f, rng);
    for (auto& v : norm) { v += 1.0f; }
    DeviceBuffer dp = upload(positions), ds = upload(seqs), dn = upload(norm);
    // Queries in place, rows of 3200 with Q at column 64.
    const int stride = 3200, column = 64;
    const auto x = normal(std::size_t(t) * stride, 1.3f, rng);
    DeviceBuffer dx = upload(x);
    k::SparkRotateArgs a;
    a.rows.positions = as<std::int32_t>(dp);
    a.rows.sequences = as<std::int32_t>(ds);
    a.rows.tokens    = t;
    a.x      = as<float>(dx);
    a.stride = stride;
    a.column = column;
    a.heads  = kHeads;
    a.begin  = 0;
    a.end    = t;
    a.norm   = as<float>(dn);
    a.eps    = kEps;
    a.theta  = kTheta;
    k::spark_rotate(a, nullptr);
    sync("spark_rotate q");
    const auto got = download<float>(dx, x.size());
    Error eq;
    bool rest = true;
    for (int r = 0; r < t; ++r) {
        for (int h = 0; h < kHeads; ++h) {
            double want[kD];
            rotate_reference(x.data() + std::size_t(r) * stride + column + h * kD, norm, positions[r], want);
            for (int d = 0; d < kD; ++d) {
                eq.add(got[std::size_t(r) * stride + column + h * kD + d], want[d]);
            }
        }
        for (int c = 0; c < stride; ++c) {
            if ((c < column || c >= column + kHeads * kD) &&
                got[std::size_t(r) * stride + c] != x[std::size_t(r) * stride + c]) {
                rest = false;
            }
        }
    }
    report(eq.rel() < 2e-5 && rest, "spark_rotate queries: " + num(eq.rel()) +
                                        (rest ? "" : ", other columns changed"));
    // Keys and values into a window ring and a full cache, rows [2, 7).
    for (const int ring : {kWindow + 32, 0}) {
        const int kv_stride = 2 * kKv * kD;
        const auto kvx = normal(std::size_t(t) * kv_stride, 1.0f, rng);
        DeviceBuffer dkv = upload(kvx);
        const int slots = ring != 0 ? ring : 50001;
        const std::int64_t head_stride = std::int64_t(slots) * kD, seq_stride = head_stride * kKv;
        DeviceBuffer kc(std::size_t(2) * seq_stride * 2), vc(std::size_t(2) * seq_stride * 2);
        kc.fill(0);
        vc.fill(0);
        k::SparkRotateArgs b = a;
        b.x           = as<float>(dkv);
        b.stride      = kv_stride;
        b.column      = 0;
        b.heads       = kKv;
        b.begin       = 2;
        b.end         = 7;
        b.k_cache     = as<__half>(kc);
        b.v_cache     = as<__half>(vc);
        b.seq_stride  = seq_stride;
        b.head_stride = head_stride;
        b.ring        = ring;
        k::spark_rotate(b, nullptr);
        sync("spark_rotate kv");
        Error ek, ev;
        for (int r = b.begin; r < b.end; ++r) {
            const int slot = ring != 0 ? positions[r] % ring : positions[r];
            for (int h = 0; h < kKv; ++h) {
                double want[kD];
                rotate_reference(kvx.data() + std::size_t(r) * kv_stride + h * kD, norm, positions[r], want);
                const std::int64_t at = seqs[r] * seq_stride + h * head_stride + std::int64_t(slot) * kD;
                std::vector<__half> gk(kD), gv(kD);
                check(cudaMemcpy(gk.data(), as<__half>(kc) + at, kD * 2, cudaMemcpyDeviceToHost), "k");
                check(cudaMemcpy(gv.data(), as<__half>(vc) + at, kD * 2, cudaMemcpyDeviceToHost), "v");
                for (int d = 0; d < kD; ++d) {
                    ek.add(__half2float(gk[d]), want[d]);
                    ev.add(__half2float(gv[d]), kvx[std::size_t(r) * kv_stride + (kKv + h) * kD + d]);
                }
            }
        }
        report(ek.rel() < 1e-3 && ev.rel() < 1e-3,
               std::string("spark_rotate keys and values, ") + (ring != 0 ? "window ring" : "full cache") +
                   ": K " + num(ek.rel()) + ", V " + num(ev.rel()));
    }
}

// ---- The recurrent head -------------------------------------------------------------------------

std::uint32_t candidate_key(float v) {
    std::uint32_t u = 0;
    std::memcpy(&u, &v, 4);
    return (u ^ ((u & 0x80000000u) != 0 ? 0xFFFFFFFFu : 0x80000000u)) >> 16;
}

// A row's candidates as spark_candidates orders them: the ids whose key exceeds the
// kSparkCandidates-th largest key ascending, then the lowest ids of those equal to it ascending.
std::vector<std::int32_t> candidates_of(const float* row, int vocab) {
    std::vector<std::uint32_t> keys(vocab);
    for (int v = 0; v < vocab; ++v) { keys[v] = candidate_key(row[v]); }
    std::vector<std::uint32_t> sorted = keys;
    std::nth_element(sorted.begin(), sorted.begin() + (k::kSparkCandidates - 1), sorted.end(),
                     std::greater<>());
    const std::uint32_t threshold = sorted[k::kSparkCandidates - 1];
    std::vector<std::int32_t> out, ties;
    for (int v = 0; v < vocab; ++v) {
        if (keys[v] > threshold) { out.push_back(v); }
        if (keys[v] == threshold) { ties.push_back(v); }
    }
    for (std::size_t i = 0; out.size() < std::size_t(k::kSparkCandidates); ++i) { out.push_back(ties[i]); }
    return out;
}

void candidates_case(const std::string& name, int vocab, int rows, bool boundary_ties, std::mt19937& rng) {
    const std::int64_t stride = vocab + 13;
    auto base = normal(std::size_t(rows) * stride, 3.0f, rng);
    if (boundary_ties) {
        // kSparkCandidates - 124 ids above every other value, then 300 equal by their BF16
        // truncation (half 8.0, half 8.0 plus less than its BF16 step) for the remaining 124 places.
        std::vector<int> ids(vocab);
        for (int v = 0; v < vocab; ++v) { ids[v] = v; }
        for (int r = 0; r < rows; ++r) {
            std::shuffle(ids.begin(), ids.end(), rng);
            float* row = base.data() + std::size_t(r) * stride;
            for (int v = 0; v < vocab; ++v) { row[v] = std::min(row[v], 6.0f); }
            const int above = k::kSparkCandidates - 124;
            for (int i = 0; i < above; ++i) { row[ids[i]] = 10.0f + 0.0001f * float(i % 1000); }
            for (int i = above; i < above + 300; ++i) { row[ids[i]] = i % 2 == 0 ? 8.0f : 8.00001f; }
        }
    }
    DeviceBuffer dbase = upload(base), ids(std::size_t(rows) * k::kSparkCandidates * 4),
                 values(std::size_t(rows) * k::kSparkCandidates * 4),
                 work(k::spark_candidates_bytes(rows));
    k::spark_candidates(as<float>(dbase), stride, vocab, rows, as<std::int32_t>(ids), as<float>(values),
                        work.p, nullptr);
    sync("spark_candidates");
    const auto gi = download<std::int32_t>(ids, std::size_t(rows) * k::kSparkCandidates);
    const auto gv = download<float>(values, std::size_t(rows) * k::kSparkCandidates);
    std::size_t wrong = 0;
    for (int r = 0; r < rows; ++r) {
        const float* row = base.data() + std::size_t(r) * stride;
        const auto want  = candidates_of(row, vocab);
        for (int o = 0; o < k::kSparkCandidates; ++o) {
            const std::size_t at = std::size_t(r) * k::kSparkCandidates + o;
            wrong += gi[at] != want[o] || gv[at] != row[want[o]];
        }
    }
    report(wrong == 0, "spark_candidates " + name + ": " + std::to_string(wrong) + " of " +
                           std::to_string(rows * k::kSparkCandidates) + " candidates differ");
}


void step_case(const std::string& name, int vocab, bool tie, std::mt19937& rng) {
    const int k_ = 2 * kRank + 2 * kHidden, steps = 3;
    const auto weight = normal(std::size_t(3 * kRank) * k_, 0.02f, rng);
    const auto bias   = normal(3 * kRank, 0.3f, rng);
    std::vector<bf16> embed(std::size_t(vocab) * kRank), table(std::size_t(vocab) * kRank);
    {
        std::normal_distribution<float> n(0.0f, 1.0f);
        for (auto& v : embed) { v = __float2bfloat16(n(rng)); }
        for (auto& v : table) { v = __float2bfloat16(tie ? 0.0f : 0.5f * n(rng)); }
    }
    auto c    = normal(vocab, tie ? 0.0f : 1.0f, rng);
    auto base = normal(std::size_t(kBlock) * vocab, 3.0f, rng);
    if (tie) {
        for (int r = 0; r < kBlock; ++r) {
            base[std::size_t(r) * vocab + 300] = 20.0f;
            base[std::size_t(r) * vocab + 7]   = 20.0f;
        }
    }
    const auto hidden = normal(std::size_t(kBlock) * kHidden, 1.0f, rng);
    const int sequences = 2, sequence = 1, anchor_position = 45;
    auto ring = normal(std::size_t(sequences) * kSlots * kHidden, 1.0f, rng);
    std::vector<std::int32_t> positions(kBlock), seqs(kBlock, sequence);
    for (int i = 0; i < kBlock; ++i) { positions[i] = anchor_position + i; }
    const std::int32_t anchor = vocab / 3;
    DeviceBuffer dw = upload(weight), db = upload(bias), de = upload(embed), dt = upload(table),
                 dc = upload(c), dbase = upload(base), dh = upload(hidden), dr = upload(ring),
                 dp = upload(positions), ds = upload(seqs),
                 state(kRank * 4), prev = upload(std::vector<std::int32_t>{anchor}),
                 drafts(kBlock * 4), conf(kBlock * 4), scratch(k::spark_step_floats() * 4),
                 cand_ids(std::size_t(steps) * k::kSparkCandidates * 4),
                 cand_values(std::size_t(steps) * k::kSparkCandidates * 4),
                 cand_work(k::spark_candidates_bytes(steps));
    state.fill(0);
    k::spark_candidates(as<float>(dbase) + vocab, vocab, vocab, steps, as<std::int32_t>(cand_ids),
                        as<float>(cand_values), cand_work.p, nullptr);
    k::SparkStepArgs a;
    a.weight       = as<float>(dw);
    a.bias         = as<float>(db);
    a.embed        = as<bf16>(de);
    a.hidden       = as<float>(dh);
    a.ring         = as<float>(dr);
    a.seq_stride   = std::int64_t(kSlots) * kHidden;
    a.slots        = kSlots;
    a.positions    = as<std::int32_t>(dp);
    a.sequences    = as<std::int32_t>(ds);
    a.candidate_ids    = as<std::int32_t>(cand_ids);
    a.candidate_values = as<float>(cand_values);
    a.m            = as<bf16>(dt);
    a.c            = as<float>(dc);
    a.hidden_size  = kHidden;
    a.cap          = kCap;
    a.state        = as<float>(state);
    a.prev         = as<std::int32_t>(prev);
    a.drafts       = as<std::int32_t>(drafts);
    a.confidence   = as<float>(conf);
    a.joint        = as<float>(scratch);
    a.logits       = a.joint + 3 * kRank;
    for (int row = 1; row <= steps; ++row) {
        a.row = row;
        k::spark_step(a, nullptr);
    }
    sync("spark_step");
    const auto got_drafts = download<std::int32_t>(drafts, steps);
    const auto got_conf   = download<float>(conf, steps);
    const auto got_state  = download<float>(state, kRank);
    // The oracle, fed the kernel's own drafts (a near tie may pick either).
    std::vector<double> s(kRank, 0.0);
    int p = anchor, mismatched = 0, near = 0;
    Error ep, es;
    const float* context = ring.data() + (std::size_t(sequence) * kSlots + (anchor_position - 1) % kSlots) * kHidden;
    for (int row = 1; row <= steps; ++row) {
        std::vector<double> x(k_);
        for (int i = 0; i < kRank; ++i) {
            x[i]         = s[i];
            x[kRank + i] = __bfloat162float(embed[std::size_t(p) * kRank + i]);
        }
        for (int i = 0; i < kHidden; ++i) {
            x[2 * kRank + i]           = hidden[std::size_t(row) * kHidden + i];
            x[2 * kRank + kHidden + i] = context[i];
        }
        std::vector<double> j(3 * kRank);
        for (int o = 0; o < 3 * kRank; ++o) {
            double sum = bias[o];
            for (int i = 0; i < k_; ++i) { sum += double(weight[std::size_t(o) * k_ + i]) * x[i]; }
            j[o] = sum;
        }
        std::vector<double> logits(vocab, -INFINITY);
        double top = -INFINITY;
        int best = 0;
        for (const std::int32_t v : candidates_of(base.data() + std::size_t(row) * vocab, vocab)) {
            double d = 0;
            for (int r = 0; r < kRank; ++r) {
                d += double(__bfloat162float(table[std::size_t(v) * kRank + r])) * std::tanh(j[2 * kRank + r]);
            }
            const double l = kCap * std::tanh((base[std::size_t(row) * vocab + v] + c[v] + d) / kCap);
            logits[v] = l;
            if (l > top || (l == top && v < best)) {
                top  = l;
                best = v;
            }
        }
        double sum = 0;
        for (const double l : logits) { sum += std::exp(l - top); }
        const int got = got_drafts[row - 1];
        if (got != best) {
            if (top - logits[std::size_t(std::clamp(got, 0, vocab - 1))] < 1e-4) {
                ++near;
            } else {
                ++mismatched;
            }
        }
        ep.add(got_conf[row - 1], 1.0 / sum);
        for (int r = 0; r < kRank; ++r) {
            const double gate = sigmoid(j[r]), candidate = std::tanh(j[kRank + r]);
            s[r] = candidate + gate * (s[r] - candidate);
        }
        p = got;
    }
    for (int r = 0; r < kRank; ++r) { es.add(got_state[r], s[r]); }
    const bool tie_ok = !tie || (got_drafts[0] == 7 && got_drafts[1] == 7 && got_drafts[2] == 7);
    report(mismatched == 0 && ep.rel() < 1e-4 && es.rel() < 1e-4 && tie_ok,
           "spark_step " + name + ": " + std::to_string(steps) + " steps, " +
               std::to_string(mismatched) + " drafts off (" + std::to_string(near) +
               " near ties), probability " + num(ep.rel()) + ", state " + num(es.rel()) +
               (tie ? (tie_ok ? ", ties to the lower id" : ", tie broken wrongly") : ""));
}

void elementwise_case(std::mt19937& rng) {
    const int rows = 5, width = 9216;
    const auto gu = normal(std::size_t(rows) * 2 * width, 2.5f, rng);
    DeviceBuffer dgu = upload(gu), out(std::size_t(rows) * width * 2);
    k::silu_mul(as<float>(dgu), width, rows, as<bf16>(out), nullptr);
    const auto h0 = normal(std::size_t(rows) * kHidden, 1.0f, rng);
    const auto y  = normal(std::size_t(rows) * (kHidden + 16), 1.0f, rng);
    DeviceBuffer dh = upload(h0), dy = upload(y);
    k::add_rows(as<float>(dh), as<float>(dy), kHidden + 16, kHidden, rows, nullptr);
    sync("elementwise");
    const auto got = download<bf16>(out, std::size_t(rows) * width);
    const auto gh  = download<float>(dh, h0.size());
    Error es;
    std::size_t wrong = 0;
    for (int t = 0; t < rows; ++t) {
        for (int c = 0; c < width; ++c) {
            const double g = gu[std::size_t(t) * 2 * width + c], u = gu[std::size_t(t) * 2 * width + width + c];
            es.add(__bfloat162float(got[std::size_t(t) * width + c]), g * sigmoid(g) * u);
        }
        for (int c = 0; c < kHidden; ++c) {
            wrong += gh[std::size_t(t) * kHidden + c] !=
                     h0[std::size_t(t) * kHidden + c] + y[std::size_t(t) * (kHidden + 16) + c];
        }
    }
    report(es.rel() < 4e-3 && wrong == 0,
           "silu_mul " + num(es.rel()) + ", add_rows " + std::to_string(wrong) + " values differ");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "no CUDA device; skipped\n";
        return 77;
    }
    std::mt19937 rng(20261121);
    try {
        const int ring = kWindow + 32;
        uniform_case("window ring", 1500, kBlock, ring, kWindow + kBlock, 1, rng);
        uniform_case("window ring", 1500, kBlock, ring, kWindow + kBlock, 4, rng);
        uniform_case("window ring near the start", 300, kBlock, ring, kWindow + kBlock, 4, rng);
        uniform_case("full cache", 2100, kBlock, 0, 0, 1, rng);
        uniform_case("full cache", 2100, kBlock, 0, 0, 20, rng);
        uniform_case("full cache", 40, 3, 0, 0, 20, rng);
        mix_case("decode rows, q8_1", 5, true, 5, 0, false, rng);
        mix_case("prompt rows, bf16", 40, false, 37, 9, true, rng);
        rotate_case(rng);
        candidates_case("vocabulary of 201,088, 7 rows", 201088, 7, false, rng);
        candidates_case("ties across the last candidate, 3 rows", 50001, 3, true, rng);
        candidates_case("every product a candidate", k::kSparkCandidates, 1, false, rng);
        step_case("vocabulary of 20,000", 20000, false, rng);
        step_case("an exact tie", 17000, true, rng);
        elementwise_case(rng);
    } catch (const std::exception& e) {
        std::cout << "FAIL " << e.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "inkling2 DSpark kernels: OK\n" : "inkling2 DSpark kernels: FAILED\n");
    return failures == 0 ? 0 : 1;
}
