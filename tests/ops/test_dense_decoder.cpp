#include "ninfer/ops/dense_decoder.h"
#include "core/arena.h"
#include "core/device.h"
#include "cuda_availability.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
using namespace ninfer;

namespace {
std::mt19937 rng(731);

float random_value() { return std::uniform_real_distribution<float>(-1.5f, 1.5f)(rng); }

template <class T>
DeviceBuffer upload(const std::vector<T>& x) {
    DeviceBuffer b(x.size() * sizeof(T));
    b.copy_from_host(x.data(), b.bytes);
    return b;
}

void criterion(const char* label, const std::vector<double>& ideal,
               const std::vector<float>& actual, double rms_limit, double max_limit) {
    double error = 0, energy = 0, peak = 0, largest = 0;
    for (std::size_t i = 0; i < ideal.size(); ++i) {
        double e = actual[i] - ideal[i];
        if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite output");
        error += e * e;
        energy += ideal[i] * ideal[i];
        peak    = std::max(peak, std::abs(ideal[i]));
        largest = std::max(largest, std::abs(e));
    }
    double rms = std::sqrt(error / std::max(energy, 1e-30));
    std::cout << label << " nrmse=" << rms << " max_abs=" << largest << '\n';
    if (rms > rms_limit || largest > max_limit * std::max(1.0, peak))
        throw std::runtime_error(label);
}

void rms() {
    int h = 7168, t = 128;
    std::vector<float> x(h * t), w(h);
    for (auto& v : x) v = random_value();
    for (auto& v : w) v = random_value();
    auto dx = upload(x), dw = upload(w);
    DeviceBuffer out(x.size() * 2);
    ops::rmsnorm_f32((float*)dx.p, (float*)dw.p, h, t, 1e-5f, (__nv_bfloat16*)out.p, 0);
    std::vector<__nv_bfloat16> b(x.size());
    out.copy_to_host(b.data(), out.bytes);
    std::vector<double> ideal(x.size());
    std::vector<float> actual(x.size());
    for (int row = 0; row < t; ++row) {
        double s = 0;
        for (int d = 0; d < h; ++d) s += double(x[row * h + d]) * x[row * h + d];
        double inv = 1 / std::sqrt(s / h + double(1e-5f));
        for (int d = 0; d < h; ++d) {
            int i     = row * h + d;
            ideal[i]  = x[i] * inv * w[d];
            actual[i] = __bfloat162float(b[i]);
        }
    }
    criterion("rms7168", ideal, actual, .004, .02);
}

void rotary() {
    int t = 4, hq = 56, hk = 8, stride = (hq + 2 * hk) * 128;
    float theta = 120000000.f;
    std::vector<float> x(t * stride);
    for (auto& v : x) v = random_value();
    std::vector<int> positions{0, 1, 1023, 262143};
    auto dx = upload(x), dp = upload(positions);
    DeviceBuffer q(t * hq * 128 * 4), k(t * hk * 128 * 2), v(t * hk * 128 * 2);
    ops::rotary_kv_f16((float*)dx.p, (int*)dp.p, t, hq, hk, theta, (float*)q.p, (__half*)k.p,
                       (__half*)v.p, 0);
    std::vector<float> aq(t * hq * 128);
    std::vector<__half> ak(t * hk * 128), av(ak.size());
    q.copy_to_host(aq.data(), q.bytes);
    k.copy_to_host(ak.data(), k.bytes);
    v.copy_to_host(av.data(), v.bytes);
    std::vector<double> iq(aq.size()), ik(ak.size()), iv(av.size());
    std::vector<float> fk(ak.size()), fv(av.size());
    for (int row = 0; row < t; ++row)
        for (int head = 0; head < hq + hk; ++head)
            for (int d = 0; d < 64; ++d) {
                int in   = row * stride + head * 128;
                double a = x[in + d], b = x[in + d + 64],
                       angle = positions[row] * std::pow(double(theta), -double(d) / 64);
                double lo    = a * std::cos(angle) - b * std::sin(angle),
                       hi    = a * std::sin(angle) + b * std::cos(angle);
                auto& dest   = head < hq ? iq : ik;
                int base     = head < hq ? (row * hq + head) * 128 : ((head - hq) * t + row) * 128;
                dest[base + d]      = lo;
                dest[base + d + 64] = hi;
            }
    for (int row = 0; row < t; ++row)
        for (int head = 0; head < hk; ++head)
            for (int d = 0; d < 128; ++d) {
                int i = (head * t + row) * 128 + d;
                iv[i] = x[row * stride + (hq + hk + head) * 128 + d];
                fk[i] = __half2float(ak[i]);
                fv[i] = __half2float(av[i]);
            }
    criterion("rotary-q", iq, aq, 2e-5, 2e-5);
    criterion("rotary-k-f16", ik, fk, .001, .001);
    criterion("value-f16", iv, fv, .001, .001);
}

void attention(int t, int n, int first) {
    int hq = 56, hk = 8;
    std::vector<float> q(t * hq * 128);
    for (auto& v : q) v = random_value();
    std::vector<__half> k(hk * n * 128), v(k.size());
    for (auto& z : k) z = __float2half(random_value());
    for (auto& z : v) z = __float2half(random_value());
    std::vector<int> pos(t);
    for (int i = 0; i < t; ++i) pos[i] = first + std::min(n - 1, n / 2 + i);
    auto dq = upload(q), dk = upload(k), dv = upload(v), dp = upload(pos);
    DeviceBuffer state(std::size_t(t) * hq * 130 * 4),
        work(ops::segment_attention_workspace_bytes(t, hq, n)), out(q.size() * 2);
    ops::segment_attention_f16((float*)dq.p, (__half*)dk.p, (__half*)dv.p, (int*)dp.p, t, hq, hk,
                               first, n, n, (float*)state.p, work.p, true, true,
                               (__nv_bfloat16*)out.p, 0);
    std::vector<__nv_bfloat16> actual_b(q.size());
    out.copy_to_host(actual_b.data(), out.bytes);
    std::vector<float> actual(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) actual[i] = __bfloat162float(actual_b[i]);
    std::vector<double> ideal(q.size());
    for (int row = 0; row < t; ++row)
        for (int head = 0; head < hq; ++head) {
            int kv = head / 7, visible = pos[row] - first + 1;
            std::vector<double> scores(visible);
            double max = -INFINITY;
            for (int j = 0; j < visible; ++j) {
                double dot = 0;
                for (int d = 0; d < 128; ++d)
                    dot += double(q[(row * hq + head) * 128 + d]) *
                           __half2float(k[(kv * n + j) * 128 + d]);
                scores[j] = dot / std::sqrt(128.0);
                max       = std::max(max, scores[j]);
            }
            double sum = 0;
            for (auto& s : scores) {
                s = std::exp(s - max);
                sum += s;
            }
            for (int d = 0; d < 128; ++d) {
                double a = 0;
                for (int j = 0; j < visible; ++j)
                    a += scores[j] * __half2float(v[(kv * n + j) * 128 + d]);
                ideal[(row * hq + head) * 128 + d] = a / sum;
            }
        }
    criterion("attention56/8", ideal, actual, .006, .03);
    // Two independent cache segments must satisfy the same oracle, including a segment
    // wholly after some queries (empty partials), and a nonzero absolute cache origin.
    int cut = n / 2;
    if (cut) {
        std::vector<__half> ka(hk * cut * 128), va(ka.size()), kb(hk * (n - cut) * 128),
            vb(kb.size());
        for (int head = 0; head < hk; ++head) {
            std::copy_n(k.data() + head * n * 128, cut * 128, ka.data() + head * cut * 128);
            std::copy_n(v.data() + head * n * 128, cut * 128, va.data() + head * cut * 128);
            std::copy_n(k.data() + (head * n + cut) * 128, (n - cut) * 128,
                        kb.data() + head * (n - cut) * 128);
            std::copy_n(v.data() + (head * n + cut) * 128, (n - cut) * 128,
                        vb.data() + head * (n - cut) * 128);
        }
        auto dka = upload(ka), dva = upload(va), dkb = upload(kb), dvb = upload(vb);
        ops::segment_attention_f16((float*)dq.p, (__half*)dka.p, (__half*)dva.p, (int*)dp.p, t, hq,
                                   hk, first, cut, cut, (float*)state.p, work.p, true, false,
                                   (__nv_bfloat16*)out.p, 0);
        ops::segment_attention_f16((float*)dq.p, (__half*)dkb.p, (__half*)dvb.p, (int*)dp.p, t, hq,
                                   hk, first + cut, n - cut, n - cut, (float*)state.p, work.p,
                                   false, true, (__nv_bfloat16*)out.p, 0);
        out.copy_to_host(actual_b.data(), out.bytes);
        for (std::size_t i = 0; i < q.size(); ++i) actual[i] = __bfloat162float(actual_b[i]);
        criterion("segmented56/8", ideal, actual, .006, .03);
    }
}

void full_context(int block) {
    constexpr int n = 262144, period = 257, hq = 56, hk = 8;
    std::vector<float> q(hq * 128);
    for (auto& z : q) z = random_value();
    std::vector<__half> pk(hk * period * 128), pv(pk.size());
    for (auto& z : pk) z = __float2half(random_value());
    for (auto& z : pv) z = __float2half(random_value());
    auto dq = upload(q), dp = upload(std::vector<int>{n - 1});
    DeviceBuffer dk(hk * block * 128 * 2), dv(dk.bytes), state(hq * 130 * 4),
        work(ops::segment_attention_workspace_bytes(1, hq, block)), out(hq * 128 * 2);
    std::vector<__half> k(hk * block * 128), v(k.size());
    for (int at = 0; at < n; at += block) {
        for (int head = 0; head < hk; ++head)
            for (int j = 0; j < block; ++j)
                for (int d = 0; d < 128; ++d) {
                    int i      = (head * block + j) * 128 + d,
                        origin = (head * period + (at + j) % period) * 128 + d;
                    k[i]       = pk[origin];
                    v[i]       = pv[origin];
                }
        dk.copy_from_host(k.data(), dk.bytes);
        dv.copy_from_host(v.data(), dv.bytes);
        ops::segment_attention_f16((float*)dq.p, (__half*)dk.p, (__half*)dv.p, (int*)dp.p, 1, hq,
                                   hk, at, block, block, (float*)state.p, work.p, at == 0,
                                   at + block == n, (__nv_bfloat16*)out.p, 0);
    }
    std::vector<__nv_bfloat16> b(q.size());
    out.copy_to_host(b.data(), out.bytes);
    std::vector<float> actual(q.size());
    for (std::size_t i = 0; i < b.size(); ++i) actual[i] = __bfloat162float(b[i]);
    std::vector<double> ideal(q.size());
    // The represented cache repeats every 257 rows. The exact FP64 softmax oracle counts
    // every row through its multiplicity, avoiding a second implementation of the segmented
    // algorithm.
    for (int head = 0; head < hq; ++head) {
        int kv = head / 7;
        std::vector<double> scores(period);
        double max = -INFINITY, sum = 0;
        for (int j = 0; j < period; ++j) {
            double dot = 0;
            for (int d = 0; d < 128; ++d)
                dot += double(q[head * 128 + d]) * __half2float(pk[(kv * period + j) * 128 + d]);
            scores[j] = dot / std::sqrt(128.0);
            max       = std::max(max, scores[j]);
        }
        for (int j = 0; j < period; ++j) {
            int count = n / period + (j < n % period);
            scores[j] = count * std::exp(scores[j] - max);
            sum += scores[j];
        }
        for (int d = 0; d < 128; ++d) {
            double weighted = 0;
            for (int j = 0; j < period; ++j)
                weighted += scores[j] * __half2float(pv[(kv * period + j) * 128 + d]);
            ideal[head * 128 + d] = weighted / sum;
        }
    }
    criterion("full-262144-segmented56/8", ideal, actual, .006, .03);
}

} // namespace

int main() {
    try {
        int n  = 0;
        auto e = cudaGetDeviceCount(&n);
        if (test::cuda_unavailable(e) || n == 0) return 77;
        CUDA_CHECK(e);
        rms();
        rotary();
        attention(1, 1, 0);
        attention(3, 1024, 261120);
        attention(128, 33, 0);
        attention(3, 8192, 253952);
        full_context(1024);
        full_context(8192);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
