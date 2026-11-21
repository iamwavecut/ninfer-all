// Inkling 2's vision kernels against FP64 oracles, at the tower's widths (16 heads of 64, hidden
// 1024, MLP 4096, a 32 x 32 grid of learned position rows, 2 x 2 cells into 4096 values):
//  - embed: the class row and each patch's projection plus its position row, as stored at a 32 x
//    32 grid and bilinearly resized (half-pixel centers, clamped edges) to smaller, larger, wide
//    and tall grids;
//  - layer_norm: BF16 and FP32 outputs of rows with a large mean;
//  - rotate_qkv: the biases, then the 2D RoPE of Q and K (the first half of each head's adjacent
//    pairs by the column position, the second by the row, the class row at 0, patch (x, y) at
//    (x + 1, y + 1)) on wide and tall grids, V with its bias only, allowing the error FP32 angles
//    carry;
//  - softmax_rows over key counts within and past one block's threads;
//  - residual: h += (y + bias) * scale in FP32;
//  - gelu with and without its bias (the erf GELU, its negative tail too);
//  - merge: each 2 x 2 cell of patches into one row (value dy * 2 hidden + dx * hidden + c), the
//    class row left out, rounded to BF16 as the only change.
// A BF16 output is compared with the oracle's rounding of its exact value; the two may differ by
// a step where the FP32 arithmetic lands on the other side of a rounding boundary. Values that only
// move and round (merge) must match bit for bit.

#include "models/inkling2/vision_kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace vk = ninfer::models::inkling2::vision_kernels;
using bf16   = __nv_bfloat16;

constexpr int kHidden = 1024, kHeadDim = 64, kWidth = 4096, kGrid = 32;

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

template <class T>
struct Device {
    T* p          = nullptr;
    std::size_t n = 0;
    explicit Device(std::size_t count) : n(count) {
        check(cudaMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T)), "malloc");
    }
    explicit Device(const std::vector<T>& host) : Device(host.size()) {
        check(cudaMemcpy(p, host.data(), n * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    ~Device() { cudaFree(p); }
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
    [[nodiscard]] std::vector<T> host() const {
        check(cudaDeviceSynchronize(), "kernel");
        std::vector<T> out(n);
        check(cudaMemcpy(out.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return out;
    }
};

std::vector<float> normal(std::size_t n, float sigma, std::mt19937& rng, float mean = 0.0f) {
    std::normal_distribution<float> dist(mean, sigma);
    std::vector<float> v(n);
    for (auto& x : v) { x = dist(rng); }
    return v;
}

double value(bf16 x) { return __bfloat162float(x); }

std::uint16_t bits(bf16 x) {
    std::uint16_t b = 0;
    std::memcpy(&b, &x, 2);
    return b;
}

// BF16 outputs against the oracle's rounding of the exact values: how many differ, and the largest
// difference in steps of BF16 at the oracle's value. `slack` is an absolute error the FP32
// arithmetic before the rounding may carry: an output within half a step plus it of the exact value
// counts as equal.
struct Tally {
    std::size_t values = 0, off = 0;
    double worst = 0.0;

    void add(bf16 got, double exact, double slack = 0.0) {
        ++values;
        const double want = value(__float2bfloat16_rn(float(exact)));
        if (value(got) == want) { return; }
        const double step =
            std::ldexp(1.0, std::max(std::ilogb(std::max(std::fabs(want), 1e-37)), -126) - 7);
        if (slack > 0.0 && std::fabs(value(got) - exact) <= 0.5 * step + slack) { return; }
        ++off;
        const double steps = std::isfinite(value(got)) ? std::fabs(value(got) - want) / step : INFINITY;
        if (!(steps <= worst)) { worst = steps; }
    }

    [[nodiscard]] bool within(double fraction, double steps) const {
        return double(off) <= fraction * double(values) && worst <= steps;
    }

    [[nodiscard]] std::string str() const {
        return std::to_string(off) + "/" + std::to_string(values) + " off by " + num(worst) +
               " steps";
    }
};

// The largest |got - want| over the oracle's |want| + 1 (FP32 outputs).
struct Relative {
    double worst = 0.0;
    void add(double got, double want) {
        worst = std::max(worst, std::fabs(got - want) / (std::fabs(want) + 1.0));
    }
};

// ggml_interpolate's bilinear sample of one axis: the lower neighbour, the upper one and the upper
// one's weight for output index i of n from `grid` inputs.
void bilinear_axis(int i, int n, int& lo, int& hi, double& weight) {
    const double f = (double(i) + 0.5) * double(kGrid) / double(n) - 0.5;
    lo     = int(std::floor(f));
    hi     = std::clamp(lo + 1, 0, kGrid - 1);
    lo     = std::clamp(lo, 0, kGrid - 1);
    weight = std::clamp(f - double(lo), 0.0, 1.0);
}

void embed(int grid_w, int grid_h, std::mt19937& rng) {
    const int patches  = grid_w * grid_h;
    const auto proj    = normal(std::size_t(patches) * kHidden, 1.0f, rng);
    const auto table   = normal(std::size_t(kGrid) * kGrid * kHidden, 0.5f, rng);
    const auto cls     = normal(kHidden, 1.0f, rng);
    const auto cls_pos = normal(kHidden, 0.5f, rng);
    Device<float> dp(proj), dt(table), dc(cls), dcp(cls_pos), h(std::size_t(patches + 1) * kHidden);
    vk::embed(dp.p, dt.p, kGrid, grid_w, grid_h, dc.p, dcp.p, kHidden, h.p, nullptr);
    const auto got = h.host();
    Relative err;
    for (int c = 0; c < kHidden; ++c) { err.add(got[std::size_t(c)], double(cls[c]) + cls_pos[c]); }
    for (int y = 0; y < grid_h; ++y) {
        int y0 = 0, y1 = 0, x0 = 0, x1 = 0;
        double dy = 0.0, dx = 0.0;
        bilinear_axis(y, grid_h, y0, y1, dy);
        for (int x = 0; x < grid_w; ++x) {
            bilinear_axis(x, grid_w, x0, x1, dx);
            const std::size_t p = std::size_t(y) * grid_w + x;
            const auto at = [&](int yy, int xx, int c) {
                return double(table[(std::size_t(yy) * kGrid + xx) * kHidden + c]);
            };
            for (int c = 0; c < kHidden; ++c) {
                const double pos = at(y0, x0, c) * (1 - dx) * (1 - dy) + at(y0, x1, c) * dx * (1 - dy) +
                                   at(y1, x0, c) * (1 - dx) * dy + at(y1, x1, c) * dx * dy;
                err.add(got[(p + 1) * kHidden + c], double(proj[p * kHidden + c]) + pos);
            }
        }
    }
    // The FP32 coordinates differ from the FP64 ones by a few units in their last place.
    report(err.worst <= 1e-5, "embed " + std::to_string(grid_w) + " x " + std::to_string(grid_h) +
                                  ": worst error " + num(err.worst));
}

void layer_norm(int rows, std::mt19937& rng) {
    const float eps = 1e-5f;
    const auto x    = normal(std::size_t(rows) * kHidden, 3.0f, rng, 5.0f);
    const auto w    = normal(kHidden, 0.3f, rng, 1.0f);
    const auto b    = normal(kHidden, 0.2f, rng);
    Device<float> dx(x), dw(w), db(b), f32(x.size());
    Device<bf16> out(x.size());
    vk::layer_norm(dx.p, dw.p, db.p, rows, kHidden, eps, out.p, nullptr, f32.p);
    const auto got   = out.host();
    const auto got_f = f32.host();
    Tally tally;
    Relative err;
    for (int r = 0; r < rows; ++r) {
        const float* row = x.data() + std::size_t(r) * kHidden;
        double mean = 0.0, var = 0.0;
        for (int i = 0; i < kHidden; ++i) { mean += row[i]; }
        mean /= kHidden;
        for (int i = 0; i < kHidden; ++i) { var += (row[i] - mean) * (row[i] - mean); }
        var /= kHidden;
        for (int i = 0; i < kHidden; ++i) {
            const double y = (row[i] - mean) / std::sqrt(var + eps) * w[i] + b[i];
            tally.add(got[std::size_t(r) * kHidden + i], y);
            err.add(got_f[std::size_t(r) * kHidden + i], y);
        }
    }
    report(tally.within(0.01, 1.0) && err.worst <= 1e-5,
           "layer_norm " + std::to_string(rows) + " rows: bf16 " + tally.str() + ", fp32 " +
               num(err.worst));
}

void rotate_qkv(int grid_w, int grid_h, std::mt19937& rng) {
    const int rows = 1 + grid_w * grid_h, quarter = kHeadDim / 4;
    const auto qkv = normal(std::size_t(rows) * 3 * kHidden, 1.0f, rng);
    const auto qb  = normal(kHidden, 0.3f, rng);
    const auto kb  = normal(kHidden, 0.3f, rng);
    const auto vb  = normal(kHidden, 0.3f, rng);
    std::vector<float> freq(quarter);
    for (int i = 0; i < quarter; ++i) {
        freq[std::size_t(i)] = float(std::pow(10000.0, -2.0 * i / double(kHeadDim / 2)));
    }
    Device<float> dq(qkv), dqb(qb), dkb(kb), dvb(vb), df(freq);
    Device<bf16> out(qkv.size());
    vk::rotate_qkv(dq.p, dqb.p, dkb.p, dvb.p, rows, grid_w, kHidden, kHeadDim, df.p, out.p, nullptr);
    const auto got = out.host();
    Tally tally;
    for (int r = 0; r < rows; ++r) {
        const double px = r == 0 ? 0 : (r - 1) % grid_w + 1, py = r == 0 ? 0 : (r - 1) / grid_w + 1;
        const std::size_t base = std::size_t(r) * 3 * kHidden;
        for (int section = 0; section < 3; ++section) {
            const auto& bias = section == 0 ? qb : section == 1 ? kb : vb;
            for (int head = 0; head < kHidden / kHeadDim; ++head) {
                for (int i = 0; i < kHeadDim / 2; ++i) {
                    const int a = head * kHeadDim + 2 * i, b = a + 1;
                    const double x1 = double(qkv[base + section * kHidden + a]) + bias[a];
                    const double x2 = double(qkv[base + section * kHidden + b]) + bias[b];
                    const std::size_t at = base + section * kHidden;
                    if (section == 2) {
                        tally.add(got[at + a], x1);
                        tally.add(got[at + b], x2);
                        continue;
                    }
                    const double angle = (i < quarter ? px : py) * freq[std::size_t(i % quarter)];
                    // The FP32 angle (a unit in its last place is up to 2^-17 radians here), sine,
                    // cosine and biased inputs: an error of a few of those units times the inputs.
                    const double slack =
                        std::ldexp((std::fabs(x1) + std::fabs(x2)) * (4.0 + angle), -22);
                    tally.add(got[at + a], x1 * std::cos(angle) - x2 * std::sin(angle), slack);
                    tally.add(got[at + b], x1 * std::sin(angle) + x2 * std::cos(angle), slack);
                }
            }
        }
    }
    report(tally.within(0.01, 1.0), "rotate_qkv " + std::to_string(grid_w) + " x " +
                                        std::to_string(grid_h) + ": " + tally.str());
}

void softmax_rows(int rows, int keys, std::mt19937& rng) {
    const auto s = normal(std::size_t(rows) * keys, 4.0f, rng);
    Device<float> ds(s);
    Device<bf16> p(s.size());
    vk::softmax_rows(ds.p, rows, keys, p.p, nullptr);
    const auto got = p.host();
    Tally tally;
    for (int r = 0; r < rows; ++r) {
        const float* row = s.data() + std::size_t(r) * keys;
        const double m   = *std::max_element(row, row + keys);
        double sum       = 0.0;
        for (int j = 0; j < keys; ++j) { sum += std::exp(row[j] - m); }
        for (int j = 0; j < keys; ++j) {
            tally.add(got[std::size_t(r) * keys + j], std::exp(row[j] - m) / sum);
        }
    }
    report(tally.within(0.01, 1.0), "softmax_rows " + std::to_string(rows) + " x " +
                                        std::to_string(keys) + ": " + tally.str());
}

void residual(int rows, std::mt19937& rng) {
    const auto h0    = normal(std::size_t(rows) * kHidden, 2.0f, rng);
    const auto y     = normal(h0.size(), 1.0f, rng);
    const auto bias  = normal(kHidden, 0.5f, rng);
    const auto scale = normal(kHidden, 0.1f, rng);
    Device<float> dh(h0), dy(y), db(bias), ds(scale);
    vk::residual(dh.p, dy.p, db.p, ds.p, rows, kHidden, nullptr);
    const auto got = dh.host();
    double worst = 0.0;
    for (std::size_t k = 0; k < h0.size(); ++k) {
        const int c       = int(k % kHidden);
        const double add  = (double(y[k]) + bias[c]) * scale[c];
        const double want = h0[k] + add;
        // An addition, a product and an addition in FP32 (a fused multiply-add may skip a rounding).
        const double bound = std::ldexp(std::fabs(h0[k]) + 2.0 * std::fabs(add) +
                                            std::fabs(double(bias[c]) * scale[c]) + 1e-30,
                                        -23);
        worst = std::max(worst, std::fabs(got[k] - want) / bound);
    }
    report(worst <= 2.0, "residual " + std::to_string(rows) + " rows: " + num(worst) +
                             " of the FP32 bound");
}

void gelu(int rows, bool with_bias, std::mt19937& rng) {
    const auto x    = normal(std::size_t(rows) * kWidth, 3.0f, rng);
    const auto bias = normal(kWidth, 0.5f, rng);
    Device<float> dx(x), db(bias);
    Device<bf16> mid(x.size());
    vk::gelu(dx.p, with_bias ? db.p : nullptr, rows, kWidth, mid.p, nullptr);
    const auto got = mid.host();
    Tally tally;
    for (std::size_t k = 0; k < x.size(); ++k) {
        const double v = double(x[k]) + (with_bias ? bias[k % kWidth] : 0.0);
        tally.add(got[k], 0.5 * v * std::erfc(-v / std::sqrt(2.0)));
    }
    report(tally.within(0.01, 1.0), std::string("gelu") + (with_bias ? " with bias " : " ") +
                                        std::to_string(rows) + " rows: " + tally.str());
}

void merge(int grid_w, int grid_h, std::mt19937& rng) {
    const auto h = normal(std::size_t(1 + grid_w * grid_h) * kHidden, 1.0f, rng);
    const int cells_w = grid_w / 2, cells_h = grid_h / 2, features = 4 * kHidden;
    Device<float> dh(h);
    Device<bf16> cells(std::size_t(cells_w) * cells_h * features);
    vk::merge(dh.p, grid_w, grid_h, kHidden, cells.p, nullptr);
    const auto got = cells.host();
    std::size_t wrong = 0;
    for (int cy = 0; cy < cells_h; ++cy) {
        for (int cx = 0; cx < cells_w; ++cx) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const std::size_t patch = std::size_t(2 * cy + dy) * grid_w + (2 * cx + dx);
                    for (int c = 0; c < kHidden; ++c) {
                        const std::size_t at = (std::size_t(cy) * cells_w + cx) * features +
                                               dy * 2 * kHidden + dx * kHidden + c;
                        wrong += bits(got[at]) != bits(__float2bfloat16_rn(h[(1 + patch) * kHidden + c]));
                    }
                }
            }
        }
    }
    report(wrong == 0, "merge " + std::to_string(grid_w) + " x " + std::to_string(grid_h) + ": " +
                           std::to_string(wrong) + " values differ");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        std::mt19937 rng(2054);
        embed(kGrid, kGrid, rng);
        embed(6, 4, rng);
        embed(40, 26, rng);
        embed(64, 2, rng);
        embed(2, 50, rng);
        layer_norm(5, rng);
        rotate_qkv(40, 3, rng);
        rotate_qkv(2, 64, rng);
        softmax_rows(4, 37, rng);
        softmax_rows(3, 4097, rng);
        residual(7, rng);
        gelu(3, true, rng);
        gelu(2, false, rng);
        merge(6, 4, rng);
        merge(2, 2, rng);
        merge(40, 26, rng);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Inkling 2 vision kernels\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Inkling 2 vision kernels: " << error.what() << '\n';
        return 1;
    }
}
