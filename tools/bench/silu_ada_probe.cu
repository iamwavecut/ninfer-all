#include "ops/kernel/silu_and_mul.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>

using namespace ninfer::ops;

int main() {
    cudaDeviceProp device{};
    cudaGetDeviceProperties(&device, 0);
    std::printf("GPU %s sm_%d%d\n", device.name, device.major, device.minor);
    if (device.major != 8 || device.minor != 9) {
        std::fprintf(stderr, "sm_89 device required\n");
        return 2;
    }
    int failures = 0;
    for (int n : {8, 9, 17408, 17415, 65536, 17408 * 8, 17408 * 16,
                  17408 * 32, 17408 * 48}) {
        std::vector<__nv_bfloat16> gate(n), up(n), got(n);
        for (int i = 0; i < n; ++i) {
            gate[i] = __float2bfloat16_rn((i % 131 - 65) * 0.125f);
            up[i] = __float2bfloat16_rn((i % 97 - 48) * 0.0625f);
        }
        __nv_bfloat16 *dg{}, *du{}, *dy{};
        cudaMalloc(&dg, n * 2); cudaMalloc(&du, n * 2); cudaMalloc(&dy, n * 2);
        cudaMemcpy(dg, gate.data(), n * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(du, up.data(), n * 2, cudaMemcpyHostToDevice);
        if (n % 8 == 0) {
            const int packs = n / 8;
            silu_and_mul_bf16x8_kernel<<<(packs + 127) / 128, 128>>>(
                reinterpret_cast<Bf16x8Pack*>(dg), reinterpret_cast<Bf16x8Pack*>(du),
                reinterpret_cast<Bf16x8Pack*>(dy), packs);
        } else {
            silu_and_mul_kernel<<<(n + 1023) / 1024, 128>>>(dg, du, dy, n);
        }
        const auto launch = cudaGetLastError();
        const auto sync = cudaDeviceSynchronize();
        if (launch != cudaSuccess || sync != cudaSuccess) {
            std::printf("n=%d CUDA error: %s / %s\n", n, cudaGetErrorString(launch),
                        cudaGetErrorString(sync));
            return 2;
        }
        cudaMemcpy(got.data(), dy, n * 2, cudaMemcpyDeviceToHost);
        int mismatches = 0;
        for (int i = 0; i < n; ++i) {
            const double g = __bfloat162float(gate[i]);
            const double u = __bfloat162float(up[i]);
            const double oracle = (g / (1.0 + std::exp(-g))) * u;
            const double actual = __bfloat162float(got[i]);
            if (std::abs(actual - oracle) > 0.02 + 0.008 * std::abs(oracle)) ++mismatches;
        }
        std::printf("n=%d mismatches=%d\n", n, mismatches);
        failures += mismatches;
        if (n % 8 == 0) {
            std::vector<__nv_bfloat16> reference(n);
            silu_and_mul_kernel<<<(n + 1023) / 1024, 128>>>(dg, du, dy, n);
            cudaDeviceSynchronize();
            cudaMemcpy(reference.data(), dy, n * 2, cudaMemcpyDeviceToHost);
            int bit_differences = 0;
            for (int i = 0; i < n; ++i) {
                if (std::memcmp(&reference[i], &got[i], 2) != 0) ++bit_differences;
            }
            std::printf("n=%d baseline_bit_differences=%d\n", n, bit_differences);
            failures += bit_differences;
            const int packs = n / 8;
            auto run = [&](bool vector) {
                cudaEvent_t start{}, stop{};
                cudaEventCreate(&start); cudaEventCreate(&stop);
                for (int i = 0; i < 20; ++i) {
                    if (vector) silu_and_mul_bf16x8_kernel<<<(packs + 127) / 128, 128>>>(
                        reinterpret_cast<Bf16x8Pack*>(dg), reinterpret_cast<Bf16x8Pack*>(du),
                        reinterpret_cast<Bf16x8Pack*>(dy), packs);
                    else silu_and_mul_kernel<<<(n + 1023) / 1024, 128>>>(dg, du, dy, n);
                }
                cudaDeviceSynchronize();
                cudaEventRecord(start);
                for (int i = 0; i < 2000; ++i) {
                    if (vector) silu_and_mul_bf16x8_kernel<<<(packs + 127) / 128, 128>>>(
                        reinterpret_cast<Bf16x8Pack*>(dg), reinterpret_cast<Bf16x8Pack*>(du),
                        reinterpret_cast<Bf16x8Pack*>(dy), packs);
                    else silu_and_mul_kernel<<<(n + 1023) / 1024, 128>>>(dg, du, dy, n);
                }
                cudaEventRecord(stop); cudaEventSynchronize(stop);
                float ms{}; cudaEventElapsedTime(&ms, start, stop);
                cudaEventDestroy(start); cudaEventDestroy(stop);
                return ms * 1000.0f / 2000;
            };
            float baseline[6], candidate[6];
            for (int rep = 0; rep < 6; ++rep) {
                if (rep & 1) { candidate[rep] = run(true); baseline[rep] = run(false); }
                else { baseline[rep] = run(false); candidate[rep] = run(true); }
            }
            float worst_ratio = 100.0f;
            for (int rep = 0; rep < 6; ++rep) {
                worst_ratio = std::min(worst_ratio, baseline[rep] / candidate[rep]);
            }
            std::sort(baseline, baseline + 6);
            std::sort(candidate, candidate + 6);
            std::printf("n=%d baseline_us=%.3f [%.3f,%.3f] candidate_us=%.3f [%.3f,%.3f] ratio=%.3f worst_pair_ratio=%.3f\n", n,
                        (baseline[2] + baseline[3]) / 2, baseline[0], baseline[5],
                        (candidate[2] + candidate[3]) / 2, candidate[0], candidate[5],
                        (baseline[2] + baseline[3]) / (candidate[2] + candidate[3]), worst_ratio);
        }
        cudaFree(dg); cudaFree(du); cudaFree(dy);
    }
    return failures ? 1 : 0;
}
