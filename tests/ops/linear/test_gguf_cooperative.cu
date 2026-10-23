#include "ops/linear/gguf/ggml_bridge_vec.cuh"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer::ops::gguf::detail;

template <ggml_type Type>
__global__ void compare_decoders(const std::uint8_t* weights, int blocks, int* failures) {
    using D = vec::Decoder<Type>;
    const int lane = threadIdx.x & 31;
    if (lane >= blocks * 8) { return; }
    const auto* block = weights + (threadIdx.x / 8) * D::kBlockBytes;
    vec::Slice plain{}, cooperative{};
    D::decode(block, lane % 8, nullptr, plain);
    D::template decode<true>(block, lane % 8, nullptr, cooperative);
    for (int i = 0; i < 8; ++i) {
        if (plain.w[i] != cooperative.w[i]) { atomicAdd(failures, 1); }
    }
    for (int i = 0; i < 2; ++i) {
        if (__float_as_uint(plain.f[i]) != __float_as_uint(cooperative.f[i])) {
            atomicAdd(failures, 1);
        }
    }
}

void checked(cudaError_t error) {
    if (error != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(error)); }
}

template <ggml_type Type>
void run() {
    using D = vec::Decoder<Type>;
    std::mt19937 random(3719);
    std::vector<std::uint8_t> weights(16 * D::kBlockBytes);
    std::uint8_t* device = nullptr;
    int* failures = nullptr;
    checked(cudaMalloc(&device, weights.size()));
    checked(cudaMalloc(&failures, sizeof(int)));
    checked(cudaMemset(failures, 0, sizeof(int)));
    for (int sample = 0; sample < 64; ++sample) {
        for (auto& value : weights) { value = std::uint8_t(random()); }
        const std::uint16_t scale = 0x3800;
        for (int b = 0; b < 16; ++b) {
            if constexpr (Type == GGML_TYPE_Q4_K) {
                std::memcpy(weights.data() + b * D::kBlockBytes, &scale, 2);
                std::memcpy(weights.data() + b * D::kBlockBytes + 2, &scale, 2);
            } else {
                std::memcpy(weights.data() + b * D::kBlockBytes + 208, &scale, 2);
            }
        }
        checked(cudaMemcpy(device, weights.data(), weights.size(), cudaMemcpyHostToDevice));
        for (int blocks = 1; blocks <= 4; ++blocks) {
            compare_decoders<Type><<<1, 128>>>(device, blocks, failures);
            checked(cudaGetLastError());
        }
    }
    int count = 0;
    checked(cudaMemcpy(&count, failures, sizeof(int), cudaMemcpyDeviceToHost));
    checked(cudaFree(failures));
    checked(cudaFree(device));
    if (count != 0) { throw std::runtime_error("cooperative decoder mismatch"); }
}
template <ggml_type Type, int T, bool Fused, int KS = 0>
void product_case(int k) {
    using D = vec::Decoder<Type>;
    constexpr int rows = 9, stride = rows + 3;
    const int row_bytes = k / 256 * D::kBlockBytes;
    std::mt19937 random(9137 + k);
    std::vector<std::uint8_t> weights(2 * rows * row_bytes);
    for (auto& value : weights) { value = std::uint8_t(random()); }
    const std::uint16_t scale = 0x1800;
    for (std::size_t b = 0; b < weights.size(); b += D::kBlockBytes) {
        if constexpr (Type == GGML_TYPE_Q4_K) {
            std::memcpy(weights.data() + b, &scale, 2);
            std::memcpy(weights.data() + b + 2, &scale, 2);
        } else {
            std::memcpy(weights.data() + b + 208, &scale, 2);
        }
    }
    std::vector<std::int8_t> qs(k * T);
    for (auto& value : qs) { value = std::int8_t(int(random() % 255) - 127); }
    std::vector<half2> ds(k / 32 * T, __floats2half2_rn(0.125f, 0.25f));
    std::vector<float> initial(stride * T, 0.125f), scalar(initial.size()), cooperative(initial.size());
    std::uint8_t* weight_device = nullptr;
    std::int8_t* qs_device = nullptr;
    half2* ds_device = nullptr;
    float* out = nullptr;
    checked(cudaMalloc(&weight_device, weights.size()));
    checked(cudaMalloc(&qs_device, qs.size()));
    checked(cudaMalloc(&ds_device, ds.size() * sizeof(half2)));
    checked(cudaMalloc(&out, initial.size() * sizeof(float)));
    checked(cudaMemcpy(weight_device, weights.data(), weights.size(), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(qs_device, qs.data(), qs.size(), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(ds_device, ds.data(), ds.size() * sizeof(half2), cudaMemcpyHostToDevice));
    VecArgs args{weight_device, row_bytes, rows, k, qs_device, ds_device, {}};
    args.out.f32 = out;
    args.out.column_stride = stride;
    for (const auto epilogue : {ninfer::ops::gguf::Epilogue::Store,
                                ninfer::ops::gguf::Epilogue::Accumulate}) {
        args.out.epilogue = epilogue;
        checked(cudaMemcpy(out, initial.data(), initial.size() * sizeof(float), cudaMemcpyHostToDevice));
        vec::kernel<Type, T, Fused, KS, false><<<1, 128>>>(args);
        checked(cudaGetLastError());
        checked(cudaMemcpy(scalar.data(), out, scalar.size() * sizeof(float), cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(out, initial.data(), initial.size() * sizeof(float), cudaMemcpyHostToDevice));
        vec::kernel<Type, T, Fused, KS, true><<<1, 128>>>(args);
        checked(cudaGetLastError());
        checked(cudaMemcpy(cooperative.data(), out, cooperative.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (std::memcmp(scalar.data(), cooperative.data(), scalar.size() * sizeof(float)) != 0) {
            throw std::runtime_error("cooperative dense product mismatch");
        }
        for (int t = 0; t < T; ++t) {
            for (int r = rows; r < stride; ++r) {
                if (cooperative[t * stride + r] != initial[t * stride + r]) {
                    throw std::runtime_error("cooperative output guard overwritten");
                }
            }
        }
    }
    checked(cudaFree(out));
    checked(cudaFree(ds_device));
    checked(cudaFree(qs_device));
    checked(cudaFree(weight_device));
}

template <ggml_type Type>
void products() {
    for (int k : {256, 768, 1536, 17408}) {
        product_case<Type, 1, false>(k);
        product_case<Type, 4, false>(k);
        product_case<Type, 8, false>(k);
        product_case<Type, 1, true>(k);
        product_case<Type, 4, true>(k);
        product_case<Type, 8, true>(k);
    }
    product_case<Type, 4, false, 5120>(5120);
    product_case<Type, 4, true, 5120>(5120);
}

}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    try {
        run<GGML_TYPE_Q4_K>();
        run<GGML_TYPE_Q6_K>();
        products<GGML_TYPE_Q4_K>();
        products<GGML_TYPE_Q6_K>();
        std::cout << "OK cooperative Q4_K/Q6_K, full and partial warps\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
