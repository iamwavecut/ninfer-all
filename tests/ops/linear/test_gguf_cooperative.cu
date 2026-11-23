#include "ops/linear/gguf/ggml_bridge_vec.cuh"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/common/device_route.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
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
        constexpr const char* key = Type == GGML_TYPE_Q4_K ? "gguf_vector/q4_k"
                                                           : "gguf_vector/q6_k";
        for (const char* schedule : {"scalar", "cooperative"}) {
            const ninfer::ops::DeviceRouteForce force(key, schedule);
            checked(cudaMemcpy(out, initial.data(), initial.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
            VecJobs one{};
            one.job[0] = args;
            one.count  = 1;
            vec_launch<Type>(one, T, Fused, nullptr);
            checked(cudaMemcpy(cooperative.data(), out, cooperative.size() * sizeof(float),
                               cudaMemcpyDeviceToHost));
            if (std::memcmp(scalar.data(), cooperative.data(), scalar.size() * sizeof(float)) != 0) {
                throw std::runtime_error("profiled dense dispatch differs from scalar kernel");
            }
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
        product_case<Type, 2, false>(k);
        product_case<Type, 3, false>(k);
        product_case<Type, 4, false>(k);
        product_case<Type, 5, false>(k);
        product_case<Type, 6, false>(k);
        product_case<Type, 7, false>(k);
        product_case<Type, 8, false>(k);
        product_case<Type, 1, true>(k);
        product_case<Type, 2, true>(k);
        product_case<Type, 3, true>(k);
        product_case<Type, 4, true>(k);
        product_case<Type, 5, true>(k);
        product_case<Type, 6, true>(k);
        product_case<Type, 7, true>(k);
        product_case<Type, 8, true>(k);
    }
    product_case<Type, 4, false, 5120>(5120);
    product_case<Type, 4, true, 5120>(5120);
}

template <ggml_type Type, int T, bool Fused>
void moe_product_case(int k) {
    using D = vec::Decoder<Type>;
    constexpr int experts = 3, rows = 9, pairs = 5;
    const int row_bytes = k / 256 * D::kBlockBytes;
    std::mt19937 random(6317 + k + T);
    std::vector<std::uint8_t> weights(experts * rows * row_bytes);
    std::vector<std::uint8_t> second_weights(experts * rows * row_bytes);
    const std::uint16_t scale = 0x1800;
    for (auto& value : weights) { value = std::uint8_t(random()); }
    for (auto& value : second_weights) { value = std::uint8_t(random()); }
    for (std::size_t b = 0; b < weights.size(); b += D::kBlockBytes) {
        if constexpr (Type == GGML_TYPE_Q4_K) {
            std::memcpy(weights.data() + b, &scale, 2);
            std::memcpy(weights.data() + b + 2, &scale, 2);
        } else {
            std::memcpy(weights.data() + b + 208, &scale, 2);
            std::memcpy(second_weights.data() + b + 208, &scale, 2);
        }
        if constexpr (Type == GGML_TYPE_Q4_K) {
            std::memcpy(second_weights.data() + b, &scale, 2);
            std::memcpy(second_weights.data() + b + 2, &scale, 2);
        }
    }
    std::vector<std::int8_t> qs(k * pairs);
    for (auto& value : qs) { value = std::int8_t(int(random() % 255) - 127); }
    std::vector<half2> ds(k / 32 * pairs, __floats2half2_rn(0.125f, 0.25f));
    const std::int32_t bounds[] = {0, 2, 2, 5};
    const std::int32_t sorted[] = {1, 4, 0, 2, 3};
    const std::int32_t active[] = {2, 1, 0};
    const std::int32_t active_count = 3;
    std::uint8_t *weight_device = nullptr, *second_weight_device = nullptr;
    const std::uint8_t **first_device = nullptr, **second_device = nullptr;
    std::int8_t* qs_device = nullptr;
    half2* ds_device = nullptr;
    std::int32_t *bounds_device = nullptr, *sorted_device = nullptr, *active_device = nullptr;
    std::int32_t* active_count_device = nullptr;
    float* out_device = nullptr;
    __nv_bfloat16* out_bf16_device = nullptr;
    checked(cudaMalloc(&weight_device, weights.size()));
    checked(cudaMalloc(&second_weight_device, second_weights.size()));
    checked(cudaMalloc(&first_device, experts * sizeof(*first_device)));
    checked(cudaMalloc(&second_device, experts * sizeof(*second_device)));
    checked(cudaMalloc(&qs_device, qs.size()));
    checked(cudaMalloc(&ds_device, ds.size() * sizeof(half2)));
    checked(cudaMalloc(&bounds_device, sizeof(bounds)));
    checked(cudaMalloc(&sorted_device, sizeof(sorted)));
    checked(cudaMalloc(&active_device, sizeof(active)));
    checked(cudaMalloc(&active_count_device, sizeof(active_count)));
    checked(cudaMalloc(&out_device, pairs * rows * sizeof(float)));
    checked(cudaMalloc(&out_bf16_device, pairs * rows * sizeof(__nv_bfloat16)));
    checked(cudaMemcpy(weight_device, weights.data(), weights.size(), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(second_weight_device, second_weights.data(), second_weights.size(),
                       cudaMemcpyHostToDevice));
    std::vector<const std::uint8_t*> first(experts);
    std::vector<const std::uint8_t*> second(experts);
    for (int e = 0; e < experts; ++e) {
        first[e] = weight_device + std::int64_t(e) * rows * row_bytes;
        second[e] = second_weight_device + std::int64_t(e) * rows * row_bytes;
    }
    checked(cudaMemcpy(first_device, first.data(), first.size() * sizeof(*first_device),
                       cudaMemcpyHostToDevice));
    checked(cudaMemcpy(second_device, second.data(), second.size() * sizeof(*second_device),
                       cudaMemcpyHostToDevice));
    checked(cudaMemcpy(qs_device, qs.data(), qs.size(), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(ds_device, ds.data(), ds.size() * sizeof(half2), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(bounds_device, bounds, sizeof(bounds), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(sorted_device, sorted, sizeof(sorted), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(active_device, active, sizeof(active), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(active_count_device, &active_count, sizeof(active_count), cudaMemcpyHostToDevice));
    MoeVecArgs args{};
    args.first = first_device;
    args.second = second_device;
    args.row_bytes = row_bytes;
    args.rows = rows;
    args.k = k;
    args.bounds = bounds_device;
    args.sorted = sorted_device;
    args.active = active_device;
    args.active_count = active_count_device;
    args.column_group = 1;
    args.per_token = 1;
    args.qs = qs_device;
    args.ds = ds_device;
    std::vector<float> scalar(pairs * rows), cooperative(pairs * rows);
    std::vector<__nv_bfloat16> scalar_bf16(pairs * rows), cooperative_bf16(pairs * rows);
    if constexpr (Fused) { args.out_bf16 = out_bf16_device; }
    else { args.out_f32 = out_device; }
    const int per_block = kVecWarps * (Fused ? 1 : 2);
    const int groups = (rows + per_block - 1) / per_block;
    ninfer::ops::DeviceRouteProfile profile;
    profile.hardware_class = "test";
    constexpr const char* route_key =
        Type == GGML_TYPE_Q4_K ? "gguf_moe_vector/q4_k" : "gguf_moe_vector/q6_k";
    profile.routes[route_key] = {{T, "scalar"}};
    ninfer::ops::install_device_route_profile(
        std::make_shared<const ninfer::ops::DeviceRouteProfile>(profile));
    moe_launch<Type>(args, active_count, T, Fused, nullptr);
    if constexpr (Fused) {
        checked(cudaMemcpy(scalar_bf16.data(), out_bf16_device,
                           scalar_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        checked(cudaMemset(out_bf16_device, 0, scalar_bf16.size() * sizeof(__nv_bfloat16)));
    } else {
        checked(cudaMemcpy(scalar.data(), out_device, scalar.size() * sizeof(float), cudaMemcpyDeviceToHost));
        checked(cudaMemset(out_device, 0, scalar.size() * sizeof(float)));
    }
    ninfer::ops::install_device_route_profile(nullptr);
    vec::moe_kernel<Type, T, Fused, true><<<dim3(groups, active_count, 1), kVecWarps * 32>>>(args);
    checked(cudaGetLastError());
    if constexpr (Fused) {
        checked(cudaMemcpy(cooperative_bf16.data(), out_bf16_device,
                           cooperative_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    } else {
        checked(cudaMemcpy(cooperative.data(), out_device, cooperative.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));
    }
    if constexpr (Fused) {
        if (std::memcmp(scalar_bf16.data(), cooperative_bf16.data(),
                        scalar_bf16.size() * sizeof(__nv_bfloat16)) != 0) {
            throw std::runtime_error("cooperative MoE product mismatch");
        }
    } else {
        if (std::memcmp(scalar.data(), cooperative.data(), scalar.size() * sizeof(float)) != 0) {
            throw std::runtime_error("cooperative MoE product mismatch");
        }
    }
    profile.routes[route_key] = {{T, "cooperative"}};
    ninfer::ops::install_device_route_profile(
        std::make_shared<const ninfer::ops::DeviceRouteProfile>(profile));
    if constexpr (Fused) {
        checked(cudaMemset(out_bf16_device, 0, scalar_bf16.size() * sizeof(__nv_bfloat16)));
    } else {
        checked(cudaMemset(out_device, 0, scalar.size() * sizeof(float)));
    }
    moe_launch<Type>(args, active_count, T, Fused, nullptr);
    if constexpr (Fused) {
        checked(cudaMemcpy(cooperative_bf16.data(), out_bf16_device,
                           cooperative_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        if (std::memcmp(scalar_bf16.data(), cooperative_bf16.data(),
                        scalar_bf16.size() * sizeof(__nv_bfloat16)) != 0) {
            throw std::runtime_error("cooperative MoE host route mismatch");
        }
    } else {
        checked(cudaMemcpy(cooperative.data(), out_device, cooperative.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));
        if (std::memcmp(scalar.data(), cooperative.data(), scalar.size() * sizeof(float)) != 0) {
            throw std::runtime_error("cooperative MoE host route mismatch");
        }
    }
    ninfer::ops::install_device_route_profile(nullptr);
    checked(cudaFree(out_bf16_device));
    checked(cudaFree(out_device));
    checked(cudaFree(active_count_device));
    checked(cudaFree(active_device));
    checked(cudaFree(sorted_device));
    checked(cudaFree(bounds_device));
    checked(cudaFree(ds_device));
    checked(cudaFree(qs_device));
    checked(cudaFree(first_device));
    checked(cudaFree(second_device));
    checked(cudaFree(second_weight_device));
    checked(cudaFree(weight_device));
}

template <ggml_type Type>
void moe_products() {
    for (const int k : {256, 768}) {
        moe_product_case<Type, 1, false>(k);
        moe_product_case<Type, 2, false>(k);
        moe_product_case<Type, 4, false>(k);
        moe_product_case<Type, 8, false>(k);
        moe_product_case<Type, 1, true>(k);
        moe_product_case<Type, 2, true>(k);
        moe_product_case<Type, 4, true>(k);
        moe_product_case<Type, 8, true>(k);
    }
}

template <ggml_type Type>
void route_dispatch() {
    using ninfer::ops::DeviceRouteProfile;
    using ninfer::ops::install_device_route_profile;
    using Family = vec::VectorFamily;
    constexpr const char* dense_key = Type == GGML_TYPE_Q4_K ? "gguf_vector/q4_k" : "gguf_vector/q6_k";
    constexpr const char* moe_key = Type == GGML_TYPE_Q4_K ? "gguf_moe_vector/q4_k" : "gguf_moe_vector/q6_k";
    constexpr const char* up_key = Type == GGML_TYPE_Q4_K ? "gguf_moe_up/q4_k" : "gguf_moe_up/q6_k";
    DeviceRouteProfile profile;
    profile.hardware_class = "test";
    for (int width = 1; width <= 8; ++width) {
        profile.routes[dense_key] = {{width, "cooperative"}};
        install_device_route_profile(std::make_shared<const DeviceRouteProfile>(profile));
        if (!vec::cooperative_loads<Type, Family::Dense>(width) ||
            vec::cooperative_loads<Type, Family::Moe>(width) ||
            vec::cooperative_loads<Type, Family::MoeUp>(width)) {
            throw std::runtime_error("a dense profile enabled the wrong GGUF vector family");
        }
        profile.routes[dense_key] = {{width, "scalar"}};
        install_device_route_profile(std::make_shared<const DeviceRouteProfile>(profile));
        if (vec::cooperative_loads<Type, Family::Dense>(width)) {
            throw std::runtime_error("a scalar dense profile selected cooperative decoding");
        }
    }
    for (const int width : {1, 2, 4, 8}) {
        profile.routes.clear();
        profile.routes[moe_key] = {{width, "cooperative"}};
        install_device_route_profile(std::make_shared<const DeviceRouteProfile>(profile));
        if (!vec::cooperative_loads<Type, Family::Moe>(width) ||
            vec::cooperative_loads<Type, Family::Dense>(width) ||
            vec::cooperative_loads<Type, Family::MoeUp>(width)) {
            throw std::runtime_error("a MoE profile enabled the wrong GGUF vector family");
        }
        profile.routes[moe_key] = {{width, "scalar"}};
        install_device_route_profile(std::make_shared<const DeviceRouteProfile>(profile));
        if (vec::cooperative_loads<Type, Family::Moe>(width)) {
            throw std::runtime_error("a scalar MoE profile selected cooperative decoding");
        }
    }
    for (const int width : {1, 2, 4}) {
        profile.routes.clear();
        profile.routes[up_key] = {{width, "cooperative"}};
        install_device_route_profile(std::make_shared<const DeviceRouteProfile>(profile));
        if (!vec::cooperative_loads<Type, Family::MoeUp>(width) ||
            vec::cooperative_loads<Type, Family::Dense>(width) ||
            vec::cooperative_loads<Type, Family::Moe>(width)) {
            throw std::runtime_error("a MoE-up profile enabled the wrong GGUF vector family");
        }
        profile.routes[up_key] = {{width, "scalar"}};
        install_device_route_profile(std::make_shared<const DeviceRouteProfile>(profile));
        if (vec::cooperative_loads<Type, Family::MoeUp>(width)) {
            throw std::runtime_error("a scalar MoE-up profile selected cooperative decoding");
        }
    }
    profile.routes[dense_key] = {{8, "unsupported"}};
    profile.routes[moe_key] = {{8, "unsupported"}};
    profile.routes[up_key] = {{8, "unsupported"}};
    install_device_route_profile(std::make_shared<const DeviceRouteProfile>(profile));
    if (vec::cooperative_loads<Type, Family::Dense>(1) ||
        vec::cooperative_loads<Type, Family::Dense>(8) ||
        vec::cooperative_loads<Type, Family::Moe>(8) ||
        vec::cooperative_loads<Type, Family::MoeUp>(8) ||
        vec::cooperative_loads<Type, Family::Dense>(0) ||
        vec::cooperative_loads<Type, Family::Dense>(9) ||
        vec::cooperative_loads<Type, Family::Moe>(3) ||
        vec::cooperative_loads<Type, Family::MoeUp>(8)) {
        throw std::runtime_error("unsupported cooperative profile route was selected");
    }
    install_device_route_profile(nullptr);
    if (vec::cooperative_loads<Type, Family::Dense>(1) ||
        vec::cooperative_loads<Type, Family::Moe>(8) ||
        vec::cooperative_loads<Type, Family::MoeUp>(8)) {
        throw std::runtime_error("cooperative route enabled without a profile");
    }
    if (vec::cooperative_loads<GGML_TYPE_Q5_K>(1)) {
        throw std::runtime_error("an unsupported vector type selected cooperative decoding");
    }
}

template <ggml_type Type>
void moe_decode_up_route_case(int chunk, bool prepared) {
    using D = vec::Decoder<Type>;
    constexpr int rows = 17, k = 2560, slots = 2;
    const int row_bytes = k / 256 * D::kBlockBytes;
    std::mt19937 random(7811 + chunk);
    std::vector<std::uint8_t> gate(rows * row_bytes), up(rows * row_bytes);
    const std::uint16_t scale = 0x1800;
    for (auto& value : gate) { value = std::uint8_t(random()); }
    for (auto& value : up) { value = std::uint8_t(random()); }
    for (std::size_t b = 0; b < gate.size(); b += D::kBlockBytes) {
        if constexpr (Type == GGML_TYPE_Q4_K) {
            std::memcpy(gate.data() + b, &scale, 2);
            std::memcpy(gate.data() + b + 2, &scale, 2);
            std::memcpy(up.data() + b, &scale, 2);
            std::memcpy(up.data() + b + 2, &scale, 2);
        } else {
            std::memcpy(gate.data() + b + 208, &scale, 2);
            std::memcpy(up.data() + b + 208, &scale, 2);
        }
    }
    std::vector<__nv_bfloat16> input(k * slots), scalar(rows * slots), cooperative(rows * slots);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = __float2bfloat16(float(int(random() % 200) - 100) / 100.0f);
    }
    std::uint8_t *gate_device = nullptr, *up_device = nullptr;
    const std::uint8_t **gate_table_device = nullptr, **up_table_device = nullptr;
    __nv_bfloat16 *input_device = nullptr, *output_device = nullptr;
    std::int8_t* qs_device = nullptr;
    half2* ds_device = nullptr;
    int* qsum_device = nullptr;
    std::int32_t* ids_device = nullptr;
    ninfer::ops::gguf::MoeDecodeRecord* records_device = nullptr;
    checked(cudaMalloc(&gate_device, gate.size()));
    checked(cudaMalloc(&up_device, up.size()));
    checked(cudaMalloc(&gate_table_device, sizeof(gate_device)));
    checked(cudaMalloc(&up_table_device, sizeof(up_device)));
    checked(cudaMalloc(&input_device, input.size() * sizeof(__nv_bfloat16)));
    checked(cudaMalloc(&output_device, scalar.size() * sizeof(__nv_bfloat16)));
    if (prepared) {
        checked(cudaMalloc(&qs_device, k * slots * sizeof(std::int8_t)));
        checked(cudaMalloc(&ds_device, k / 32 * slots * sizeof(half2)));
        checked(cudaMalloc(&qsum_device, k / 32 * slots * sizeof(int)));
        checked(cudaMalloc(&ids_device, slots * sizeof(std::int32_t)));
        checked(cudaMalloc(&records_device, slots * sizeof(*records_device)));
    }
    checked(cudaMemcpy(gate_device, gate.data(), gate.size(), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(up_device, up.data(), up.size(), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(gate_table_device, &gate_device, sizeof(gate_device), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(up_table_device, &up_device, sizeof(up_device), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(input_device, input.data(), input.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice));
    if (prepared) {
        const std::int32_t ids[] = {0, 0};
        checked(cudaMemcpy(ids_device, ids, sizeof(ids), cudaMemcpyHostToDevice));
        ninfer::ops::gguf::MoeDecodePrepArgs prep{};
        prep.m = input_device;
        prep.tokens = slots;
        prep.ids = ids_device;
        prep.slots = slots;
        prep.per_token = 1;
        prep.records = records_device;
        prep.qs = qs_device;
        prep.ds = ds_device;
        prep.qsum = qsum_device;
        ninfer::ops::gguf::moe_decode_prep(
            Type == GGML_TYPE_Q4_K ? ninfer::ops::gguf::GgmlType::Q4_K
                                   : ninfer::ops::gguf::GgmlType::Q6_K,
            prep, nullptr);
    }
    ninfer::ops::gguf::MoeDecodeUpArgs args{};
    args.m = input_device;
    args.k = k;
    args.rows = rows;
    args.slots = slots;
    args.per_token = 1;
    args.gate = gate_table_device;
    args.up = up_table_device;
    args.row_bytes = row_bytes;
    args.middle = output_device;
    if (prepared) {
        args.records = records_device;
        args.qs = qs_device;
        args.ds = ds_device;
        args.qsum = qsum_device;
    }
    ninfer::ops::DeviceRouteProfile profile;
    profile.hardware_class = "test";
    constexpr const char* key = Type == GGML_TYPE_Q4_K ? "gguf_moe_up/q4_k" : "gguf_moe_up/q6_k";
    profile.routes[key] = {{chunk, "scalar"}};
    ninfer::ops::install_device_route_profile(std::make_shared<const ninfer::ops::DeviceRouteProfile>(profile));
    ninfer::ops::gguf::moe_decode_up(Type == GGML_TYPE_Q4_K ? ninfer::ops::gguf::GgmlType::Q4_K
                                                            : ninfer::ops::gguf::GgmlType::Q6_K,
                                    args, chunk, nullptr);
    checked(cudaMemcpy(scalar.data(), output_device, scalar.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost));
    profile.routes[key] = {{chunk, "cooperative"}};
    ninfer::ops::install_device_route_profile(std::make_shared<const ninfer::ops::DeviceRouteProfile>(profile));
    ninfer::ops::gguf::moe_decode_up(Type == GGML_TYPE_Q4_K ? ninfer::ops::gguf::GgmlType::Q4_K
                                                            : ninfer::ops::gguf::GgmlType::Q6_K,
                                    args, chunk, nullptr);
    checked(cudaMemcpy(cooperative.data(), output_device, cooperative.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost));
    ninfer::ops::install_device_route_profile(nullptr);
    if (std::memcmp(scalar.data(), cooperative.data(), scalar.size() * sizeof(__nv_bfloat16)) != 0) {
        throw std::runtime_error("cooperative MoE decode up mismatch");
    }
    if (prepared) {
        checked(cudaFree(records_device));
        checked(cudaFree(ids_device));
        checked(cudaFree(qsum_device));
        checked(cudaFree(ds_device));
        checked(cudaFree(qs_device));
    }
    checked(cudaFree(output_device));
    checked(cudaFree(input_device));
    checked(cudaFree(up_table_device));
    checked(cudaFree(gate_table_device));
    checked(cudaFree(up_device));
    checked(cudaFree(gate_device));
}

template <ggml_type Type>
void moe_decode_up_route() {
    for (const int chunk : {1, 2, 4}) {
        moe_decode_up_route_case<Type>(chunk, false);
        moe_decode_up_route_case<Type>(chunk, true);
    }
}

}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        run<GGML_TYPE_Q4_K>();
        run<GGML_TYPE_Q6_K>();
        products<GGML_TYPE_Q4_K>();
        products<GGML_TYPE_Q6_K>();
        moe_products<GGML_TYPE_Q4_K>();
        moe_products<GGML_TYPE_Q6_K>();
        route_dispatch<GGML_TYPE_Q4_K>();
        route_dispatch<GGML_TYPE_Q6_K>();
        moe_decode_up_route<GGML_TYPE_Q4_K>();
        moe_decode_up_route<GGML_TYPE_Q6_K>();
        std::cout << "OK cooperative Q4_K/Q6_K, full and partial warps\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
