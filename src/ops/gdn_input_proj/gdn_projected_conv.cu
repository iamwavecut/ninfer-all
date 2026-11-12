#include "ops/gdn_input_proj/gdn_projected_conv.h"

#include "core/device_launch.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// The rows are template constants for the whole-layer geometries and zero for a partition passed at
// run time (a tensor-parallel rank's share): `rows` then holds them.
struct ProjectedConvRows {
    std::int32_t channels = 0;
    std::int32_t query    = 0;
    std::int32_t key      = 0;
};

template <int Channels, int QueryRows, int KeyRows, int ValueRows, int StaticWidth, class Publish>
__global__ void gdn_projected_conv_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ conv_weight,
    const __nv_bfloat16* __restrict__ state_read, const std::int32_t* __restrict__ valid_columns,
    const std::int32_t* __restrict__ initial_state_slots, __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, std::int32_t width,
    Publish publish, ProjectedConvRows rows) {
    static_assert(Channels == QueryRows + KeyRows + ValueRows);
    const std::int32_t channels   = Channels != 0 ? Channels : rows.channels;
    const std::int32_t query_rows = Channels != 0 ? QueryRows : rows.query;
    const std::int32_t key_rows   = Channels != 0 ? KeyRows : rows.key;
    const std::int32_t value_rows = channels - query_rows - key_rows;
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= channels) { return; }
    const std::int32_t batch = static_cast<std::int32_t>(blockIdx.y);
    if constexpr (StaticWidth != 0) { width = StaticWidth; }

    std::int32_t valid       = valid_columns == nullptr ? width : valid_columns[batch];
    valid                    = valid < 0 ? 0 : (valid > width ? width : valid);
    const std::int64_t slot_stride = static_cast<std::int64_t>(channels) * 3;
    const std::int64_t initial_base =
        static_cast<std::int64_t>(initial_state_slots[batch]) * slot_stride;
    float s0       = __bfloat162float(state_read[initial_base + row]);
    float s1       = __bfloat162float(state_read[initial_base + channels + row]);
    float s2       = __bfloat162float(state_read[initial_base + 2LL * channels + row]);
    const float w0 = __bfloat162float(conv_weight[row]);
    const float w1 = __bfloat162float(conv_weight[channels + row]);
    const float w2 = __bfloat162float(conv_weight[2LL * channels + row]);
    const float w3 = __bfloat162float(conv_weight[3LL * channels + row]);

    for (std::int32_t token = 0; token < width; ++token) {
        const std::int64_t column = static_cast<std::int64_t>(batch) * width + token;
        if (token >= valid) {
            if (row < query_rows) {
                query[column * query_rows + row] = __float2bfloat16_rn(0.0F);
            } else if (row < query_rows + key_rows) {
                key[column * key_rows + row - query_rows] = __float2bfloat16_rn(0.0F);
            } else {
                value[column * value_rows + row - query_rows - key_rows] =
                    __float2bfloat16_rn(0.0F);
            }
            continue;
        }

        const float p              = __bfloat162float(projected[column * channels + row]);
        float conv                 = fmaf(w0, s0, 0.0F);
        conv                       = fmaf(w1, s1, conv);
        conv                       = fmaf(w2, s2, conv);
        conv                       = fmaf(w3, p, conv);
        const __nv_bfloat16 output = __float2bfloat16_rn(silu(conv));
        if (row < query_rows) {
            query[column * query_rows + row] = output;
        } else if (row < query_rows + key_rows) {
            key[column * key_rows + row - query_rows] = output;
        } else {
            value[column * value_rows + row - query_rows - key_rows] = output;
        }
        publish.publish(token, batch, row, s1, s2, p);
        s0 = s1;
        s1 = s2;
        s2 = p;
    }
}

template <int Channels, int QueryRows, int KeyRows, int ValueRows, class Publish>
void launch(const Tensor& projected, const Tensor& conv_weight, const Tensor& state_read,
            const Tensor& valid_columns, const Tensor& initial_state_slots, Tensor& query,
            Tensor& key, Tensor& value, Publish publish, cudaStream_t stream) {
    constexpr int kDefaultThreads = 256;
    const ProjectedConvRows rows{projected.ne[0], query.ne[0], key.ne[0]};
    const std::int32_t width      = projected.ne[1];
    const std::int32_t batch      = projected.ne[2];
    if constexpr (Channels == 10240) {
        if (width == 4 && batch == 1) {
            constexpr int kT4Threads = 64;
            gdn_projected_conv_kernel<Channels, QueryRows, KeyRows, ValueRows, 4>
                <<<(Channels + kT4Threads - 1) / kT4Threads, kT4Threads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(projected.data),
                    static_cast<const __nv_bfloat16*>(conv_weight.data),
                    static_cast<const __nv_bfloat16*>(state_read.data),
                    valid_columns.data == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int32_t*>(initial_state_slots.data),
                    static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
                    static_cast<__nv_bfloat16*>(value.data), width, publish, rows);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    const dim3 grid((rows.channels + kDefaultThreads - 1) / kDefaultThreads,
                    static_cast<unsigned>(batch));
    gdn_projected_conv_kernel<Channels, QueryRows, KeyRows, ValueRows, 0>
        <<<grid, kDefaultThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected.data),
            static_cast<const __nv_bfloat16*>(conv_weight.data),
            static_cast<const __nv_bfloat16*>(state_read.data),
            valid_columns.data == nullptr ? nullptr
                                          : static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(initial_state_slots.data),
            static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
            static_cast<__nv_bfloat16*>(value.data), width, publish, rows);
    CUDA_CHECK(cudaGetLastError());
}

template <class Publish>
void dispatch(const Tensor& projected, const Tensor& conv_weight, const Tensor& state_read,
              const Tensor& valid_columns, const Tensor& initial_state_slots, Tensor& query,
              Tensor& key, Tensor& value, Publish publish, cudaStream_t stream) {
    if (projected.ne[0] == 10240 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 6144) {
        launch<10240, 2048, 2048, 6144>(projected, conv_weight, state_read, valid_columns,
                                        initial_state_slots, query, key, value, publish, stream);
        return;
    }
    if (projected.ne[0] == 8192 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 4096) {
        launch<8192, 2048, 2048, 4096>(projected, conv_weight, state_read, valid_columns,
                                       initial_state_slots, query, key, value, publish, stream);
        return;
    }
    // Any other partition, such as a tensor-parallel rank's share, passes its rows at run time.
    if (query.ne[0] <= 0 || key.ne[0] <= 0 || value.ne[0] <= 0 ||
        static_cast<std::int64_t>(query.ne[0]) + key.ne[0] + value.ne[0] != projected.ne[0]) {
        throw std::invalid_argument("GDN projected-conv destinations must cover its channels");
    }
    launch<0, 0, 0, 0>(projected, conv_weight, state_read, valid_columns, initial_state_slots, query,
                       key, value, publish, stream);
}

} // namespace

void gdn_projected_conv_snapshot_launch(const Tensor& projected, const Tensor& conv_weight,
                                        Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_state_slots,
                                        const Tensor& snapshot_base_slots, Tensor& query,
                                        Tensor& key, Tensor& value, cudaStream_t stream) {
    dispatch(projected, conv_weight, conv_states, valid_columns, initial_state_slots, query, key,
             value,
             SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                    static_cast<const std::int32_t*>(snapshot_base_slots.data),
                                    projected.ne[0]},
             stream);
}

void gdn_projected_conv_record_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                      const Tensor& conv_states, const Tensor& valid_columns,
                                      const Tensor& initial_state_slots, Tensor& query, Tensor& key,
                                      Tensor& value, cudaStream_t stream) {
    dispatch(conv_record, conv_weight, conv_states, valid_columns, initial_state_slots, query, key,
             value, NoHistoryPublish{}, stream);
}

} // namespace ninfer::ops::detail
