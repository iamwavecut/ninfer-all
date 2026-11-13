#pragma once

// ninfer::ops::detail - the causal attention route of a call (small-T, chunked small-T or prompt)
// and the split capacity of its small-T launches. Host code only.

#include "ninfer/kv_cache_storage.h"
#include "ninfer/ops/softmax_attention_envelope.h"
#include "ops/softmax_attention/dense/causal_cache/int8/coding.h"

#include <cstdint>

namespace ninfer::ops::detail {

enum class CausalAttentionRoute { SmallT, ChunkedSmallT, Prompt };

std::int32_t causal_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                             KvCacheStorage cache_storage,
                                             CausalAttentionExecutionEnvelope envelope,
                                             std::int32_t batch_size = 1);

CausalAttentionRoute causal_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                                    std::int32_t batch_size, KvCacheStorage storage,
                                                    CausalAttentionExecutionEnvelope envelope);

const char* causal_attention_route_name(CausalAttentionRoute route);

// The coding of an INT8-family storage whose value plane is packed int4 (a U8 plane) or not. A U8
// key plane is a packed key coding, told apart by storage.
[[nodiscard]] constexpr CausalI8Coding causal_i8_coding(KvCacheStorage storage,
                                                        bool packed_values) {
    switch (storage) {
    case KvCacheStorage::RotatedLloyd4KeyInt4Value:
        return CausalI8Coding::Lloyd4;
    case KvCacheStorage::RotatedInt4KeyInt4ValueE8:
        return CausalI8Coding::Int4E8;
    case KvCacheStorage::RotatedE8RootKeyInt4Value:
        return CausalI8Coding::RootE8;
    default:
        return packed_values ? CausalI8Coding::PackedValues : CausalI8Coding::Int8;
    }
}

} // namespace ninfer::ops::detail
