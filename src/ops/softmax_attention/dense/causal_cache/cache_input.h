#pragma once

// Where a causal small-T launch takes the keys and values of its own columns: the kernels' CacheInput
// argument. Host routers and the kernel units both name these types; the header carries no device
// code.

#include <cstdint>

struct __nv_bfloat16;

namespace ninfer::ops {

// The call's rows arrive as BF16 K and V, which the kernel appends to the cache before it reads them.
struct CausalAppendInput {
    static constexpr bool writes_cache = true;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
};

// The call's rows are in the cache already.
struct CausalCachedInput {
    static constexpr bool writes_cache = false;
};

// A packed BF16 cache (KvCacheStorage::Bf16Packed): the kernel's K and V planes hold 384-byte rows
// and these their side records (ops/kv_cache/bf16p_codec.cuh). The call's rows were appended before
// it, so it only reads.
struct CausalPackedInput {
    static constexpr bool writes_cache = false;
    static constexpr bool packed       = true;
    const std::uint8_t* k_side;
    const std::uint8_t* v_side;
};

template <class CacheInput>
inline constexpr bool kCausalPackedInput = requires { CacheInput::packed; };

} // namespace ninfer::ops
