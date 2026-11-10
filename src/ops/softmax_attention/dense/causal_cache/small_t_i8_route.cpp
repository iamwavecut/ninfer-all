// The INT8-family small-T route key, in a unit of its own: the per-width launch units and the
// dispatcher both read it, and where the launch units live in the Windows kernel DLL each module
// links its own copy of this unit's archive.
#include "ops/softmax_attention/dense/causal_cache/small_t_i8_launch.h"

#include <string>

namespace ninfer::ops::detail {

std::string small_t_i8_route_key(std::int32_t q_heads, KvCacheStorage storage,
                                 std::int32_t width) {
    const char* coding = "int8";
    switch (storage) {
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        coding = "rk8v4";
        break;
    case KvCacheStorage::RotatedLloyd4KeyInt4Value:
        coding = "rk4v4";
        break;
    case KvCacheStorage::RotatedInt4KeyInt4ValueE8:
        coding = "rk4v4-e8";
        break;
    case KvCacheStorage::RotatedE8RootKeyInt4Value:
        coding = "rk2v4-e8";
        break;
    default:
        break;
    }
    return "attn_i8_small/h" + std::to_string(q_heads) + "/" + coding + "/w" +
           std::to_string(width);
}

} // namespace ninfer::ops::detail
