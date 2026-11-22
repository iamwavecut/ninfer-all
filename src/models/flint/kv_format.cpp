#include "models/flint/kv_format.h"

#include "models/kv_storage_name.h"

#include <stdexcept>
#include <string>

namespace ninfer::models::flint {

KvFormat kv_format(KvCacheStorage storage) {
    switch (storage) {
    // The unquantized cache is F16, which --kv-dtype fp16 names and bf16 (the default) selects.
    case KvCacheStorage::BFloat16:
    case KvCacheStorage::Float16: return {storage, KvCoding::F16, KvCoding::F16};
    case KvCacheStorage::Int8Group64: return {storage, KvCoding::Int8G64, KvCoding::Int8G64};
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return {storage, KvCoding::Int8G64, KvCoding::Int4G32};
    case KvCacheStorage::RotatedLloyd4KeyInt4Value:
        return {storage, KvCoding::Lloyd4, KvCoding::Int4G32, true};
    case KvCacheStorage::Fp8E4M3Row256:
    case KvCacheStorage::Nvfp4Group16:
    case KvCacheStorage::Fp8KeyNvfp4Value:
    case KvCacheStorage::RotatedInt4KeyInt4ValueE8:
    case KvCacheStorage::RotatedE8RootKeyInt4Value:
    case KvCacheStorage::Bf16Packed:
    case KvCacheStorage::Mxfp8:
        throw std::invalid_argument("Flint: --kv-dtype " + std::string(kv_storage_name(storage)) +
                                    " is not built for this family (int8, rk8v4, rk4v4, or the "
                                    "F16 cache: bf16, the default, or fp16)");
    }
    throw std::invalid_argument("Flint: unknown KV-cache storage");
}

} // namespace ninfer::models::flint
