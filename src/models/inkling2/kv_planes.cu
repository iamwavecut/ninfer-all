// Inkling 2's quantized KV planes: kv_store (kernels.h, kv_planes.h). One warp encodes one row of
// one head; the codings are the engine's D128 variants of its KV codecs (kv_codec.cuh).

#include "models/inkling2/kv_planes.h"

#include "models/inkling2/kv_codec.cuh"

#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2::kernels {
namespace {

constexpr int kWarps = 4; // warps per block, one (row, head) unit each

void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling2 ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

template <KvCoding C>
__device__ __forceinline__ void kv_store_row(const KvStoreArgs& a, int unit, int warp, int lane) {
    const int head = unit % a.heads;
    const int r    = unit / a.heads;
    const int slot = physical(a.rows, r);
    const int sequence = row_sequence(a.rows, r);
    const __half* src = a.src + head * a.src_head_stride + std::int64_t(r) * kKvRowDims;
    float x[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { x[i] = __half2float(src[lane + 32 * i]); }
    std::uint8_t* codes  = a.dst.codes + sequence * a.dst.code_seq + head * a.dst.code_head +
                          std::int64_t(slot) * a.dst.code_row;
    std::uint8_t* scales = a.dst.scales + sequence * a.dst.scale_seq + head * a.dst.scale_head +
                           std::int64_t(slot) * a.dst.scale_row;
    kv::encode_row<C>(x, codes, scales, lane);
}

template <KvCoding C>
__launch_bounds__(kWarps * 32) __global__ void kv_store_kernel(KvStoreArgs a) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int unit = blockIdx.x * kWarps + warp;
    if (unit >= a.rows_count * a.heads) { return; }
    kv_store_row<C>(a, unit, warp, lane);
}

// The key and value stores of one visit in one launch: plane 0 encodes with CK, plane 1 with CV
// (the format's key and value codings differ). Every unit's row reads and encodes exactly as
// kv_store_kernel's does, so the planes' bytes are its bit for bit.
template <KvCoding CK, KvCoding CV>
__launch_bounds__(kWarps * 32) __global__ void kv_store_pair_kernel(KvStoreArgs ak, KvStoreArgs av) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool value = int(blockIdx.y) != 0;
    const KvStoreArgs& a = value ? av : ak;
    const int unit = blockIdx.x * kWarps + warp;
    if (unit >= a.rows_count * a.heads) { return; }
    if (value) {
        kv_store_row<CV>(a, unit, warp, lane);
    } else {
        kv_store_row<CK>(a, unit, warp, lane);
    }
}

} // namespace

void kv_store(const KvStoreArgs& args, cudaStream_t stream) {
    const int units = args.rows_count * args.heads;
    if (units <= 0) { return; }
    const dim3 grid((units + kWarps - 1) / kWarps), block(kWarps * 32);
    switch (args.dst.coding) {
    case KvCoding::Int8G64: kv_store_kernel<KvCoding::Int8G64><<<grid, block, 0, stream>>>(args); break;
    case KvCoding::Int4G32: kv_store_kernel<KvCoding::Int4G32><<<grid, block, 0, stream>>>(args); break;
    case KvCoding::Lloyd4: kv_store_kernel<KvCoding::Lloyd4><<<grid, block, 0, stream>>>(args); break;
    case KvCoding::Mxfp8: kv_store_kernel<KvCoding::Mxfp8><<<grid, block, 0, stream>>>(args); break;
    default: throw std::invalid_argument("inkling2 kv_store: not an Inkling 2 plane coding");
    }
    check_launch("kv_store");
}

void kv_store_pair(const KvStoreArgs& key, const KvStoreArgs& value, cudaStream_t stream) {
    const int units_k = key.rows_count * key.heads, units_v = value.rows_count * value.heads;
    if (units_k <= 0 && units_v <= 0) { return; }
    // One launch for int8 G64 and int4 G32 planes (int8, rk8v4); Lloyd4 keys (rk4v4), MXFP8
    // planes and unequal row counts take the two kv_store launches, which decoded rk4v4 faster
    // on an RTX 3090.
    const auto fuses = [](KvCoding c) {
        return c == KvCoding::Int8G64 || c == KvCoding::Int4G32;
    };
    if (units_k != units_v || units_k <= 0 || key.dst.coding == KvCoding::F16 ||
        value.dst.coding == KvCoding::F16 || !fuses(key.dst.coding) || !fuses(value.dst.coding)) {
        kv_store(key, stream);
        kv_store(value, stream);
        return;
    }
    const dim3 grid((units_k + kWarps - 1) / kWarps, 2, 1), block(kWarps * 32);
    const auto launch = [&](auto ck, auto cv) {
        kv_store_pair_kernel<decltype(ck)::value, decltype(cv)::value><<<grid, block, 0, stream>>>(
            key, value);
    };
    switch (key.dst.coding) {
    case KvCoding::Int8G64:
        switch (value.dst.coding) {
        case KvCoding::Int8G64: launch(std::integral_constant<KvCoding, KvCoding::Int8G64>{},
                                       std::integral_constant<KvCoding, KvCoding::Int8G64>{}); break;
        case KvCoding::Int4G32: launch(std::integral_constant<KvCoding, KvCoding::Int8G64>{},
                                       std::integral_constant<KvCoding, KvCoding::Int4G32>{}); break;
        default: throw std::invalid_argument("inkling2 kv_store_pair: unsupported value coding");
        }
        break;
    case KvCoding::Int4G32:
        switch (value.dst.coding) {
        case KvCoding::Int8G64: launch(std::integral_constant<KvCoding, KvCoding::Int4G32>{},
                                       std::integral_constant<KvCoding, KvCoding::Int8G64>{}); break;
        case KvCoding::Int4G32: launch(std::integral_constant<KvCoding, KvCoding::Int4G32>{},
                                       std::integral_constant<KvCoding, KvCoding::Int4G32>{}); break;
        default: throw std::invalid_argument("inkling2 kv_store_pair: unsupported value coding");
        }
        break;
    default: throw std::invalid_argument("inkling2 kv_store_pair: unsupported key coding");
    }
    check_launch("kv_store_pair");
}

} // namespace ninfer::models::inkling2::kernels
