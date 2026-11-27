#pragma once

// Inkling 2's quantized KV planes: storing attention_inputs' F16 rows into them (kv_codec.cuh
// encodes one row per warp). Reading them back happens in the attention's tile loader
// (kernels.h), which decodes eight-value chunks as it loads a tile — there is no F16 copy of a
// quantized cache.

#include "models/inkling2/kernels.h"
#include "models/inkling2/kv_format.h"

namespace ninfer::models::inkling2::kernels {

// Stores F16 rows into a quantized plane: row r's values, head h at src + h * src_head_stride +
// r * 128 (KvArgs's `row_slots` layout), head h's plane row of position physical(rows, r) at slot
// physical(rows, r) (a full-attention visit: ring 0) of sequence row_sequence(rows, r).
struct KvStoreArgs {
    Rows rows;
    const __half* src = nullptr;
    std::int64_t src_head_stride = 0;
    int rows_count = 0, heads = 0;
    KvPlaneView dst;
};
void kv_store(const KvStoreArgs& args, cudaStream_t stream);

// The key and value stores of one visit, the planes' bytes kv_store's bit for bit: one launch for
// int8 G64 and int4 G32 planes holding the same rows (int8, rk8v4), else kv_store's two launches.
void kv_store_pair(const KvStoreArgs& key, const KvStoreArgs& value, cudaStream_t stream);
} // namespace ninfer::models::inkling2::kernels
