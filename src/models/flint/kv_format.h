#pragma once

// Flint's KV-cache storage (--kv-dtype). The 14 full-attention blocks, whose caches grow with the
// context, store a quantized format's 128-wide rows (one KV head of one position) in the engine's
// row codings (ops/kv_cache/row128.h), codes and scales in two planes per K and V:
//
//   int8    Int8G64 keys and values
//   rk8v4   Int8G64 keys; Int4G32 values
//   rk4v4   Lloyd4 keys, rotated; Int4G32 values
//
// The full blocks carry no rotary positions, so a plane's keys can be rotated (row128.h: each key
// row stored as H k / 8, the queries turned by H / 16). Over WikiText's 61,623 tokens that lowered
// rk4v4's mean NLL from 2.976107 to 2.954204 (F16: 2.959979) and raised rk8v4's from 2.959716 to
// 2.962335, so only the Lloyd-Max keys are rotated; values never are. The sliding blocks' rings of
// 768 positions and the DSpine drafter's caches stay F16 whatever the flag says: a ring's size does
// not grow with the context.

#include "ninfer/kv_cache_storage.h"
#include "ops/kv_cache/row128.h"

namespace ninfer::models::flint {

using ops::kv128::kv_code_bytes;
using ops::kv128::kv_scale_bytes;
using ops::kv128::KvCoding;
using ops::kv128::KvPlaneView;

struct KvFormat {
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    KvCoding key = KvCoding::F16, value = KvCoding::F16;
    bool rotated_keys = false;
    [[nodiscard]] bool quantized() const noexcept { return key != KvCoding::F16; }
};

// The format --kv-dtype names; refuses, by name, one Flint does not store.
[[nodiscard]] KvFormat kv_format(KvCacheStorage storage);

} // namespace ninfer::models::flint
