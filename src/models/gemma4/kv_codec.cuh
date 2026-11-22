#pragma once

// Device side of Gemma 4's quantized KV planes (kv_format.h): the engine's row codings at D = 256
// (ops/kv_cache/kv_row_codec.cuh), a warp encoding or decoding one 256-wide block and the
// eight-value chunk reads of the decode attention's tile loads. Only the addressing is Gemma's own:
// a row's codes and scales are contiguous, block after block, and chunk c (dimensions [8c, 8c + 8))
// of a row has its codes at byte c * bytes-per-chunk and its scale at index c / chunks-per-group of
// the row's scales, whatever block it is in.

#include "models/gemma4/kv_format.h"
#include "ops/kv_cache/kv_row_codec.cuh"

namespace ninfer::models::gemma4::kv {

using ops::kv_row::Chunk;
using ops::kv_row::decode_block;
using ops::kv_row::encode_block;
using ops::kv_row::pack8;

} // namespace ninfer::models::gemma4::kv
