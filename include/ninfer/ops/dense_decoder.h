#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {
// FP32 x [T,H] and weight [H], BF16 y [T,H]:
// y=round_bf16(x * weight / sqrt(mean(x*x)+eps)). H is a positive multiple of 32.
// Inputs/output are disjoint and contiguous. FP64 represented-input oracle; BF16 output
// criterion: normalized RMS error <= .004 and max absolute error <= .02*max(1,max|ideal|).
void rmsnorm_f32(const float* x, const float* weight, int hidden, int tokens, float eps,
                 __nv_bfloat16* out, cudaStream_t stream);
// Raw projected FP32 [T,(Hq+2Hkv)*128], no Q/K normalization. Rotate Q/K pairs
// (d,d+64) by position*theta^(-2*d/128). Q remains FP32 [T,Hq,128]; K/V cross an F16
// storage boundary into [Hkv,T,128]. Neither bias, gate nor softmax sink is applied.
// Independent FP64 rotation oracle: FP32 Q <= 2e-5 normalized RMS, F16 K/V <= .001.
void rotary_kv_f16(const float* qkv, const std::int32_t* positions, int tokens, int query_heads,
                   int kv_heads, float theta, float* q, __half* k, __half* v, cudaStream_t stream);
// Causal GQA over a segment of already stored F16 keys/values. D=128, Hq % Hkv=0.
// q [T,Hq,128], k/v [Hkv,head_stride,128], head_stride >= N, positions [T] absolute; segment begins
// at first_key. A query sees precisely segment keys whose absolute position <= positions[t]. Each
// query head reads KV head floor(h/(Hq/Hkv)); scores=dot(q,k)/sqrt(128). state [T,Hq,130] holds
// unnormalized weighted values, max score and exponential sum. initialize clears the state; later
// disjoint segments merge into it in caller order. finish writes BF16 normalized output [T,Hq,128];
// every query must then have a visible key. This Op owns no position frontier, allocation, transfer
// or request policy. q/k/v/positions, state, partial workspace and output are disjoint. Workspace
// bytes below. Independent FP64 softmax oracle over represented F16 cache rows, FP32 Q: output
// normalized RMS <= .006 and max_abs <= .03*max(1,max|ideal|).
std::size_t segment_attention_workspace_bytes(int tokens, int query_heads, int keys);
void segment_attention_f16(const float* q, const __half* k, const __half* v,
                           const std::int32_t* positions, int tokens, int query_heads, int kv_heads,
                           int first_key, int keys, int head_stride, float* state, void* workspace,
                           bool initialize, bool finish, __nv_bfloat16* out, cudaStream_t stream);
} // namespace ninfer::ops
