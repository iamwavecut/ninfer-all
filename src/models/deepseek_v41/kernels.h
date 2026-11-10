#pragma once

// DeepSeek-V4.1-Flash's own kernels: hyper-connections, the latent attention's rotations and
// activation roundings, the compressor, the indexer, the sparse attention, routing and the Engram.
// Activations are token-major: the residual stream h[t][stream][i] in FP32, projection inputs as
// BF16 rows x[t][k] (a [K, T] Tensor). Rows of a call belong to one sequence (`sequence`) or, when
// `sequences` is given, each to its own; a sequence's caches lie `seq_stride` elements apart.
//
// Where the reference implementation (DeepSeek's inference/model.py and kernel.py) keeps an
// activation in BF16 between two operations, these kernels round it to BF16 there too, so that the
// FP8 and FP4 roundings it applies before the caches see the same inputs:
//   fp8 (act_quant, ue8m0): per 32 values s = 2^ceil(log2(max(amax, 1e-4) / 448)), each value
//     v / s clamped to +-448 and rounded to E4M3 (to nearest, ties to even), times s;
//   fp4 E8M0 (indexer): per 32 values s = 2^ceil(log2(max(amax, 6 * 2^-126) / 6)), v / s clamped
//     to +-6 and rounded to E2M1 (ties to even), times s;
//   fp4 E4M3 (compressed latents): per 16 values s = e4m3(max(amax, 6 * 2^-9) / 6), likewise.
// Every rounded value is exact in BF16, so the caches hold BF16. The residual stream stays FP32
// (the reference rounds it to BF16 after each block half), and the indexer's scores and the
// attention's probabilities are FP32 (the reference rounds them to BF16).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ninfer::models::deepseek_v41::kernels {

// out[t][m] = sum_k w[m][k] x[t][k] in FP32 for w row-major [m][k] and x [n][k]: up to eight rows
// by the GEMV kernel (a fixed grid, so CUDA graphs replay it), more by cuBLAS SGEMM.
void gemm_f32(int m, int n, int k, const float* w, const float* x, float* out, cudaStream_t stream);

// h[t][s] = h[t][0] for the streams s > 0.
void expand_streams(float* h, int tokens, int streams, int hidden, cudaStream_t stream);

// h[(row + i) * stride + c] = rows[i][c] for `count` BF16 rows of `hidden`: an image's span rows
// in place of their tokens' embeddings.
void place_rows(const __nv_bfloat16* rows, int count, int hidden, float* h, std::int64_t stride,
                cudaStream_t stream);

// The hyper-connection coefficients of a site from its projection `mixes` [T][M] (M = (2 + S) S,
// fn . flatten(h[t])) and the stream h: m = mixes * rsqrt(mean(h[t]^2) + norm_eps), then
//   pre[t][s]     = sigmoid(m[s] scale[0] + base[s]) + hc_eps
//   post[t][s]    = 2 sigmoid(m[S + s] scale[1] + base[S + s])
//   comb[t][j][k] = Sinkhorn of m[2S + jS + k] scale[2] + base[...]: a row softmax plus hc_eps,
//                   a column normalization, then iterations - 1 times a row and a column
//                   normalization (each dividing by its sum plus hc_eps).
struct HcSplitArgs {
    const float* h     = nullptr; // [T][S][H]
    const float* mixes = nullptr; // [T][M]
    int tokens = 0, streams = 4, hidden = 0;
    const float* scale = nullptr; // [3]
    const float* base  = nullptr; // [M]
    float norm_eps = 0, hc_eps = 0;
    int iterations = 20;
    float* pre  = nullptr; // [T][S]
    float* post = nullptr; // [T][S]
    float* comb = nullptr; // [T][S][S]
};
void hc_split(const HcSplitArgs& args, cudaStream_t stream);

// y = bf16(sum_s mix[t][s] h[t][s]) (stream 0 when mix is null); out[t] = bf16(y * rsqrt(mean(y^2)
// + eps) * w), and out_f32 (when given) the same BF16 values in FP32.
void hc_pre_norm(const float* h, const float* mix, const float* w, int tokens, int streams,
                 int hidden, float eps, __nv_bfloat16* out, float* out_f32, cudaStream_t stream);

// h[t][k] = post[t][k] y[t] + sum_j comb[t][j][k] h[t][j], in place.
void hc_post(float* h, const float* y, const float* post, const float* comb, int tokens,
             int streams, int hidden, cudaStream_t stream);

// out[t] = bf16(rms(bf16(x[t])) * w) for rows of n values (x_stride apart), and out_f32 (when given)
// the same BF16 values in FP32 (rows of n).
void rms_rows(const float* x, std::int64_t x_stride, int n, int tokens, const float* w, float eps,
              __nv_bfloat16* out, float* out_f32, cudaStream_t stream);

// The caches keep the release's roundings packed, each value exactly the BF16 value its rounding
// gives (its code's value times its group's scale):
// - a window row: 512 E4M3 codes, then the 16 E8M0 scales of its groups of 32 values;
// - a compressed latent row: 256 bytes of E2M1 codes (value 2i in byte i's low nibble), then the 32
//   E4M3 scales of its groups of 16 values;
// - an indexer key: 64 bytes of E2M1 codes in one plane and the 4 E8M0 scales of its groups of 32
//   values in another.
inline constexpr int kWindowRowBytes = 528;
inline constexpr int kLatentRowBytes = 288;
inline constexpr int kKeyRowBytes    = 64;
inline constexpr int kKeyScaleBytes  = 4;

// The window key/value latent of each row: bf16(rms(bf16(kv)) * norm), its last 64 values turned
// at the row's position (adjacent pairs, angle position * freq[i]), rounded to BF16 and fp8, into
// slot `slot % ring` of the row's sequence's cache ([ring] window rows per sequence, seq_stride
// bytes apart), slot the row's position unless `slots` names another.
struct KvPostArgs {
    const float* kv = nullptr; // [T] rows `kv_stride` apart
    std::int64_t kv_stride = 512;
    int tokens = 0;
    const float* norm = nullptr;
    float eps = 0;
    const std::int32_t* positions = nullptr;
    const std::int32_t* slots     = nullptr;
    const float* freq = nullptr; // [32]
    std::uint8_t* cache = nullptr;
    std::int64_t seq_stride = 0; // bytes
    const std::int32_t* sequences = nullptr;
    int sequence = 0, ring = 0;
};
void kv_post(const KvPostArgs& args, cudaStream_t stream);

// The last 64 of each head's `dim` values of rows [T][heads][dim] turned at the rows' positions:
// bf16(x), turned, bf16 again into out; with `fp4` each 32 values then take the E8M0 fp4 rounding
// (the indexer's queries).
void rotate_heads(const float* x, int tokens, int heads, int dim, const std::int32_t* positions,
                  const float* freq, bool fp4, __nv_bfloat16* out, cudaStream_t stream);
// The same for BF16 rows, in place when out == x.
void rotate_heads(const __nv_bfloat16* x, int tokens, int heads, int dim,
                  const std::int32_t* positions, const float* freq, bool fp4, __nv_bfloat16* out,
                  cudaStream_t stream);

// The compressor of a kv source block. ratio 1: latent[t] = bf16(rms(bf16(kv[t])) * norm).
// ratio 2: a row whose position p completes a group (p odd) pools the group's two rows,
// pooled = sum_i kv_i softmax_i(score) per value, the earlier row taken from the call's previous
// row when that is the same sequence's position p - 1, else from the sequence's tail (`tail_kv`,
// `tail_score`: [slots][512] FP32 per sequence, slot = position % slots);
// latent[t] = bf16(rms(bf16(pooled)) * norm). Rows that complete no group leave their latent row
// alone. compress_tail then keeps the kv and score of rows tail_from.. in their tail slots.
struct CompressArgs {
    const float* kv    = nullptr; // [T][512]
    const float* score = nullptr; // [T][512], ratio 2
    int tokens = 0, ratio = 1;
    const std::int32_t* positions = nullptr;
    const std::int32_t* sequences = nullptr;
    int sequence = 0;
    float* tail_kv    = nullptr;
    float* tail_score = nullptr;
    std::int64_t tail_seq_stride = 0;
    int slots = 16, tail_from = 0;
    const float* norm = nullptr;
    float eps = 0;
    float* latent = nullptr; // [T][512], BF16 values in FP32
};
void compress(const CompressArgs& args, cudaStream_t stream);
void compress_tail(const CompressArgs& args, cudaStream_t stream);

// The caches of a row that completes group j = (p + 1) / ratio - 1 (its first position
// g = j * ratio): the indexer key bf16(rms(bf16(key[t])) * key_norm), turned at g and fp4-rounded
// (E8M0, 32) into key j of index_cache and index_scales; the latent turned at g, rounded to BF16
// and fp4-rounded (E4M3, 16) into compressed row j of kv_cache. Strides in bytes.
struct LatentArgs {
    const float* latent = nullptr; // [T][512]
    const float* key    = nullptr; // [T][128], the indexer key projection of the latent
    int tokens = 0, ratio = 1;
    const std::int32_t* positions = nullptr;
    const std::int32_t* sequences = nullptr;
    int sequence = 0;
    const float* key_norm = nullptr;
    float eps = 0;
    const float* freq = nullptr;
    std::uint8_t* index_cache = nullptr; // [N][kKeyRowBytes] per sequence
    std::int64_t index_seq_stride = 0;
    std::uint8_t* index_scales = nullptr; // [N][kKeyScaleBytes] per sequence
    std::int64_t index_scale_seq_stride = 0;
    std::uint8_t* kv_cache = nullptr; // [N][kLatentRowBytes] per sequence
    std::int64_t kv_seq_stride = 0;
};
void latent_post(const LatentArgs& args, cudaStream_t stream);

// Index scores of each row's visible compressed positions j < (p + 1) / ratio:
//   out[t][j] = sum_h relu(q[t][h] . key[j]) * weights[t][h] * weight_scale
// -inf outside the candidate blocks of `candidates` when given (a bit per block of `block`
// positions, `candidate_words` words per row). Columns past a row's visible ones stay unwritten;
// `width` is the row stride of out and bounds every row's visible count. `group` consecutive rows
// of one sequence share a pass over the keys (1 or 4).
// With `list` (and `candidates`), a row holds only its group's listed blocks (candidate_lists):
// column s is position list[g][s / block] * block + s % block of group g, -inf where the row does
// not keep that block, and `width`, at least capacity * block, bounds the listed columns instead.
struct ScoreArgs {
    const __nv_bfloat16* q = nullptr; // [T][heads][128]
    const float* weights   = nullptr; // [T][heads]
    float weight_scale     = 1;
    int heads = 0, tokens = 0, ratio = 1, width = 0, group = 1;
    const std::int32_t* positions = nullptr;
    const std::int32_t* sequences = nullptr;
    int sequence = 0;
    const std::uint8_t* keys = nullptr; // [N][kKeyRowBytes] per sequence, strides in bytes
    std::int64_t key_seq_stride = 0;
    const std::uint8_t* key_scales = nullptr; // [N][kKeyScaleBytes] per sequence
    std::int64_t key_scale_seq_stride = 0;
    const std::uint32_t* candidates = nullptr;
    int candidate_words = 0, block = 8;
    const std::int32_t* list   = nullptr; // [groups][capacity]
    const std::int32_t* counts = nullptr; // [groups]
    int capacity               = 0;
    float* out = nullptr; // [T][width]
};
void index_scores(const ScoreArgs& args, cudaStream_t stream);

// The blocks that any row of each group of `group` consecutive rows keeps in mask [T][words] (a
// bit per block), in ascending order: list [groups][capacity] and their count, counts [groups].
// `capacity` bounds a group's count (every row keeps at most capacity / group blocks).
void candidate_lists(const std::uint32_t* mask, int words, int tokens, int group, int capacity,
                     std::int32_t* list, std::int32_t* counts, cudaStream_t stream);

// The candidate blocks of each row (scores [T][width] of its visible positions): the `keep` blocks
// of `block` positions with the highest maximum score, the block of its last visible position
// always among them, blocks without a finite score never (lower block first among equal maxima);
// a bit per block in mask [T][words]. `scratch` holds [T][stride] floats, a row's block maxima;
// `stride` is at least the blocks of `width` positions.
void select_candidates(const float* scores, int width, int tokens, const std::int32_t* positions,
                       int ratio, int block, int keep, std::uint32_t* mask, int words,
                       float* scratch, int stride, cudaStream_t stream);

// The k highest finite scores of each row's visible positions (lower position first among equal
// scores), as positions in ascending order, -1 past the row's count: out [T][k].
void select_topk(const float* scores, int width, int tokens, const std::int32_t* positions,
                 int ratio, int k, std::int32_t* out, cudaStream_t stream);

// select_topk over rows of listed blocks (index_scores with `list`): row t's columns stand for the
// listed positions of its group t / group, of which it takes those below its visible count.
struct ListedRows {
    const std::int32_t* list   = nullptr; // [groups][capacity]
    const std::int32_t* counts = nullptr; // [groups]
    int capacity = 0, group = 1, block = 8;
};
void select_topk(const float* scores, int width, int tokens, const std::int32_t* positions,
                 int ratio, int k, const ListedRows& listed, std::int32_t* out,
                 cudaStream_t stream);

// Sparse attention of each row's heads over its window (positions p - window_size + 1 .. p, none
// below window_floor, from slot position % ring of the window cache, window_size at most the ring
// and, beside k selected latents, 1152 - k) and its selected compressed latents
// (compressed[topk[t][i]] for the leading entries that are not negative), every latent its own key
// and value; each head's sink logit joins the softmax denominator:
//   out = sum_j e^(s_j - m) v_j / (sum_j e^(s_j - m) + e^(sink - m)),  s_j = scale q . k_j.
// With splits > 1 each row's keys split over that many key ranges, merged after.
struct AttentionArgs {
    const __nv_bfloat16* q = nullptr; // [T][heads][512]
    int tokens = 0, heads = 0;
    const std::int32_t* positions = nullptr;
    const std::int32_t* sequences = nullptr;
    int sequence = 0;
    const std::uint8_t* window = nullptr; // [ring][kWindowRowBytes] per sequence
    std::int64_t window_seq_stride = 0;   // bytes
    int ring = 0, window_size = 0;
    int window_floor = 0; // the lowest window position a row reads (a prefill tail's first row)
    const std::uint8_t* compressed = nullptr; // [N][kLatentRowBytes] per sequence, or null
    std::int64_t compressed_seq_stride = 0;  // bytes
    const std::int32_t* topk = nullptr; // [T][k]
    int k = 0;
    const float* sinks = nullptr;
    float scale = 1;
    int splits = 1;
    float* partial = nullptr; // splits > 1: sparse_attention_workspace_floats
    __nv_bfloat16* out = nullptr; // [T][heads][512]
};
[[nodiscard]] std::size_t sparse_attention_workspace_floats(int tokens, int heads, int splits);
void sparse_attention(const AttentionArgs& args, cudaStream_t stream);

// The attention output for the grouped output projection: its last 64 values of each head turned
// back at the row's position, rounded to BF16, fp8-rounded per 32 values, into
// out[g][t][heads / groups * 512] for group g of the heads, or with `token_major` out[t][heads * 512].
void attention_output(const __nv_bfloat16* o, int tokens, int heads, int groups,
                      const std::int32_t* positions, const float* freq, bool token_major,
                      __nv_bfloat16* out, cudaStream_t stream);

// The router: scores s = sqrt(softplus(logits)), the top_k experts by s + bias (lower id first on
// equal values), each weighted s / (sum of the chosen s + 1e-20) * scale. ids/weights [T][top_k].
// With `image` (a flag per token), an image token's bias is image_bias.
void route(const float* logits, const float* bias, int tokens, int experts, int top_k, float scale,
           std::int32_t* ids, float* weights, cudaStream_t stream,
           const float* image_bias = nullptr, const std::uint8_t* image = nullptr);

// out[r][i] = bf16(silu(min(g, limit)) * clamp(u, -limit, limit)) for rows of [g (width);
// u (width)] FP32 x, `x_stride` apart, g and u first multiplied by scale[r / per_token] (when
// given) and rounded to BF16; out rows `out_stride` apart.
void clamped_swiglu(const float* x, std::int64_t x_stride, int rows, int width, float limit,
                    const float* scale, int per_token, __nv_bfloat16* out, std::int64_t out_stride,
                    cudaStream_t stream);

// y[t] += the routed experts' output: sum_k weights[t * top_k + k] rows[t * top_k + k], or the
// 2^-32 fixed-point sums fixed[t].
struct MoeCombineArgs {
    float* y = nullptr; // [T][H]
    int tokens = 0, hidden = 0;
    const float* rows    = nullptr; // [T * top_k][H]
    const float* weights = nullptr;
    int top_k            = 0;
    const unsigned long long* fixed = nullptr;
};
void moe_combine(const MoeCombineArgs& args, cudaStream_t stream);

// sums[i] += products[i] for i < n when *flag is nonzero: the 2^-32 fixed-point sums of the
// routed experts the CPU computed for a call (ExpertDelivery), added as the expert kernels add
// theirs, modulo 2^64.
void add_fixed(unsigned long long* sums, const void* products, const std::int32_t* flag,
               std::int64_t n, cudaStream_t stream);

// IQ4_NL rows of 256 values (144 bytes each) as BF16.
void engram_rows(const std::uint8_t* rows, int count, __nv_bfloat16* out, cudaStream_t stream);

// The Engram's addition to each stream: key[s] and value from bf16(kv[t]) ([S][H] keys, then H
// values),
//   dot = sum_i h[s][i] w[s][i] key[s][i] * rsqrt(mean(h[s]^2) + eps) * rsqrt(mean(key[s]^2) + eps)
//         / sqrt(H)
//   h[s] += sigmoid(sign(dot) sqrt(max(|dot|, 1e-6))) * value.
// With `image` (a flag per token), an image token's streams stay as they are.
void engram_gate(float* h, const float* kv, const float* w, int tokens, int streams, int hidden,
                 float eps, cudaStream_t stream, const std::uint8_t* image = nullptr);

// The matrix and vector activations of some GGUF types keep each 32 values' sum in F16
// (gguf::*_activation_f16_sums): a column of the BF16 x [columns][rows] (rows % 32 == 0) whose
// largest such sum passes kF16SumLimit is multiplied by 2^-k, the least power of two that brings it
// under, and scale[c] = 2^k (1 for the others); unscale_columns multiplies the product's columns
// back. Both factors are exact.
inline constexpr float kF16SumLimit = 16384.0f;
__host__ __device__ inline int f16_sum_exponent(float largest) {
    if (!(largest > kF16SumLimit) || !(largest <= 3.40282347e38f)) { return 0; }
    int k = 0;
    frexpf(largest / kF16SumLimit, &k);
    return k;
}
void guard_columns(__nv_bfloat16* x, int rows, int columns, float* scale, cudaStream_t stream);
void unscale_columns(float* out, std::int64_t out_stride, int out_rows, int columns,
                     const float* scale, cudaStream_t stream);
void unscale_columns(__nv_bfloat16* out, std::int64_t out_stride, int out_rows, int columns,
                     const float* scale, cudaStream_t stream);

// Positions [begin, end) of one sequence's window rows (kWindowRowBytes) from a ring of src_ring slots to
// one of dst_ring (position p in slot p % ring of each).
void ring_copy(const std::uint8_t* src, int src_ring, std::uint8_t* dst, int dst_ring, int begin,
               int end, cudaStream_t stream);

void f32_to_bf16(const float* in, __nv_bfloat16* out, std::int64_t n, cudaStream_t stream);

// The DSpark drafter's taps: out[t][i] = sum_s h[t][s][i] / streams for rows of h [T][S][H], out
// rows `out_stride` apart.
void stream_mean(const float* h, int tokens, int streams, int hidden, float* out,
                 std::int64_t out_stride, cudaStream_t stream);

// The rank of the drafter's Markov tables.
inline constexpr int kMarkovRank = 256;

// One greedy step of the drafter's Markov chain: ids[step + 1] = the token v < domain with the
// highest logits[v] + sum_r head[v][r] embed[ids[step]][r] (lower v first among equal values),
// logits FP32 [domain or more], head and embed BF16 [vocab][kMarkovRank]. `scratch` holds
// markov_scratch_bytes().
[[nodiscard]] std::size_t markov_scratch_bytes();
void markov_step(const float* logits, const __nv_bfloat16* head, const __nv_bfloat16* embed,
                 int domain, std::int32_t* ids, int step, void* scratch, cudaStream_t stream);

// The drafter's confidence of each row t: sigmoid(sum_i x[i] w[i] + sum_r embed[ids[t]][r]
// w[hidden + r]) for the FP32 collapse x = sum_s pre[t][s] h[t][s] (h [T][S][H], pre [T][S]).
void dspark_confidence(const float* h, const float* pre, int tokens, int streams, int hidden,
                       const float* w, const __nv_bfloat16* embed, const std::int32_t* ids,
                       float* out, cudaStream_t stream);

// The reference roundings, for tests: one value to E4M3 (|x| <= 448) and to E2M1 (|x| <= 6), to
// nearest with ties to even.
__host__ __device__ inline float round_e4m3(float x) {
    const float a = fabsf(x);
    if (a == 0.0f) { return x; }
    int e = 0;
    frexpf(a, &e);
    int exponent = e - 1;
    if (exponent < -6) { exponent = -6; }
    const float step = ldexpf(1.0f, exponent - 3);
    return copysignf(rintf(a / step) * step, x);
}
__host__ __device__ inline float round_e2m1(float x) {
    const float a = fabsf(x);
    if (a == 0.0f) { return x; }
    int e = 0;
    frexpf(a, &e);
    int exponent = e - 1;
    if (exponent < 0) { exponent = 0; }
    const float step = ldexpf(1.0f, exponent - 1);
    return copysignf(fminf(rintf(a / step) * step, 6.0f), x);
}

} // namespace ninfer::models::deepseek_v41::kernels
