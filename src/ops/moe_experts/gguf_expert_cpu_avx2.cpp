#include "ops/common/silu.h"
#include "ops/moe_experts/gguf_expert_cpu_x86_rows.h"

namespace ninfer::ops::gguf_cpu_detail {
namespace {

// silu.h's e^v, eight values at a time: the same operations in the same order, so the same bits.
inline __m256 exp_ps(__m256 v) {
    const __m256 one = _mm256_set1_ps(1.0F);
    const __m256 n   = _mm256_round_ps(_mm256_mul_ps(v, _mm256_set1_ps(1.44269504F)),
                                       _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fmadd_ps(n, _mm256_set1_ps(-0.693145751953125F), v);
    r        = _mm256_fmadd_ps(n, _mm256_set1_ps(-1.42860677e-6F), r);
    __m256 p = _mm256_fmadd_ps(_mm256_set1_ps(1.9875691500e-4F), r, _mm256_set1_ps(1.3981999507e-3F));
    p        = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3F));
    p        = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2F));
    p        = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1F));
    p        = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1F));
    p        = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), r);
    p        = _mm256_add_ps(p, one);
    // n is -126 .. 128 where the polynomial's value is kept; 2^128 is 2^127 times 2.
    const __m256i k = _mm256_cvttps_epi32(n);
    const __m256i clamped =
        _mm256_min_epi32(_mm256_max_epi32(k, _mm256_set1_epi32(-126)), _mm256_set1_epi32(127));
    const __m256 power = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(clamped, _mm256_set1_epi32(127)), 23));
    const __m256 beyond = _mm256_castsi256_ps(_mm256_cmpgt_epi32(k, _mm256_set1_epi32(127)));
    __m256 e = _mm256_mul_ps(_mm256_mul_ps(p, power), _mm256_blendv_ps(one, _mm256_set1_ps(2.0F), beyond));
    e = _mm256_blendv_ps(e, _mm256_setzero_ps(),
                         _mm256_cmp_ps(v, _mm256_set1_ps(-87.3365479F), _CMP_LT_OQ));
    e = _mm256_blendv_ps(e, _mm256_castsi256_ps(_mm256_set1_epi32(0x7F800000)),
                         _mm256_cmp_ps(v, _mm256_set1_ps(88.7228394F), _CMP_GE_OQ));
    return _mm256_blendv_ps(e, v, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
}

inline __m256 negated(__m256 x) { return _mm256_xor_ps(x, _mm256_set1_ps(-0.0F)); }

} // namespace

void rows_avx2(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
               float* y, std::size_t y_stride) {
    rows_x86(w, x, row_begin, row_end, y, y_stride);
}

// silu.h eight values at a time: the same operations in the same order, so the same bits.
void swiglu_avx2(const float* gate, const float* up, std::size_t count, std::uint16_t* out) {
    const __m256 one = _mm256_set1_ps(1.0F);
    std::size_t i    = 0;
    alignas(32) float products[8];
    for (; i + 8 <= count; i += 8) {
        const __m256 x    = _mm256_loadu_ps(gate + i);
        const __m256 silu = _mm256_div_ps(x, _mm256_add_ps(one, exp_ps(negated(x))));
        _mm256_store_ps(products, _mm256_mul_ps(silu, _mm256_loadu_ps(up + i)));
        for (int j = 0; j < 8; ++j) { out[i + j] = float_to_bf16(products[j]); }
    }
    for (; i < count; ++i) { out[i] = float_to_bf16(ops::portable_silu(gate[i]) * up[i]); }
}

// silu.h's gelu_tanh eight values at a time.
void geglu_avx2(const float* gate, const float* up, std::size_t count, std::uint16_t* out) {
    const __m256 one = _mm256_set1_ps(1.0F);
    std::size_t i    = 0;
    alignas(32) float products[8];
    for (; i + 8 <= count; i += 8) {
        const __m256 x     = _mm256_loadu_ps(gate + i);
        const __m256 cube  = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.044715F), x), x), x);
        const __m256 twice = _mm256_mul_ps(_mm256_set1_ps(1.5957691216057308F), _mm256_add_ps(x, cube));
        const __m256 gelu  = _mm256_div_ps(x, _mm256_add_ps(one, exp_ps(negated(twice))));
        _mm256_store_ps(products, _mm256_mul_ps(gelu, _mm256_loadu_ps(up + i)));
        for (int j = 0; j < 8; ++j) { out[i + j] = float_to_bf16(products[j]); }
    }
    for (; i < count; ++i) { out[i] = float_to_bf16(ops::gelu_tanh(gate[i]) * up[i]); }
}

// DeepSeek-V4.1's clamped middle (clamped_swiglu_scalar) eight values at a time: BF16 rounding to
// nearest even by the word's carry, min and max of finite values, silu.h's SiLU.
void clamped_swiglu_avx2(const float* gate, const float* up, std::size_t count, float scale,
                         float limit, std::uint16_t* out) {
    const __m256 one = _mm256_set1_ps(1.0F), s = _mm256_set1_ps(scale);
    const __m256 high = _mm256_set1_ps(limit), low = _mm256_set1_ps(-limit);
    const auto bf16r = [](__m256 v) {
        const __m256i bits = _mm256_castps_si256(v);
        const __m256i odd  = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
        const __m256i up16 = _mm256_add_epi32(_mm256_add_epi32(bits, _mm256_set1_epi32(0x7FFF)), odd);
        return _mm256_castsi256_ps(_mm256_and_si256(up16, _mm256_set1_epi32(static_cast<int>(0xFFFF0000U))));
    };
    std::size_t i = 0;
    alignas(32) float products[8];
    for (; i + 8 <= count; i += 8) {
        const __m256 g    = _mm256_min_ps(bf16r(_mm256_mul_ps(_mm256_loadu_ps(gate + i), s)), high);
        const __m256 u    = _mm256_min_ps(_mm256_max_ps(bf16r(_mm256_mul_ps(_mm256_loadu_ps(up + i), s)), low), high);
        const __m256 silu = _mm256_div_ps(g, _mm256_add_ps(one, exp_ps(negated(g))));
        _mm256_store_ps(products, _mm256_mul_ps(silu, u));
        for (int j = 0; j < 8; ++j) { out[i + j] = float_to_bf16(products[j]); }
    }
    clamped_swiglu_scalar(gate + i, up + i, count - i, scale, limit, out + i);
}

} // namespace ninfer::ops::gguf_cpu_detail
