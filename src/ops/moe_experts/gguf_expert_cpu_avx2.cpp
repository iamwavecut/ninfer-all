#include "ops/moe_experts/gguf_expert_cpu.h"

#include "core/qtype.h"
#include "ops/common/silu.h"
#include "ops/moe_experts/gguf_cpu_tables.h"

#include <immintrin.h>

#include <cstring>
#include <type_traits>

// The scalar products' arithmetic (gguf_expert_cpu.cpp) eight slices at a time. Group g holds
// slices 8 g .. 8 g + 7, which are GPU lanes 8 (g mod 4) .. 8 (g mod 4) + 7: one vector of each
// token's four holds their partial sums, and groups g and g + 4 add to it in slice order.

namespace ninfer::ops::gguf_cpu_detail {
namespace {

inline float half_at(const std::uint8_t* p) {
    std::uint16_t bits;
    std::memcpy(&bits, p, 2);
    // The vector F16C form: MSVC does not declare the scalar _cvtsh_ss.
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

// Unsigned codes times signed activations: eight int32 lanes, each the sum of four adjacent
// products.
inline __m256i dot_u8s8(__m256i codes, __m256i a) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(codes, a), _mm256_set1_epi16(1));
}

inline std::uint32_t load32(const std::uint8_t* p) {
    std::uint32_t value;
    std::memcpy(&value, p, 4);
    return value;
}

// 32 bits, bit j for byte j, as 32 bytes of -1 (bit set) or 0: ggml's bit-selector expansion
// (each byte broadcast to eight lanes, masked by its bit, compared).
inline __m256i bit_bytes(std::uint32_t bits) {
    const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                                            2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i select = _mm256_set1_epi64x(0x8040201008040201LL);
    const __m256i set    = _mm256_and_si256(
        _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(bits)), spread), select);
    return _mm256_cmpeq_epi8(set, select);
}

// 32 sign bits, bit j for value j, as 32 bytes of -1 (bit set) or +1.
inline __m256i sign_bytes(std::uint32_t bits) {
    return _mm256_or_si256(bit_bytes(bits), _mm256_set1_epi8(1));
}

// The 32 bytes at `p` shifted right by `shift` bits each and masked by `mask`.
inline __m256i bits_of(const std::uint8_t* p, int shift, int mask) {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    return _mm256_and_si256(_mm256_srl_epi16(v, _mm_cvtsi32_si128(shift)), _mm256_set1_epi8(char(mask)));
}

// ggml's 16 IQ4_NL values for 16 bytes of two 4-bit indices: the low nibbles are values 0..15,
// the high ones 16..31.
inline __m256i iq4_values(const std::uint8_t* q) {
    const __m128i table = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q));
    const __m128i nib   = _mm_set1_epi8(15);
    return _mm256_set_m128i(_mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(bytes, 4), nib)),
                            _mm_shuffle_epi8(table, _mm_and_si128(bytes, nib)));
}

// A group's eight slices: per slice magnitudes (unsigned) and signs (bytes whose sign is the
// weight's, never zero where the weight is not), so that its products are
// maddubs(magnitude, sign(a, sign)), and its factors (decode_slice's). A slice past the row's end
// has factors 0 and no codes. Signs and magnitudes from signed codes c: abs(c) and c (a zero weight
// zeroes its activation, adding nothing).
struct Group {
    __m256i magnitude[8], sign[8];
    __m256 factor[2];
};

// Eight unsigned 8-bit or 16-bit values at `p` as eight int32 lanes.
inline __m256i u8x8(const std::uint8_t* p) {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
}
inline __m256i u16x8(const std::uint8_t* p) {
    return _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

// float(2 s + 1) of eight int32 lanes, exactly.
inline __m256 odd_floats(__m256i s) {
    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_slli_epi32(s, 1), _mm256_set1_epi32(1)));
}

// Lanes i < live set (all bits), the others clear.
inline __m256i live_mask(int live) {
    return _mm256_cmpgt_epi32(_mm256_set1_epi32(live), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
}

// Eight slices' binary16 multipliers as floats (_mm256_cvtph_ps converts each exactly, as half_at
// does): slice i's at `p + (i >> shift) * stride` for i < live, zero past it. Built in a register:
// eight narrow stores read back as one vector would stall the load.
inline __m256 halves_at(const std::uint8_t* p, std::size_t stride, int shift, int live) {
    const auto word = [&](int i) -> std::uint32_t {
        std::uint16_t bits;
        std::memcpy(&bits, p + std::size_t(i) * stride, 2);
        return bits;
    };
    if (live == 8) {
        // Two words to a lane: slices 2 j, 2 j + 1, or (shift 1) word j for both.
        if (shift == 1) {
            const __m128i four = _mm_setr_epi32(int(word(0) | word(1) << 16), int(word(2) | word(3) << 16), 0, 0);
            return _mm256_cvtph_ps(_mm_unpacklo_epi16(four, four));
        }
        return _mm256_cvtph_ps(_mm_setr_epi32(int(word(0) | word(1) << 16), int(word(2) | word(3) << 16),
                                              int(word(4) | word(5) << 16), int(word(6) | word(7) << 16)));
    }
    const auto h = [&](int i) -> short { return i < live ? static_cast<short>(word(i >> shift)) : short(0); };
    return _mm256_cvtph_ps(_mm_setr_epi16(h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7)));
}

// Eight 4-bit fields of the 32-bit word at `p`, lane i bits 4 i .. 4 i + 3.
inline __m256i nibbles8(const std::uint8_t* p) {
    return _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(load32(p))),
                                              _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28)),
                            _mm256_set1_epi32(15));
}

// The second 32-bit word of each of eight 8-byte records from `p`.
inline __m256i odd_words(const std::uint8_t* p) {
    const __m256 a = _mm256_loadu_ps(reinterpret_cast<const float*>(p));
    const __m256 b = _mm256_loadu_ps(reinterpret_cast<const float*>(p + 32));
    // Records 0, 1, 4, 5 in the low half and 2, 3, 6, 7 in the high one; the permute orders them.
    const __m256 odd_lanes = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));
    return _mm256_permute4x64_epi64(_mm256_castps_si256(odd_lanes), _MM_SHUFFLE(3, 1, 2, 0));
}

// The integer dots of a group's eight slices for one token, lane i slice i's: its first 16 values
// and its last 16 (IQ2_S's and Q6_K's halves, which have their own factors), or its whole dot.
struct Dots {
    __m256i first, second;
};

// The dots from each slice's eight int32 lanes of four products each (lanes 0..3: values 0..15).
// hadd sums adjacent lanes of two slices within each 128-bit half, so after two rounds the low half
// holds lanes 0..3 of four slices and the high half their lanes 4..7; the permutes put slices 0..7
// in order.
inline Dots halves8(const __m256i* u) {
    const __m256i h0123 =
        _mm256_hadd_epi32(_mm256_hadd_epi32(u[0], u[1]), _mm256_hadd_epi32(u[2], u[3]));
    const __m256i h4567 =
        _mm256_hadd_epi32(_mm256_hadd_epi32(u[4], u[5]), _mm256_hadd_epi32(u[6], u[7]));
    return {_mm256_permute2x128_si256(h0123, h4567, 0x20),
            _mm256_permute2x128_si256(h0123, h4567, 0x31)};
}

// The warp butterfly over a token's 32 lanes (four vectors of eight): lane l takes lane l + 16,
// then + 8, + 4, + 2, + 1.
inline float butterfly8(const __m256* lanes) {
    const __m256 c = _mm256_add_ps(_mm256_add_ps(lanes[0], lanes[2]), _mm256_add_ps(lanes[1], lanes[3]));
    const __m128 d = _mm_add_ps(_mm256_castps256_ps128(c), _mm256_extractf128_ps(c, 1));
    const __m128 e = _mm_add_ps(d, _mm_movehl_ps(d, d));
    return _mm_cvtss_f32(_mm_add_ss(e, _mm_shuffle_ps(e, e, 1)));
}

// Eight per-slice values from `from`: the live ones, zero for slices past the row's end (a
// group of a 640-value row holds four).
inline __m256 live_floats(const float* from, int live) {
    if (live == 8) { return _mm256_loadu_ps(from); }
    return _mm256_maskload_ps(from, live_mask(live));
}

inline __m256i live_ints(const std::int32_t* from, int live) {
    if (live == 8) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(from)); }
    return _mm256_maskload_epi32(reinterpret_cast<const int*>(from), live_mask(live));
}

constexpr int kMaxTokens = 16;

// A slice's product from its dots (decode_slice's factors f0, f1 and the activation's scale a):
// Whole (f0 a) dot; Halves IQ2_S's and IQ1_M's a fma(f0, dot0, f1 dot1); SixK Q6_K's
// a (f0 dot0 + f1 dot1); Minimum the K-quants' (f0 a) dot - f1 s and Delta IQ1_S's
// f0 fma(a, dot, f1 s), s the activation's value sum.
enum class Form { Whole, Halves, SixK, Minimum, Delta };

// Token t's products of group g's slices from their integer dots `d` (Dots for the halves'
// forms, else the whole dots) and the group's factors.
template <Form kForm, class D>
inline __m256 slice_products(const GgufCpuActivation& x, int t, int g, int live, const Group& group,
                             const D& d) {
    const __m256 a = live_floats(x.scales(t) + 8 * g, live);
    if constexpr (kForm == Form::Halves) {
        const __m256 inner = _mm256_fmadd_ps(group.factor[0], _mm256_cvtepi32_ps(d.first),
                                             _mm256_mul_ps(group.factor[1], _mm256_cvtepi32_ps(d.second)));
        return _mm256_mul_ps(a, inner);
    } else if constexpr (kForm == Form::Delta) {
        const __m256 sums = live_floats(x.value_sums(t) + 8 * g, live);
        return _mm256_mul_ps(group.factor[0], _mm256_fmadd_ps(a, _mm256_cvtepi32_ps(d),
                                                              _mm256_mul_ps(group.factor[1], sums)));
    } else if constexpr (kForm == Form::SixK) {
        const __m256 low  = _mm256_mul_ps(group.factor[0], _mm256_cvtepi32_ps(d.first));
        const __m256 high = _mm256_mul_ps(group.factor[1], _mm256_cvtepi32_ps(d.second));
        return _mm256_mul_ps(a, _mm256_add_ps(low, high));
    } else {
        const __m256 product = _mm256_mul_ps(_mm256_mul_ps(group.factor[0], a), _mm256_cvtepi32_ps(d));
        if constexpr (kForm == Form::Minimum) {
            return _mm256_sub_ps(product,
                                 _mm256_mul_ps(group.factor[1], live_floats(x.value_sums(t) + 8 * g, live)));
        } else {
            return product;
        }
    }
}

// Every format's rows: `decode(row, g, live, group)` fills group g's slices (live of them) and
// `dot(t, g, live, group)` gives token t's integer dots of them (Dots for the halves' forms, else
// the whole dots). With one token, `single(row, g, live, group)` (if given) fills only the factors
// and returns that token's dots, decoding each slice as it goes.
template <Form kForm, class Decode, class Dot, class Single = std::nullptr_t>
void rows_in_lane_order(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin,
                        int row_end, float* y, std::size_t y_stride, const Decode& decode,
                        const Dot& dot, const Single& single = nullptr) {
    const int tokens = x.tokens();
    const int slices = w.k / 32;
    const int groups = (slices + 7) / 8;
    Group group;
    if constexpr (!std::is_same_v<Single, std::nullptr_t>) {
        if (tokens == 1) {
            for (int row = row_begin; row < row_end; ++row) {
                const std::uint8_t* r = w.data + std::size_t(row) * w.row_bytes;
                __m256 acc[4]         = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                                         _mm256_setzero_ps()};
                for (int g = 0; g < groups; ++g) {
                    const int live = slices - 8 * g < 8 ? slices - 8 * g : 8;
                    const auto d   = single(r, g, live, group);
                    acc[g & 3]     = _mm256_add_ps(acc[g & 3], slice_products<kForm>(x, 0, g, live, group, d));
                }
                y[row] = butterfly8(acc);
            }
            return;
        }
    }
    __m256 acc[kMaxTokens][4];
    for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t* r = w.data + std::size_t(row) * w.row_bytes;
        for (int t = 0; t < tokens; ++t) {
            for (__m256& v : acc[t]) { v = _mm256_setzero_ps(); }
        }
        for (int g = 0; g < groups; ++g) {
            const int live = slices - 8 * g < 8 ? slices - 8 * g : 8;
            decode(r, g, live, group);
            for (int t = 0; t < tokens; ++t) {
                acc[t][g & 3] =
                    _mm256_add_ps(acc[t][g & 3], slice_products<kForm>(x, t, g, live, group, dot(t, g, live, group)));
            }
        }
        for (int t = 0; t < tokens; ++t) { y[std::size_t(t) * y_stride + row] = butterfly8(acc[t]); }
    }
}

// IQ1_M's block multiplier: a nibble of each of its four scale words.
inline float iq1m_multiplier(const std::uint8_t* block) {
    std::uint16_t sc[4];
    std::memcpy(sc, block + 48, 8);
    const auto bits = std::uint16_t((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0U) | ((sc[2] >> 4) & 0x0F00U) |
                                    (sc[3] & 0xF000U));
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

// An i-quant block's eight slices' factors, decode_slice's products in its order.
template <QType F>
inline void iq_factors(const std::uint8_t* block, Group& out) {
    const __m256 d = _mm256_set1_ps(F == QType::GGUF_IQ1_M ? iq1m_multiplier(block) : half_at(block));
    const auto times = [](__m256 a, __m256 b) { return _mm256_mul_ps(a, b); };
    const __m256 eighth = _mm256_set1_ps(0.125F);
    if constexpr (F == QType::GGUF_IQ1_S) {
        const __m256i qh = u16x8(block + 34);
        out.factor[0] = times(d, odd_floats(_mm256_and_si256(_mm256_srli_epi32(qh, 12), _mm256_set1_epi32(7))));
        // Bit 15 of qh as the sign bit picks -1.125.
        out.factor[1] = _mm256_blendv_ps(_mm256_set1_ps(-0.875F), _mm256_set1_ps(-1.125F),
                                         _mm256_castsi256_ps(_mm256_slli_epi32(qh, 16)));
    } else if constexpr (F == QType::GGUF_IQ1_M) {
        // Slice s's six bits at bit 6 (s mod 2) of scale word s / 2.
        const __m128i words = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(block + 48));
        const __m256i tmp   = _mm256_srlv_epi32(_mm256_cvtepu16_epi32(_mm_unpacklo_epi16(words, words)),
                                                _mm256_setr_epi32(0, 6, 0, 6, 0, 6, 0, 6));
        const __m256i seven = _mm256_set1_epi32(7);
        out.factor[0] = times(times(d, odd_floats(_mm256_and_si256(tmp, seven))), eighth);
        out.factor[1] = times(times(d, odd_floats(_mm256_and_si256(_mm256_srli_epi32(tmp, 3), seven))), eighth);
    } else if constexpr (F == QType::GGUF_IQ2_XXS) {
        out.factor[0] = times(times(d, odd_floats(_mm256_srli_epi32(odd_words(block + 2), 28))), eighth);
    } else if constexpr (F == QType::GGUF_IQ2_S) {
        const __m256i scale = u8x8(block + 74);
        out.factor[0] = times(times(d, odd_floats(_mm256_and_si256(scale, _mm256_set1_epi32(15)))), eighth);
        out.factor[1] = times(times(d, odd_floats(_mm256_srli_epi32(scale, 4))), eighth);
    } else if constexpr (F == QType::GGUF_IQ3_XXS) {
        const __m256i aux = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + 66));
        out.factor[0] = times(times(d, odd_floats(_mm256_srli_epi32(aux, 28))), _mm256_set1_ps(0.25F));
    } else if constexpr (F == QType::GGUF_IQ3_S) {
        out.factor[0] = times(d, odd_floats(nibbles8(block + 106)));
    } else {
        static_assert(F == QType::GGUF_IQ4_XS);
        std::uint16_t high;
        std::memcpy(&high, block + 2, 2);
        const __m256i top = _mm256_and_si256(
            _mm256_srlv_epi32(_mm256_set1_epi32(high), _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14)),
            _mm256_set1_epi32(3));
        const __m256i scale = _mm256_sub_epi32(_mm256_or_si256(nibbles8(block + 4), _mm256_slli_epi32(top, 4)),
                                               _mm256_set1_epi32(32));
        out.factor[0] = times(d, _mm256_cvtepi32_ps(scale));
    }
}

inline __m128i u8x4(const std::uint8_t* p) {
    return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(static_cast<int>(load32(p))));
}

// IQ1_M forms its grid indices in a register: a slice's two qh bytes hold a nibble a grid row,
// the index's top three bits and the row's offset (the other i-quants form theirs one at a time,
// which measured faster for them). The qh nibbles of slice `sub`, lane l the l-th row's.
inline __m128i iq1m_nibbles(const std::uint8_t* block, int sub) {
    std::uint16_t qh;
    std::memcpy(&qh, block + 32 + 2 * sub, 2);
    return _mm_srlv_epi32(_mm_set1_epi32(qh), _mm_setr_epi32(0, 4, 8, 12));
}
inline __m128i iq1m_index(const std::uint8_t* block, int sub, __m128i nibbles) {
    return _mm_or_si128(u8x4(block + 4 * sub), _mm_slli_epi32(_mm_and_si128(nibbles, _mm_set1_epi32(7)), 8));
}
// IQ1_M's codes from its grid rows: eight times each (at most 16: no bit leaves its byte) less 7,
// or 9 where the row's nibble has bit 3.
inline __m256i iq1m_values(__m256i rows, __m128i nibbles) {
    const __m256i nine = _mm256_cmpeq_epi64(_mm256_cvtepi32_epi64(_mm_and_si128(nibbles, _mm_set1_epi32(8))),
                                            _mm256_set1_epi64x(8));
    const __m256i offsets = _mm256_add_epi8(_mm256_set1_epi8(7), _mm256_and_si256(nine, _mm256_set1_epi8(2)));
    return _mm256_sub_epi8(_mm256_slli_epi16(_mm256_add_epi8(rows, _mm256_set1_epi8(1)), 3), offsets);
}
// IQ1_M's four grid rows by loads, row l at index lane l (stored, then read back as scalars).
inline __m256i load64_rows(const std::uint64_t* table, __m128i index) {
    alignas(16) std::int32_t at[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(at), index);
    return _mm256_setr_epi64x(static_cast<long long>(table[at[0]]), static_cast<long long>(table[at[1]]),
                              static_cast<long long>(table[at[2]]), static_cast<long long>(table[at[3]]));
}

// Slice `sub` of an i-quant block: its magnitudes and signs.
template <QType F>
inline void iq_slice(const std::uint8_t* block, int sub, __m256i& magnitude, __m256i& sign) {
    if constexpr (F == QType::GGUF_IQ1_S) {
        const std::uint8_t* qs = block + 2 + 4 * sub;
        std::uint16_t qh;
        std::memcpy(&qh, block + 34 + 2 * sub, 2);
        // Index l: qs[l] and bits 3 l .. 3 l + 2 of qh at bit 8; the codes 0..2 are the grid's
        // values plus one.
        magnitude = _mm256_add_epi8(
            _mm256_setr_epi64x(static_cast<long long>(iq1s_grid[qs[0] | ((unsigned(qh) << 8) & 0x700U)]),
                               static_cast<long long>(iq1s_grid[qs[1] | ((unsigned(qh) << 5) & 0x700U)]),
                               static_cast<long long>(iq1s_grid[qs[2] | ((unsigned(qh) << 2) & 0x700U)]),
                               static_cast<long long>(iq1s_grid[qs[3] | ((unsigned(qh) >> 1) & 0x700U)])),
            _mm256_set1_epi8(1));
    } else if constexpr (F == QType::GGUF_IQ1_M) {
        const __m128i nibbles = iq1m_nibbles(block, sub);
        const __m256i value   = iq1m_values(load64_rows(iq1s_grid, iq1m_index(block, sub, nibbles)), nibbles);
        magnitude             = _mm256_sign_epi8(value, value);
        sign                  = value; // never zero
    } else if constexpr (F == QType::GGUF_IQ2_XXS || F == QType::GGUF_IQ3_XXS) {
        const std::uint8_t* qs = block + 2 + 8 * sub;
        std::uint32_t aux;
        if constexpr (F == QType::GGUF_IQ2_XXS) {
            aux       = load32(block + 6 + 8 * sub);
            magnitude = _mm256_set_epi64x(
                static_cast<long long>(iq2xxs_grid[qs[3]]), static_cast<long long>(iq2xxs_grid[qs[2]]),
                static_cast<long long>(iq2xxs_grid[qs[1]]), static_cast<long long>(iq2xxs_grid[qs[0]]));
        } else {
            aux       = load32(block + 66 + 4 * sub);
            magnitude = _mm256_set_epi32(
                static_cast<int>(iq3xxs_grid[qs[7]]), static_cast<int>(iq3xxs_grid[qs[6]]),
                static_cast<int>(iq3xxs_grid[qs[5]]), static_cast<int>(iq3xxs_grid[qs[4]]),
                static_cast<int>(iq3xxs_grid[qs[3]]), static_cast<int>(iq3xxs_grid[qs[2]]),
                static_cast<int>(iq3xxs_grid[qs[1]]), static_cast<int>(iq3xxs_grid[qs[0]]));
        }
        const std::uint32_t bits = std::uint32_t(ksigns_iq2xs[aux & 127U]) |
                                   std::uint32_t(ksigns_iq2xs[(aux >> 7) & 127U]) << 8 |
                                   std::uint32_t(ksigns_iq2xs[(aux >> 14) & 127U]) << 16 |
                                   std::uint32_t(ksigns_iq2xs[(aux >> 21) & 127U]) << 24;
        sign = sign_bytes(bits);
    } else if constexpr (F == QType::GGUF_IQ2_S) {
        const std::uint8_t* qs = block + 2 + 4 * sub;
        const unsigned qh      = block[66 + sub];
        magnitude              = _mm256_setr_epi64x(static_cast<long long>(iq2s_grid[qs[0] | ((qh << 8) & 0x300U)]),
                                                    static_cast<long long>(iq2s_grid[qs[1] | ((qh << 6) & 0x300U)]),
                                                    static_cast<long long>(iq2s_grid[qs[2] | ((qh << 4) & 0x300U)]),
                                                    static_cast<long long>(iq2s_grid[qs[3] | ((qh << 2) & 0x300U)]));
        sign = sign_bytes(load32(block + 34 + 4 * sub));
    } else if constexpr (F == QType::GGUF_IQ3_S) {
        const std::uint8_t* qs = block + 2 + 8 * sub;
        const unsigned qh      = block[66 + sub];
        magnitude              = _mm256_setr_epi32(static_cast<int>(iq3s_grid[qs[0] | ((qh << 8) & 256U)]),
                                                   static_cast<int>(iq3s_grid[qs[1] | ((qh << 7) & 256U)]),
                                                   static_cast<int>(iq3s_grid[qs[2] | ((qh << 6) & 256U)]),
                                                   static_cast<int>(iq3s_grid[qs[3] | ((qh << 5) & 256U)]),
                                                   static_cast<int>(iq3s_grid[qs[4] | ((qh << 4) & 256U)]),
                                                   static_cast<int>(iq3s_grid[qs[5] | ((qh << 3) & 256U)]),
                                                   static_cast<int>(iq3s_grid[qs[6] | ((qh << 2) & 256U)]),
                                                   static_cast<int>(iq3s_grid[qs[7] | ((qh << 1) & 256U)]));
        sign = sign_bytes(load32(block + 74 + 4 * sub));
    } else {
        static_assert(F == QType::GGUF_IQ4_XS);
        const __m256i value = iq4_values(block + 8 + 16 * sub);
        magnitude           = _mm256_sign_epi8(value, value);
        sign                = value; // no IQ4_NL value is zero
    }
}

// An i-quant group is one 256-value block.
template <QType F>
inline void decode_iq(const std::uint8_t* block, Group& out) {
    iq_factors<F>(block, out);
    for (int sub = 0; sub < 8; ++sub) { iq_slice<F>(block, sub, out.magnitude[sub], out.sign[sub]); }
}

template <QType F>
void iq_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
             float* y, std::size_t y_stride) {
    constexpr int bytes   = gguf_block_shape(F).bytes;
    constexpr bool halves = F == QType::GGUF_IQ2_S || F == QType::GGUF_IQ1_M;
    constexpr bool delta  = F == QType::GGUF_IQ1_S; // unsigned codes
    const auto dots = [](const __m256i* u) {
        const Dots d = halves8(u);
        if constexpr (halves) {
            return d;
        } else {
            return _mm256_add_epi32(d.first, d.second);
        }
    };
    const auto decode = [](const std::uint8_t* row, int g, int, Group& group) {
        decode_iq<F>(row + std::size_t(g) * bytes, group);
    };
    const auto dot = [&x, &dots](int t, int g, int, const Group& group) {
        const std::int8_t* a = x.codes(t) + g * 256;
        __m256i u[8];
        for (int i = 0; i < 8; ++i) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32 * i));
            u[i] = dot_u8s8(group.magnitude[i], delta ? v : _mm256_sign_epi8(v, group.sign[i]));
        }
        return dots(u);
    };
    // One token: each slice's products as it is decoded, with no group kept for other tokens.
    [[maybe_unused]] const auto single = [&x, &dots](const std::uint8_t* row, int g, int,
                                                     Group& group) {
        const std::uint8_t* block = row + std::size_t(g) * bytes;
        iq_factors<F>(block, group);
        const std::int8_t* a = x.codes(0) + g * 256;
        __m256i u[8];
        for (int i = 0; i < 8; ++i) {
            __m256i magnitude, sign;
            iq_slice<F>(block, i, magnitude, sign);
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32 * i));
            u[i] = dot_u8s8(magnitude, delta ? v : _mm256_sign_epi8(v, sign));
        }
        return dots(u);
    };
    constexpr Form form = halves ? Form::Halves : delta ? Form::Delta : Form::Whole;
    if constexpr (F == QType::GGUF_IQ1_S || F == QType::GGUF_IQ1_M) {
        // IQ1's one-token rows decode each group whole before its products, as wider calls do:
        // on an EPYC 7K62 its slice-by-slice products were 3-4% slower.
        rows_in_lane_order<form>(w, x, row_begin, row_end, y, y_stride, decode, dot);
    } else {
        rows_in_lane_order<form>(w, x, row_begin, row_end, y, y_stride, decode, dot, single);
    }
}

// Q2_0: a 64-value block is two slices, so a group is four blocks. Its 16 code bytes hold values
// 4 i + L at bits 2 L of byte i; the activations' lane order stores value 4 i + L at position
// 16 L + i, so the codes of L = 0, 1 (and 2, 3) pair with the block's first (second) 32 positions.
// Of a block's int32 lanes, 0, 1, 4 and 5 then belong to its first slice and 2, 3, 6 and 7 to its
// second, which hadd pairs up as halves8 does. Codes c stand for c - 1: each slice's sum of
// activation codes comes off its dot.
void q2_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
             float* y, std::size_t y_stride) {
    const auto decode = [](const std::uint8_t* row, int g, int live, Group& group) {
        const __m128i three = _mm_set1_epi8(3);
        group.factor[0]     = halves_at(row + std::size_t(4 * g) * 18, 18, 1, live);
        for (int b = 0; 2 * b < live; ++b) {
            const __m128i q =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + std::size_t(4 * g + b) * 18 + 2));
            group.magnitude[2 * b] =
                _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 2), three), _mm_and_si128(q, three));
            group.magnitude[2 * b + 1] = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 6), three),
                                                          _mm_and_si128(_mm_srli_epi16(q, 4), three));
        }
    };
    const auto dot = [&x](int t, int g, int live, const Group& group) {
        __m256i v[4];
        for (int b = 0; b < 4; ++b) {
            if (2 * b >= live) {
                v[b] = _mm256_setzero_si256();
                continue;
            }
            const std::int8_t* a = x.codes(t) + (4 * g + b) * 64;
            v[b] = _mm256_add_epi32(
                dot_u8s8(group.magnitude[2 * b], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a))),
                dot_u8s8(group.magnitude[2 * b + 1],
                         _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32))));
        }
        const __m256i h01 = _mm256_hadd_epi32(v[0], v[1]), h23 = _mm256_hadd_epi32(v[2], v[3]);
        const __m256i dots = _mm256_add_epi32(_mm256_permute2x128_si256(h01, h23, 0x20),
                                              _mm256_permute2x128_si256(h01, h23, 0x31));
        return _mm256_sub_epi32(dots, live_ints(x.sums(t) + 8 * g, live));
    };
    rows_in_lane_order<Form::Whole>(w, x, row_begin, row_end, y, y_stride, decode, dot);
}

// The signed codes of a 32-value block: Q8_0's bytes, IQ4_NL's values, Q4_0's nibbles less 8 or
// Q5_0's five bits less 16 (decode_slice's).
template <QType F>
inline __m256i block32_codes(const std::uint8_t* block) {
    if constexpr (F == QType::GGUF_IQ4_NL) {
        return iq4_values(block + 2);
    } else if constexpr (F == QType::GGUF_Q8_0) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + 2));
    } else {
        const std::uint8_t* q = block + (F == QType::GGUF_Q5_0 ? 6 : 2);
        const __m128i bytes   = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q));
        const __m128i nib     = _mm_set1_epi8(15);
        __m256i codes = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(bytes, 4), nib), _mm_and_si128(bytes, nib));
        if constexpr (F == QType::GGUF_Q5_0) {
            codes = _mm256_or_si256(codes, _mm256_and_si256(bit_bytes(load32(block + 2)), _mm256_set1_epi8(16)));
            return _mm256_sub_epi8(codes, _mm256_set1_epi8(16));
        } else {
            return _mm256_sub_epi8(codes, _mm256_set1_epi8(8));
        }
    }
}

// Q8_0, IQ4_NL, Q4_0 and Q5_0: a slice is a block.
template <QType F>
void block32_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
                  float* y, std::size_t y_stride) {
    constexpr int bytes = gguf_block_shape(F).bytes;
    const auto decode   = [](const std::uint8_t* row, int g, int live, Group& group) {
        group.factor[0] = halves_at(row + std::size_t(8 * g) * bytes, bytes, 0, live);
        for (int i = 0; i < 8; ++i) {
            if (i >= live) {
                group.magnitude[i] = group.sign[i] = _mm256_setzero_si256();
                continue;
            }
            const __m256i codes = block32_codes<F>(row + std::size_t(8 * g + i) * bytes);
            group.magnitude[i]  = _mm256_sign_epi8(codes, codes);
            group.sign[i]       = codes;
        }
    };
    const auto dot = [&x](int t, int g, int live, const Group& group) {
        const std::int8_t* a = x.codes(t) + g * 256;
        __m256i u[8];
        for (int i = 0; i < 8; ++i) {
            // A slice past the row's end has no codes: its sign bytes zero the activation.
            const __m256i v = i < live ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32 * i))
                                       : _mm256_setzero_si256();
            u[i] = dot_u8s8(group.magnitude[i], _mm256_sign_epi8(v, group.sign[i]));
        }
        const Dots d = halves8(u);
        return _mm256_add_epi32(d.first, d.second);
    };
    rows_in_lane_order<Form::Whole>(w, x, row_begin, row_end, y, y_stride, decode, dot);
}

// Q4_K, Q5_K and Q6_K: a group is one 256-value block, decoded as decode_slice does. The K-quants
// with a minimum store unsigned codes (no signs); Q6_K's are signed, with a factor per half.
template <QType F>
void k_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
            float* y, std::size_t y_stride) {
    constexpr int bytes = gguf_block_shape(F).bytes;
    constexpr bool six  = F == QType::GGUF_Q6_K;
    const auto decode   = [](const std::uint8_t* row, int g, int, Group& group) {
        const std::uint8_t* block = row + std::size_t(g) * bytes;
        if constexpr (six) {
            // Slice s's two signed scales are bytes 2 s and 2 s + 1 of the 16 at 192.
            const __m256 d       = _mm256_set1_ps(half_at(block + 208));
            const __m128i scales = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 192));
            group.factor[0]      = _mm256_mul_ps(
                d, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_srai_epi16(_mm_slli_epi16(scales, 8), 8))));
            group.factor[1] =
                _mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_srai_epi16(scales, 8))));
            for (int sub = 0; sub < 8; ++sub) {
                const int h = sub / 4, q = sub % 4;
                const __m256i low   = bits_of(block + 64 * h + 32 * (q & 1), 4 * (q >> 1), 15);
                const __m256i high  = bits_of(block + 128 + 32 * h, 2 * q, 3);
                const __m256i codes = _mm256_sub_epi8(
                    _mm256_or_si256(low, _mm256_slli_epi16(high, 4)), _mm256_set1_epi8(32));
                group.magnitude[sub] = _mm256_sign_epi8(codes, codes);
                group.sign[sub]      = codes;
            }
        } else {
            constexpr bool five = F == QType::GGUF_Q5_K;
            // The twelve bytes at 4 as eight 6-bit scales and eight 6-bit minimums (ggml's utmp):
            // slices 0..3 keep the low six bits of bytes 0..3 (scales) and 4..7 (minimums); slices
            // 4..7 take the low (scales) or high (minimums) nibble of bytes 8..11 and the top two
            // bits of bytes 0..3 (scales) or 4..7 (minimums).
            const std::uint32_t u0 = load32(block + 4), u1 = load32(block + 8), u2 = load32(block + 12);
            const std::uint32_t low6 = 0x3F3F3F3FU, nib = 0x0F0F0F0FU, top2 = 0x03030303U;
            const std::uint32_t sc_lo = u0 & low6, sc_hi = (u2 & nib) | (((u0 >> 6) & top2) << 4);
            const std::uint32_t m_lo = u1 & low6, m_hi = ((u2 >> 4) & nib) | (((u1 >> 6) & top2) << 4);
            const auto widen = [](std::uint32_t lo, std::uint32_t hi) {
                return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                    _mm_set_epi32(0, 0, static_cast<int>(hi), static_cast<int>(lo))));
            };
            group.factor[0] = _mm256_mul_ps(_mm256_set1_ps(half_at(block)), widen(sc_lo, sc_hi));
            group.factor[1] = _mm256_mul_ps(_mm256_set1_ps(half_at(block + 2)), widen(m_lo, m_hi));
            for (int sub = 0; sub < 8; ++sub) {
                __m256i codes = bits_of(block + (five ? 48 : 16) + 32 * (sub / 2), 4 * (sub & 1), 15);
                if constexpr (five) {
                    codes = _mm256_or_si256(codes, _mm256_slli_epi16(bits_of(block + 16, sub, 1), 4));
                }
                group.magnitude[sub] = codes;
            }
        }
    };
    const auto dot = [&x](int t, int g, int, const Group& group) {
        const std::int8_t* a = x.codes(t) + g * 256;
        __m256i u[8];
        for (int i = 0; i < 8; ++i) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32 * i));
            u[i] = dot_u8s8(group.magnitude[i], six ? _mm256_sign_epi8(v, group.sign[i]) : v);
        }
        const Dots d = halves8(u);
        if constexpr (six) {
            return d;
        } else {
            return _mm256_add_epi32(d.first, d.second);
        }
    };
    rows_in_lane_order<six ? Form::SixK : Form::Minimum>(w, x, row_begin, row_end, y, y_stride,
                                                         decode, dot);
}

// Q3_K: a group is one 256-value block, decoded as decode_slice does. Slice u = 4 h + j takes
// bits 2 j of the 32 bytes at 32 + 32 h and bit 4 h + j of the 32 bytes at 0, value
// low + 4 high - 4, and its two halves' 6-bit scales less 32 times d.
void q3k_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
              float* y, std::size_t y_stride) {
    const auto decode = [](const std::uint8_t* row, int g, int, Group& group) {
        const std::uint8_t* block = row + std::size_t(g) * 110;
        const float d             = half_at(block + 108);
        const auto scale          = [block](int k) {
            return int(((block[96 + k % 8] >> (4 * (k / 8))) & 15U) |
                       (((block[104 + k % 4] >> (2 * (k / 4))) & 3U) << 4)) - 32;
        };
        alignas(32) float f0[8], f1[8];
        for (int sub = 0; sub < 8; ++sub) {
            const int h = sub / 4, j = sub % 4;
            f0[sub]             = d * float(scale(8 * h + 2 * j));
            f1[sub]             = d * float(scale(8 * h + 2 * j + 1));
            const __m256i low   = bits_of(block + 32 + 32 * h, 2 * j, 3);
            const __m256i high  = bits_of(block, 4 * h + j, 1);
            const __m256i codes = _mm256_sub_epi8(_mm256_or_si256(low, _mm256_slli_epi16(high, 2)),
                                                  _mm256_set1_epi8(4));
            group.magnitude[sub] = _mm256_sign_epi8(codes, codes);
            group.sign[sub]      = codes;
        }
        group.factor[0] = _mm256_load_ps(f0);
        group.factor[1] = _mm256_load_ps(f1);
    };
    const auto dot = [&x](int t, int g, int, const Group& group) {
        const std::int8_t* a = x.codes(t) + g * 256;
        __m256i u[8];
        for (int i = 0; i < 8; ++i) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32 * i));
            u[i]            = dot_u8s8(group.magnitude[i], _mm256_sign_epi8(v, group.sign[i]));
        }
        return halves8(u);
    };
    rows_in_lane_order<Form::SixK>(w, x, row_begin, row_end, y, y_stride, decode, dot);
}

// NVFP4: a group is four 64-value blocks, two slices each, decoded as decode_slice does: ggml's
// doubled E2M1 values of the low nibbles of a slice's 16 code bytes (values 0..7 and 16..23) and of
// their high nibbles (8..15 and 24..31), and the halved UE4M3 scales of its two 16-value halves.
void nvfp4_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
                float* y, std::size_t y_stride) {
    const auto decode = [](const std::uint8_t* row, int g, int live, Group& group) {
        const __m128i table = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_mxfp4));
        const __m128i nib   = _mm_set1_epi8(15);
        alignas(32) float f0[8], f1[8];
        for (int i = 0; i < 8; ++i) {
            if (i >= live) {
                f0[i] = f1[i]      = 0.0F;
                group.magnitude[i] = group.sign[i] = _mm256_setzero_si256();
                continue;
            }
            const std::uint8_t* block = row + std::size_t(4 * g + i / 2) * 36;
            const int half            = i % 2;
            f0[i]                     = ue4m3_half(block[2 * half]);
            f1[i]                     = ue4m3_half(block[2 * half + 1]);
            const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 4 + 16 * half));
            const __m128i lo    = _mm_shuffle_epi8(table, _mm_and_si128(bytes, nib));
            const __m128i hi    = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(bytes, 4), nib));
            const __m256i codes = _mm256_set_m128i(_mm_unpackhi_epi64(lo, hi), _mm_unpacklo_epi64(lo, hi));
            group.magnitude[i]  = _mm256_sign_epi8(codes, codes);
            group.sign[i]       = codes;
        }
        group.factor[0] = _mm256_load_ps(f0);
        group.factor[1] = _mm256_load_ps(f1);
    };
    const auto dot = [&x](int t, int g, int live, const Group& group) {
        const std::int8_t* a = x.codes(t) + g * 256;
        __m256i u[8];
        for (int i = 0; i < 8; ++i) {
            // A slice past the row's end reads no activation.
            const __m256i v = i < live ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32 * i))
                                       : _mm256_setzero_si256();
            u[i] = dot_u8s8(group.magnitude[i], _mm256_sign_epi8(v, group.sign[i]));
        }
        return halves8(u);
    };
    rows_in_lane_order<Form::SixK>(w, x, row_begin, row_end, y, y_stride, decode, dot);
}

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
    if (x.tokens() > kMaxTokens) { return rows_scalar(w, x, row_begin, row_end, y, y_stride); }
    switch (w.format) {
    case QType::GGUF_IQ1_S: return iq_rows<QType::GGUF_IQ1_S>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_IQ1_M: return iq_rows<QType::GGUF_IQ1_M>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_IQ2_XXS:
        return iq_rows<QType::GGUF_IQ2_XXS>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_IQ2_S: return iq_rows<QType::GGUF_IQ2_S>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_IQ3_XXS:
        return iq_rows<QType::GGUF_IQ3_XXS>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_IQ3_S: return iq_rows<QType::GGUF_IQ3_S>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_IQ4_XS:
        return iq_rows<QType::GGUF_IQ4_XS>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_Q2_0: return q2_rows(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_IQ4_NL:
        return block32_rows<QType::GGUF_IQ4_NL>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_Q8_0: return block32_rows<QType::GGUF_Q8_0>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_Q4_0: return block32_rows<QType::GGUF_Q4_0>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_Q5_0: return block32_rows<QType::GGUF_Q5_0>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_Q4_K: return k_rows<QType::GGUF_Q4_K>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_Q5_K: return k_rows<QType::GGUF_Q5_K>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_Q6_K: return k_rows<QType::GGUF_Q6_K>(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_Q3_K: return q3k_rows(w, x, row_begin, row_end, y, y_stride);
    case QType::GGUF_NVFP4: return nvfp4_rows(w, x, row_begin, row_end, y, y_stride);
    default: return rows_scalar(w, x, row_begin, row_end, y, y_stride);
    }
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
