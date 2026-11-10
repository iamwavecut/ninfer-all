#pragma once

// The DSpark drafter's experts of each block, most routed first, for --draft-experts. Taken from
// Basalt's data/profiles/deepseek41-40x384-dspark.prof (https://github.com/jesdga95/basalt, MIT,
// copyright 2026 Jesús Gulfo), counted on live runs of DeepSeek-V4.1-Flash-1017. The drafter
// artifacts number the experts as the release does.

#include <array>
#include <cstdint>

namespace ninfer::models::deepseek_v41 {

// The experts a block keeps by default with the STQ drafter and host or disk experts: of 64, 32 and
// all 128 kept, 64 decoded fastest with host experts on an RTX 4090, whose expert cache takes the
// memory the drafter gives back. With device experts nothing takes it, and 64 lost acceptance on an
// RTX PRO 6000 (docs/deepseek-v41.md#the-drafters-experts), so those keep all 128.
inline constexpr std::uint32_t kDefaultDraftExperts = 64;

inline constexpr std::array<std::array<std::uint8_t, 128>, 3> kDSparkExpertRanking{{
    {{
         97,   8,  19,  33,  85,  43,  36, 117,  68,  47,  67,  93,  92,  17,  72,   5,
         95, 123,   3,  61,   2, 101, 121, 110,  62,  27,  14,  65,  71,  24,  73,  70,
         23,  63,  64, 115,  59,  34,  96, 109,  11, 124,  15,  79,  58,  26,  39, 113,
         66,  51,  99,  54,  77,  57,  13,  49, 100,  74,  35,  21,  88,  60,   4, 126,
         20,  89, 106,  46,  76,  98, 108,   6,  31,  87, 118,  75,  48, 112, 119, 111,
         50,  83,  90,  22, 105,  38,  42, 102,  52,  30,  78,  80,  82, 114, 125, 120,
         53,  28,   0,  56,  55, 107,  16, 122,  86,  69,   7,  45,   9,  29,  41, 104,
         32,  37,   1,  10,  12,  18,  25,  40,  44,  81,  84,  91,  94, 103, 116, 127,
    }},
    {{
        121,  98, 109,  61, 113, 126,  58,  78,  24,  56, 114,  82,  26,  79,  63,  62,
         29, 100,  25,  52,  10,  66,  76,  49,  89, 122, 115,  90,  32,  19,  28,  11,
         45,  97,  27,  36,   1,  68,  22, 111,  53, 123,  21,  39,  30,  77, 101,  41,
         37, 107,  64, 110,  18,  31,  47,  80, 118, 102, 124,  23,  43, 125, 116,  42,
         55, 112,  60,  96, 120,  75,  12,  35,  20, 108,  51,   6,   7, 127,  57,  74,
         70,   4,  44,   5, 103,  38,  46,  71,  83,   3,  50,  95,   0,   2,   8,   9,
         13,  14,  15,  16,  17,  33,  34,  40,  48,  54,  59,  65,  67,  69,  72,  73,
         81,  84,  85,  86,  87,  88,  91,  92,  93,  94,  99, 104, 105, 106, 117, 119,
    }},
    {{
         19, 101,  58,  98,  97,  81,  17, 103, 127,  78,   6,  80,   9, 122,  16,  53,
         12,  39, 120, 102, 111,  27,  83,  82,  96,  84,  88,  14,  49,  15,  93,  51,
         21,  20,  13, 114,  67,  87, 112,  77,  38, 118,  79,  91,  47, 119, 110,  48,
         94,  52,  69,  59,   1,  24,  54,  11,  35,  40,  46,  66,   0,   2,   3,   4,
          5,   7,   8,  10,  18,  22,  23,  25,  26,  28,  29,  30,  31,  32,  33,  34,
         36,  37,  41,  42,  43,  44,  45,  50,  55,  56,  57,  60,  61,  62,  63,  64,
         65,  68,  70,  71,  72,  73,  74,  75,  76,  85,  86,  89,  90,  92,  95,  99,
        100, 104, 105, 106, 107, 108, 109, 113, 115, 116, 117, 121, 123, 124, 125, 126,
    }},
}};

} // namespace ninfer::models::deepseek_v41
