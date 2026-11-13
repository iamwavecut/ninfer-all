// bf16p KV cache (--kv-dtype bf16p). The append's rows against an independent host decoder
// (kv_bf16p_host.h): every packable row decodes to its input bits, a row with more than six
// escapes is the poison row, rows the call did not write keep their bytes. Attention over a bf16p
// cache against the same attention over a bf16 cache: the same output bits on every route (small-T,
// chunked small-T, prompt; batched, masked and cached forms).

#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/softmax_attention.h"
#include "core/arena.h"
#include "core/device_buffer.h"
#include "core/device_launch.h"
#include "core/paged_kv_view.h"
#include "core/paged_kv_storage.h"
#include "ops/kv_bf16p_host.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kDim  = 256;
constexpr int kPage = static_cast<int>(kPagedKVPageSize);
constexpr float kScale = 0.0625F;

std::uint16_t bf16_bits(float value) { return f32_to_bf16(value); }

std::uint16_t with_exponent(std::mt19937& g, unsigned exponent) {
    const unsigned sign = g() & 1U;
    const unsigned frac = g() & 0x7FU;
    return static_cast<std::uint16_t>(sign << 15 | (exponent & 0xFFU) << 7 | frac);
}

// One K or V row of kind 0..7; `poison` says whether it has no packed form.
std::vector<std::uint16_t> row_of(int kind, std::mt19937& g, bool& poison) {
    std::vector<std::uint16_t> row(kDim);
    std::uniform_real_distribution<float> wide(-6.0F, 6.0F), small(-1.0e-3F, 1.0e-3F),
        magnitude(0.5F, 6.0F);
    poison = false;
    switch (kind) {
    case 0: // ordinary values
        for (auto& x : row) x = bf16_bits(wide(g));
        break;
    case 1: // small values, zeros of both signs, subnormals
        for (int d = 0; d < kDim; ++d) {
            row[d] = d % 5 == 0   ? 0x0000
                     : d % 7 == 0 ? 0x8000
                     : d % 11 == 0
                         ? static_cast<std::uint16_t>((g() & 1U) << 15 | (1U + g() % 127U))
                         : bf16_bits(small(g));
        }
        break;
    case 2:
    case 3: { // six escapes, then a seventh: the poison row
        for (auto& x : row) x = bf16_bits((g() & 1U) ? magnitude(g) : -magnitude(g));
        const int at[]                = {3, 50, 99, 128, 200, 255, 17};
        const std::uint16_t escapes[] = {0x7F80, 0xFF80, 0x7FC1, 0x7F7F, 0x0080, 0x6400, 0x0100};
        for (int i = 0; i < (kind == 2 ? 6 : 7); ++i) row[at[i]] = escapes[i];
        poison = kind == 3;
        break;
    }
    case 4: // zeros of both signs
        for (int d = 0; d < kDim; ++d) row[d] = d % 2 == 0 ? 0x0000 : 0x8000;
        break;
    case 5: // exponent 0 throughout
        for (auto& x : row) x = with_exponent(g, 0);
        break;
    case 6: // exactly the fourteen binades of the window
        for (int d = 0; d < kDim; ++d) row[d] = with_exponent(g, 100U + static_cast<unsigned>(d % 14));
        break;
    default: // fifteen binades: the three values of the highest escape
        for (int d = 0; d < kDim; ++d) {
            row[d] = with_exponent(g, d < 253 ? 100U + static_cast<unsigned>(d % 14) : 114U);
        }
        break;
    }
    return row;
}

struct Planes {
    GuardedDeviceBuffer k, v, k_side, v_side;
    Planes(int heads, int pages)
        : k(static_cast<std::size_t>(bf16p::kRowBytes) * kPage * heads * pages),
          v(k.bytes()), k_side(static_cast<std::size_t>(bf16p::kSideBytes) * kPage * heads * pages),
          v_side(k_side.bytes()) {}
};

// Appends T rows at P.. into a bf16p cache of shuffled pages and checks every row of the planes.
int append_case(int kv_heads, int first_position, int tokens, std::uint32_t seed) {
    const int logical  = (first_position + tokens + kPage - 1) / kPage;
    const int physical = logical + 2;
    std::vector<std::int32_t> table(static_cast<std::size_t>(logical));
    std::iota(table.begin(), table.end(), 2);
    std::mt19937 g(seed);
    std::shuffle(table.begin(), table.end(), g);

    const std::size_t elements = static_cast<std::size_t>(kDim) * kv_heads * tokens;
    std::vector<std::uint16_t> k(elements), v(elements);
    std::vector<char> k_poison(static_cast<std::size_t>(kv_heads) * tokens),
        v_poison(k_poison.size());
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < kv_heads; ++h) {
            const std::size_t r = static_cast<std::size_t>(t) * kv_heads + h;
            bool poison         = false;
            const auto kr       = row_of(static_cast<int>((r + seed) % 8), g, poison);
            k_poison[r]         = poison;
            const auto vr       = row_of(static_cast<int>((r * 3 + seed + 1) % 8), g, poison);
            v_poison[r]         = poison;
            std::copy(kr.begin(), kr.end(), k.begin() + static_cast<std::ptrdiff_t>(r * kDim));
            std::copy(vr.begin(), vr.end(), v.begin() + static_cast<std::ptrdiff_t>(r * kDim));
        }
    }
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    std::iota(positions.begin(), positions.end(), first_position);

    Planes planes(kv_heads, physical);
    for (auto* plane : {&planes.k, &planes.v, &planes.k_side, &planes.v_side}) plane->fill(0x5A);
    DeviceBuffer dk = to_device(k), dv = to_device(v), dp = to_device(positions),
                 dt = to_device(table);
    const Tensor tk(dk.p, DType::BF16, {kDim, kv_heads, tokens});
    const Tensor tv(dv.p, DType::BF16, {kDim, kv_heads, tokens});
    const Tensor tp(dp.p, DType::I32, {tokens});
    PagedKVLayerView cache{
        .k_pages       = Tensor(planes.k.data(), DType::U8, {bf16p::kRowBytes, kPage, kv_heads, physical}),
        .v_pages       = Tensor(planes.v.data(), DType::U8, {bf16p::kRowBytes, kPage, kv_heads, physical}),
        .k_scale_pages = Tensor(planes.k_side.data(), DType::U8, {bf16p::kSideBytes, kPage, kv_heads, physical}),
        .v_scale_pages = Tensor(planes.v_side.data(), DType::U8, {bf16p::kSideBytes, kPage, kv_heads, physical}),
        .block_table   = Tensor(dt.p, DType::I32, {logical}),
        .head_dim      = kDim,
        .num_kv_heads  = kv_heads,
        .storage       = KvCacheStorage::Bf16Packed,
    };
    ops::kv_cache_append(tk, tv, tp, cache, nullptr);
    cuda_synchronize();

    const auto rows_k = from_device<std::uint8_t>(planes.k.data(), planes.k.bytes());
    const auto rows_v = from_device<std::uint8_t>(planes.v.data(), planes.v.bytes());
    const auto side_k = from_device<std::uint8_t>(planes.k_side.data(), planes.k_side.bytes());
    const auto side_v = from_device<std::uint8_t>(planes.v_side.data(), planes.v_side.bytes());
    const std::string label = "bf16p append Hkv=" + std::to_string(kv_heads) +
                              " P=" + std::to_string(first_position) + " T=" + std::to_string(tokens);
    int failures            = 0;
    std::vector<char> written(static_cast<std::size_t>(physical) * kv_heads * kPage, 0);
    const auto check = [&](const std::vector<std::uint8_t>& rows,
                           const std::vector<std::uint8_t>& sides,
                           const std::vector<std::uint16_t>& input, const std::vector<char>& poison,
                           const char* plane) {
        for (int t = 0; t < tokens; ++t) {
            const int p = first_position + t;
            for (int h = 0; h < kv_heads; ++h) {
                const std::size_t slot =
                    (static_cast<std::size_t>(table[static_cast<std::size_t>(p / kPage)]) * kv_heads + h) *
                        kPage + static_cast<std::size_t>(p % kPage);
                written[slot]             = 1;
                const std::uint8_t* row   = rows.data() + slot * bf16p::kRowBytes;
                const std::uint8_t* side  = sides.data() + slot * bf16p::kSideBytes;
                const std::size_t r       = static_cast<std::size_t>(t) * kv_heads + h;
                if (poison[r] != 0) {
                    bool bytes = side[0] == 240 && side[1] == 0;
                    for (int i = 0; i < kDim; ++i) bytes = bytes && row[i] == 0x7F;
                    for (int i = kDim; i < bf16p::kRowBytes; ++i) bytes = bytes && row[i] == 0xFF;
                    for (int d = 0; d < kDim; ++d) bytes = bytes && bf16p::decode(row, side, d) == 0x7FFF;
                    if (!bytes && failures++ < 8) {
                        std::cerr << label << ' ' << plane << " t=" << t << " h=" << h
                                  << ": not the poison row\n";
                    }
                    continue;
                }
                for (int d = 0; d < kDim; ++d) {
                    const std::uint16_t want = input[r * kDim + static_cast<std::size_t>(d)];
                    const std::uint16_t got  = bf16p::decode(row, side, d);
                    if (got != want) {
                        if (failures++ < 8) {
                            std::cerr << label << ' ' << plane << " t=" << t << " h=" << h
                                      << " d=" << d << ": decoded 0x" << std::hex << got
                                      << " want 0x" << want << std::dec << '\n';
                        }
                        break;
                    }
                }
            }
        }
    };
    check(rows_k, side_k, k, k_poison, "K");
    check(rows_v, side_v, v, v_poison, "V");
    for (std::size_t slot = 0; slot < written.size(); ++slot) {
        if (written[slot] != 0) continue;
        bool kept = true;
        for (const auto* plane : {&rows_k, &rows_v}) {
            for (int i = 0; i < bf16p::kRowBytes; ++i) {
                kept = kept && (*plane)[slot * bf16p::kRowBytes + static_cast<std::size_t>(i)] == 0x5A;
            }
        }
        for (const auto* plane : {&side_k, &side_v}) {
            for (int i = 0; i < bf16p::kSideBytes; ++i) {
                kept = kept && (*plane)[slot * bf16p::kSideBytes + static_cast<std::size_t>(i)] == 0x5A;
            }
        }
        if (!kept && failures++ < 8) std::cerr << label << ": unwritten slot " << slot << " changed\n";
    }
    for (auto* plane : {&planes.k, &planes.v, &planes.k_side, &planes.v_side}) {
        failures += plane->verify_guards(label.c_str());
    }
    return failures;
}

// A bf16 and a bf16p cache over the same pages and the same history.
struct CachePair {
    int heads, logical, rows, physical;
    GuardedDeviceBuffer bf16_k, bf16_v;
    Planes packed;
    DeviceBuffer tables;

    CachePair(int heads_, int logical_, int rows_, const std::vector<std::int32_t>& table)
        : heads(heads_), logical(logical_), rows(rows_), physical(logical_ * rows_ + 1),
          bf16_k(static_cast<std::size_t>(kDim) * 2 * kPage * heads_ * physical),
          bf16_v(bf16_k.bytes()), packed(heads_, physical), tables(to_device(table)) {}

    PagedKVBatchLayerView view(bool bf16p) {
        PagedKVBatchLayerView out{};
        if (bf16p) {
            out.k_pages = Tensor(packed.k.data(), DType::U8, {bf16p::kRowBytes, kPage, heads, physical});
            out.v_pages = Tensor(packed.v.data(), DType::U8, {bf16p::kRowBytes, kPage, heads, physical});
            out.k_scale_pages =
                Tensor(packed.k_side.data(), DType::U8, {bf16p::kSideBytes, kPage, heads, physical});
            out.v_scale_pages =
                Tensor(packed.v_side.data(), DType::U8, {bf16p::kSideBytes, kPage, heads, physical});
        } else {
            out.k_pages = Tensor(bf16_k.data(), DType::BF16, {kDim, kPage, heads, physical});
            out.v_pages = Tensor(bf16_v.data(), DType::BF16, {kDim, kPage, heads, physical});
        }
        out.block_tables = Tensor(tables.p, DType::I32, {logical, rows});
        out.head_dim     = kDim;
        out.num_kv_heads = heads;
        out.storage      = bf16p ? KvCacheStorage::Bf16Packed : KvCacheStorage::BFloat16;
        return out;
    }

    PagedKVLayerView row_view(bool bf16p, int row) {
        const PagedKVBatchLayerView all = view(bf16p);
        return PagedKVLayerView{
            .k_pages       = all.k_pages,
            .v_pages       = all.v_pages,
            .k_scale_pages = all.k_scale_pages,
            .v_scale_pages = all.v_scale_pages,
            .block_table   = Tensor(static_cast<std::int32_t*>(tables.p) + row * logical, DType::I32, {logical}),
            .head_dim      = kDim,
            .num_kv_heads  = heads,
            .storage       = all.storage,
        };
    }
};

std::vector<std::uint16_t> random_rows(std::size_t rows, std::mt19937& g, float bound,
                                       bool escapes) {
    std::uniform_real_distribution<float> u(-bound, bound);
    std::vector<std::uint16_t> out(rows * kDim);
    for (std::size_t r = 0; r < rows; ++r) {
        for (int d = 0; d < kDim; ++d) out[r * kDim + static_cast<std::size_t>(d)] = bf16_bits(u(g));
        // Finite escapes: four values 2^-70, far below the row's window.
        if (escapes && r % 29 == 3) {
            for (int d : {1, 64, 130, 254}) out[r * kDim + static_cast<std::size_t>(d)] = 0x1C80;
        }
    }
    return out;
}

int compare(const std::string& label, const std::vector<std::uint16_t>& want,
            const std::vector<std::uint16_t>& got) {
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (want[i] != got[i]) {
            std::cerr << label << ": output " << i << " bf16 0x" << std::hex << want[i]
                      << " bf16p 0x" << got[i] << std::dec << '\n';
            return 1;
        }
    }
    return 0;
}

// B rows of `history` keys each, then a call of W columns (row r keeps valid[r]); the bf16p
// cache's output bits against the bf16 cache's, and for B = 1 the cached form's after it.
int attention_case(int q_heads, int kv_heads, int history, int width, std::vector<int> valid,
                   std::uint32_t seed, ops::CausalAttentionExecutionEnvelope hints = {}) {
    const int batch   = static_cast<int>(valid.size());
    const int keys    = history + width;
    const int logical = (keys + kPage - 1) / kPage;
    std::mt19937 g(seed);
    std::vector<std::int32_t> table(static_cast<std::size_t>(logical) * batch);
    std::iota(table.begin(), table.end(), 1);
    std::shuffle(table.begin(), table.end(), g);
    CachePair caches(kv_heads, logical, batch, table);
    for (auto* plane : {&caches.bf16_k, &caches.bf16_v, &caches.packed.k, &caches.packed.v,
                        &caches.packed.k_side, &caches.packed.v_side}) {
        plane->fill(0);
    }
    // The history, appended row by row into both caches.
    if (history > 0) {
        std::vector<std::int32_t> positions(static_cast<std::size_t>(history));
        std::iota(positions.begin(), positions.end(), 0);
        DeviceBuffer dp = to_device(positions);
        for (int row = 0; row < batch; ++row) {
            DeviceBuffer dk = to_device(random_rows(static_cast<std::size_t>(history) * kv_heads, g, 1.0F, true));
            DeviceBuffer dv = to_device(random_rows(static_cast<std::size_t>(history) * kv_heads, g, 2.0F, true));
            const Tensor tk(dk.p, DType::BF16, {kDim, kv_heads, history});
            const Tensor tv(dv.p, DType::BF16, {kDim, kv_heads, history});
            const Tensor tp(dp.p, DType::I32, {history});
            for (const bool packed : {false, true}) {
                const PagedKVLayerView view = caches.row_view(packed, row);
                ops::kv_cache_append(tk, tv, tp, view, nullptr);
            }
        }
    }
    const std::size_t columns = static_cast<std::size_t>(width) * batch;
    DeviceBuffer dq = to_device(random_rows(columns * q_heads, g, 1.0F, false));
    DeviceBuffer dk = to_device(random_rows(columns * kv_heads, g, 1.0F, true));
    DeviceBuffer dv = to_device(random_rows(columns * kv_heads, g, 2.0F, true));
    std::vector<std::int32_t> positions(columns), rows(static_cast<std::size_t>(batch));
    for (int b = 0; b < batch; ++b) {
        rows[static_cast<std::size_t>(b)] = b;
        for (int w = 0; w < width; ++w) positions[static_cast<std::size_t>(b * width + w)] = history + w;
    }
    DeviceBuffer dp = to_device(positions), drows = to_device(rows), dvalid = to_device(valid);
    const bool masked = std::any_of(valid.begin(), valid.end(), [&](int v) { return v != width; });
    const Tensor tq(dq.p, DType::BF16, {kDim, q_heads, width, batch});
    const Tensor tk(dk.p, DType::BF16, {kDim, kv_heads, width, batch});
    const Tensor tv(dv.p, DType::BF16, {kDim, kv_heads, width, batch});
    const Tensor tp(dp.p, DType::I32, {width, batch});
    const Tensor trows(drows.p, DType::I32, {batch});
    const Tensor tvalid = masked ? Tensor(dvalid.p, DType::I32, {batch}) : Tensor{};
    const ops::AttentionHeadGeometry geometry{kDim, q_heads, kv_heads};
    ops::CausalAttentionExecutionEnvelope envelope = hints;
    envelope.min_visible_keys = static_cast<std::uint32_t>(keys);
    envelope.max_visible_keys = static_cast<std::uint32_t>(keys);

    const std::string label = "bf16p attention h" + std::to_string(q_heads) + "kv" +
                              std::to_string(kv_heads) + " H=" + std::to_string(history) +
                              " W=" + std::to_string(width) + " B=" + std::to_string(batch) +
                              (masked ? " masked" : "") + " route=" +
                              std::to_string(ops::causal_softmax_attention_route_family(
                                  geometry, KvCacheStorage::Bf16Packed, envelope, batch, width));
    std::vector<std::uint16_t> outputs[2], cached[2];
    int append_failures = 0;
    for (const bool packed : {false, true}) {
        const KvCacheStorage storage = packed ? KvCacheStorage::Bf16Packed : KvCacheStorage::BFloat16;
        const std::size_t bytes      = std::max<std::size_t>(
            256, ops::causal_softmax_attention_workspace_capacity_bytes(geometry, storage, envelope,
                                                                        batch, width, width));
        GuardedDeviceBuffer workspace_memory(bytes);
        WorkspaceArena workspace(DeviceSpan{workspace_memory.data(), workspace_memory.bytes()});
        GuardedDeviceBuffer out(columns * q_heads * kDim * 2);
        out.fill(0x77);
        Tensor tout(out.data(), DType::BF16, {kDim, q_heads, width, batch});
        // The cache planes this storage writes, to compare the append alone with the attention.
        std::vector<GuardedDeviceBuffer*> planes{&caches.bf16_k, &caches.bf16_v};
        if (packed) {
            planes = {&caches.packed.k, &caches.packed.v, &caches.packed.k_side,
                      &caches.packed.v_side};
        }
        const auto snapshot = [&] {
            std::vector<std::vector<std::uint8_t>> bytes;
            for (GuardedDeviceBuffer* plane : planes) {
                bytes.push_back(from_device<std::uint8_t>(plane->data(), plane->bytes()));
            }
            return bytes;
        };
        const auto restore = [&](const std::vector<std::vector<std::uint8_t>>& bytes) {
            for (std::size_t i = 0; i < planes.size(); ++i) {
                planes[i]->copy_from_host(bytes[i].data(), bytes[i].size());
            }
        };
        const auto before = snapshot();
        ops::causal_softmax_attention(tq, tk, tv, tp, tvalid, trows, geometry, kScale,
                                      caches.view(packed), envelope, workspace, tout, nullptr);
        cuda_synchronize();
        outputs[packed] = from_device<std::uint16_t>(out.data(), columns * q_heads * kDim);
        if (ops::causal_softmax_attention_route_family(geometry, storage, envelope, batch, width) ==
            2) {
            // The append alone writes the bytes the attention call wrote.
            const auto attended = snapshot();
            restore(before);
            ops::causal_softmax_attention_append(tk, tv, tp, tvalid, trows, geometry,
                                                 caches.view(packed), envelope, nullptr);
            cuda_synchronize();
            if (snapshot() != attended) {
                std::cerr << "FAIL " << label << (packed ? " bf16p" : " bf16")
                          << ": the append alone wrote other cache bytes\n";
                ++append_failures;
            }
            restore(attended);
        }
        if (batch == 1) {
            out.fill(0x77);
            Tensor q1(dq.p, DType::BF16, {kDim, q_heads, width});
            Tensor p1(dp.p, DType::I32, {width});
            Tensor out1(out.data(), DType::BF16, {kDim, q_heads, width});
            ops::causal_softmax_attention_cached(q1, p1, geometry, kScale, caches.row_view(packed, 0),
                                                 envelope, workspace, out1, nullptr);
            cuda_synchronize();
            cached[packed] = from_device<std::uint16_t>(out.data(), columns * q_heads * kDim);
        }
    }
    int failures = append_failures + compare(label, outputs[0], outputs[1]);
    if (batch == 1) failures += compare(label + " cached", cached[0], cached[1]);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (const int kv_heads : {1, 2, 4}) {
        failures += append_case(kv_heads, 0, 100, 11U + static_cast<std::uint32_t>(kv_heads));
        failures += append_case(kv_heads, 37, 130, 21U + static_cast<std::uint32_t>(kv_heads));
        failures += append_case(kv_heads, 63, 1, 31U + static_cast<std::uint32_t>(kv_heads));
    }
    // Attention over a packed cache needs sm_75 or newer; a Volta card checks the appends only.
    if (ops::current_device_volta()) {
        std::cout << "SKIP attention: --kv-dtype bf16p needs sm_75 or newer\n";
        std::cout << (failures == 0 ? "PASS" : "FAIL") << " kv_bf16p failures=" << failures << '\n';
        return failures == 0 ? 0 : 1;
    }
    struct Shape {
        int history, width;
    };
    for (const auto [q_heads, kv_heads] : {std::pair{24, 4}, std::pair{16, 2}, std::pair{6, 1}}) {
        std::uint32_t seed = 100U * static_cast<std::uint32_t>(q_heads);
        for (const Shape shape : {Shape{0, 1}, Shape{40, 1}, Shape{300, 1}, Shape{2000, 1},
                                  Shape{130, 4}, Shape{100, 8}, Shape{300, 8}, Shape{100, 12},
                                  Shape{500, 12}, Shape{1000, 16}, Shape{0, 64}, Shape{200, 64},
                                  Shape{700, 200}, Shape{3000, 130}}) {
            failures += attention_case(q_heads, kv_heads, shape.history, shape.width,
                                       {shape.width}, ++seed);
        }
        ops::CausalAttentionExecutionEnvelope small_prefill{};
        small_prefill.small_prefill = true;
        failures += attention_case(q_heads, kv_heads, 4000, 40, {40}, ++seed, small_prefill);
        ops::CausalAttentionExecutionEnvelope wide{};
        wide.wide_verification = true;
        failures += attention_case(q_heads, kv_heads, 1000, 32, {32}, ++seed, wide);
        failures += attention_case(q_heads, kv_heads, 300, 4, {4, 3, 4}, ++seed);
        failures += attention_case(q_heads, kv_heads, 300, 9, {9, 9, 5}, ++seed);
        failures += attention_case(q_heads, kv_heads, 1000, 1, {1, 1}, ++seed);
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " kv_bf16p failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
