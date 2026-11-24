// Inkling 2's --loop-kv modes on a real artifact (skipped without one):
//   NINFER_INKLING_ARTIFACT  an Inkling 2 .ninfer (its text component is enough)
//   NINFER_INKLING_EXPERTS   host or disk places the routed experts (device by default)
// LastFull and LastAll keep only the core's last pass's caches of the layers they share: a call's
// earlier passes read the last pass's rows for the positions before the call and their own for
// the call's positions. The reference is the per-pass executor: before each call its state goes
// out as an image, the earlier passes' caches of the shared layers in it are replaced with the
// last pass's, and it comes back, so the call reads what the shared mode reads, through the same
// kernels; the logits must be the same bits. The calls, for every KV format: a first prompt chunk
// (nothing lies before it, so every mode gives the per-pass logits), a second chunk far enough in
// for the far field, a decode step, a verification of four rows kept up to its second, and a
// decode step after it. Then the shared mode's images: they hold four of the eight full-attention
// visits' rows a position (LastAll also 20 of the 40 rings), and a sequence loaded from one decodes
// as the sequence it came from.

#include "artifact/reader.h"
#include "core/device.h"
#include "models/inkling2/executor.h"
#include "models/inkling2/kv_format.h"
#include "models/inkling2/model.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
namespace ink = ninfer::models::inkling2;

int failures = 0;
// Host or disk experts: each executor's device cache, small enough for three executors at once.
std::optional<std::uint64_t> expert_cache;

void expect(bool ok, const std::string& what) {
    std::cout << (ok ? "ok   " : "FAIL ") << what << '\n';
    if (!ok) { ++failures; }
}

// The logits of the last call's `rows`, as stored (BF16 words).
std::vector<std::uint16_t> logits(ink::Executor& executor, std::uint32_t rows) {
    const Tensor t = executor.logits(rows);
    std::vector<std::uint16_t> out(std::size_t(t.ne[0]) * rows);
    CUDA_CHECK(cudaStreamSynchronize(executor.head_stream()));
    CUDA_CHECK(cudaMemcpy(out.data(), t.data, out.size() * 2, cudaMemcpyDeviceToHost));
    return out;
}

// Deterministic ordinary token ids.
std::vector<std::int32_t> tokens(std::size_t n, std::uint32_t seed) {
    std::vector<std::int32_t> out(n);
    std::uint32_t state = seed;
    for (auto& token : out) {
        state = state * 1664525u + 1013904223u;
        token = std::int32_t(1000 + (state >> 8) % 60000);
    }
    return out;
}

const char* mode_name(LoopKvCache mode) {
    switch (mode) {
    case LoopKvCache::PerPass: return "per-pass";
    case LoopKvCache::LastFull: return "last-full";
    case LoopKvCache::LastAll: return "last-all";
    }
    return "?";
}

// Whether visit (layer, pass) shares the last pass's caches under `mode`.
bool shared(const ink::TextConfig& c, LoopKvCache mode, std::uint32_t layer, std::uint32_t pass) {
    return mode != LoopKvCache::PerPass && c.core(layer) && pass + 1 < c.num_passes &&
           (c.full_attention[layer] || mode == LoopKvCache::LastAll);
}

// A per-pass executor's image at `position`: each visit's caches in visit order (a full visit's
// rows below the position, a windowed visit's two rings), then the convolution histories.
struct Layout {
    std::vector<std::size_t> offset, bytes;
    std::size_t ring = 0, full_position = 0;
};

Layout layout(const ink::Model& model, KvCacheStorage storage, const ink::Executor& executor,
              std::uint32_t position) {
    const ink::TextConfig& c = model.config();
    const auto schedule      = c.visits();
    const ink::KvFormat f    = ink::kv_format(storage);
    const std::size_t heads = c.num_key_value_heads, d = c.head_dim;
    const auto plane = [](ink::KvCoding coding) {
        return std::size_t(ink::kv_code_bytes(coding)) + std::size_t(ink::kv_scale_bytes(coding));
    };
    Layout out;
    out.full_position = f.quantized() ? heads * (plane(f.key.coding) + plane(f.value.coding))
                                      : 2 * heads * d * sizeof(__half);
    const std::size_t history = schedule.size() * std::size_t(ink::Executor::kHistory) *
                                (2 * heads * d + 2 * std::size_t(c.hidden_size)) * 4;
    std::size_t windowed = 0;
    for (const auto& [layer, pass] : schedule) { windowed += c.full_attention[layer] ? 0 : 1; }
    out.ring = (executor.image_bytes(0) - history) / windowed;
    std::size_t at = 0;
    for (const auto& [layer, pass] : schedule) {
        out.offset.push_back(at);
        out.bytes.push_back(c.full_attention[layer] ? position * out.full_position : out.ring);
        at += out.bytes.back();
    }
    if (at + history != executor.image_bytes(position) ||
        windowed * out.ring + history != executor.image_bytes(0)) {
        throw std::runtime_error("the per-pass image is not laid out as the test reads it");
    }
    return out;
}

// Replaces, in a per-pass executor's sequence, the caches every visit that `mode` shares with
// the last pass's: out as an image, the sections copied, back in.
void transplant(const ink::Model& model, KvCacheStorage storage, ink::Executor& executor,
                std::uint32_t sequence, LoopKvCache mode) {
    const ink::TextConfig& c   = model.config();
    const auto schedule        = c.visits();
    const std::uint32_t at     = executor.position(sequence);
    const Layout l             = layout(model, storage, executor, at);
    const std::size_t bytes    = executor.image_bytes(at);
    std::vector<std::byte> image(bytes);
    std::byte* page = image.data();
    const std::span<std::byte* const> pages(&page, 1);
    const ink::SequenceImage saved = executor.save_image(sequence, nullptr, pages, bytes);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (std::size_t v = 0; v < schedule.size(); ++v) {
        const auto [layer, pass] = schedule[v];
        if (!shared(c, mode, layer, pass)) { continue; }
        std::size_t last = v;
        for (std::size_t u = 0; u < schedule.size(); ++u) {
            if (schedule[u].first == layer && schedule[u].second + 1 == c.num_passes) { last = u; }
        }
        std::memcpy(image.data() + l.offset[v], image.data() + l.offset[last], l.bytes[v]);
    }
    executor.load_image(sequence, saved, pages, bytes);
    CUDA_CHECK(cudaDeviceSynchronize());
}

ink::ExecutorOptions executor_options(KvCacheStorage storage, LoopKvCache mode) {
    ink::ExecutorOptions o;
    o.max_context  = 4096;
    o.sequences    = 2;
    o.kv_cache     = storage;
    o.draft_tokens = 3;
    o.loop_kv      = mode;
    o.expert_cache_bytes = expert_cache;
    return o;
}

void compare_calls(const ink::Model& model, DeviceContext& device, KvCacheStorage storage,
                   const char* format, LoopKvCache mode) {
    const std::vector<std::int32_t> first = tokens(1500, 1), second = tokens(700, 2),
                                    more = tokens(8, 3);
    // The calls: each returns its logits.
    using Call = std::function<std::vector<std::uint16_t>(ink::Executor&)>;
    const std::vector<std::pair<std::string, Call>> calls{
        {"a first prompt chunk",
         [&](ink::Executor& e) {
             e.forward(0, first, 4);
             return logits(e, 4);
         }},
        {"a second chunk",
         [&](ink::Executor& e) {
             e.forward(0, second, 4);
             return logits(e, 4);
         }},
        {"a decode step",
         [&](ink::Executor& e) {
             const std::vector<std::uint32_t> s{0};
             e.decode(s, std::span(more).first(1));
             return logits(e, 1);
         }},
        {"a verification of four rows",
         [&](ink::Executor& e) {
             const std::vector<std::uint32_t> s{0}, kept{2};
             e.verify(s, std::span(more).subspan(1, 4));
             auto out = logits(e, 4);
             e.commit(s, kept);
             return out;
         }},
        {"a decode step after the commit",
         [&](ink::Executor& e) {
             const std::vector<std::uint32_t> s{0};
             e.decode(s, std::span(more).subspan(5, 1));
             return logits(e, 1);
         }},
    };
    std::vector<std::vector<std::uint16_t>> expected;
    {
        ink::Executor reference(model, device, executor_options(storage, LoopKvCache::PerPass));
        reference.warm_up();
        reference.reset(0);
        for (std::size_t i = 0; i < calls.size(); ++i) {
            if (i > 0) { transplant(model, storage, reference, 0, mode); }
            expected.push_back(calls[i].second(reference));
        }
    }
    ink::Executor shared_store(model, device, executor_options(storage, mode));
    shared_store.warm_up();
    shared_store.reset(0);
    for (std::size_t i = 0; i < calls.size(); ++i) {
        const auto got = calls[i].second(shared_store);
        expect(got == expected[i], std::string(format) + ", " + mode_name(mode) + ": " +
                                       calls[i].first +
                                       " reads the last pass's past as the per-pass reference "
                                       "does, bit for bit");
    }
    // A sequence loaded from the shared mode's image decodes as the sequence it came from.
    const std::uint32_t at    = shared_store.position(0);
    const std::size_t bytes   = shared_store.image_bytes(at);
    std::vector<std::byte> image(bytes);
    std::byte* page = image.data();
    const std::span<std::byte* const> pages(&page, 1);
    const ink::SequenceImage saved = shared_store.save_image(0, nullptr, pages, bytes);
    CUDA_CHECK(cudaDeviceSynchronize());
    shared_store.reset(1);
    shared_store.load_image(1, saved, pages, bytes);
    const std::vector<std::uint32_t> both{0, 1};
    const std::vector<std::int32_t> step{more[6], more[6]};
    shared_store.decode(both, step);
    const auto rows   = logits(shared_store, 2);
    const std::size_t vocab = rows.size() / 2;
    expect(std::memcmp(rows.data(), rows.data() + vocab, vocab * 2) == 0,
           std::string(format) + ", " + mode_name(mode) +
               ": a sequence loaded from the image decodes as the sequence it came from");
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_INKLING_ARTIFACT");
    if (artifact_path == nullptr) {
        std::cout << "SKIP Inkling 2 --loop-kv: NINFER_INKLING_ARTIFACT unset\n";
        return 77;
    }
    DeviceContext device(0);
    device.bind_to_current_thread();
    const artifact::Reader reader(artifact_path);
    ink::LoadOptions load{.artifact = artifact_path};
    if (const char* experts = std::getenv("NINFER_INKLING_EXPERTS")) {
        const std::string_view place(experts);
        if (place == "host") { load.experts = ExpertResidency::Host; }
        if (place == "disk") { load.experts = ExpertResidency::Disk; }
        if (place == "host" || place == "disk") { expert_cache = std::uint64_t{512} << 20; }
    }
    auto model = ink::load_model(reader, load, device);
    const ink::TextConfig& c = model->config();

    // The images' and caches' sizes: a position of the shared full-attention store, a ring.
    {
        const KvCacheStorage storage = KvCacheStorage::Mxfp8;
        ink::Executor per_pass(*model, device, executor_options(storage, LoopKvCache::PerPass));
        ink::Executor last_full(*model, device, executor_options(storage, LoopKvCache::LastFull));
        ink::Executor last_all(*model, device, executor_options(storage, LoopKvCache::LastAll));
        const Layout l = layout(*model, storage, per_pass, 1);
        std::size_t full = 0, shared_full = 0, shared_rings = 0;
        for (const auto& [layer, pass] : c.visits()) {
            full += c.full_attention[layer] ? 1 : 0;
            if (c.full_attention[layer]) {
                shared_full += shared(c, LoopKvCache::LastFull, layer, pass) ? 1 : 0;
            } else {
                shared_rings += shared(c, LoopKvCache::LastAll, layer, pass) ? 1 : 0;
            }
        }
        const auto position = [](const ink::Executor& e) { return e.image_bytes(1) - e.image_bytes(0); };
        std::cout << "full-attention visits " << full << ", shared " << shared_full
                  << "; windowed visits sharing a ring under last-all " << shared_rings
                  << "; image bytes a position: per-pass " << position(per_pass) << ", last-full "
                  << position(last_full) << ", last-all " << position(last_all)
                  << "; KV bytes: per-pass " << per_pass.memory().kv_bytes << ", last-full "
                  << last_full.memory().kv_bytes << ", last-all " << last_all.memory().kv_bytes
                  << '\n';
        expect(full == 8 && shared_full == 4 && shared_rings == 20,
               "the core's earlier passes of layers 11 and 17 share the last pass's store, and "
               "under last-all its 10 windowed layers' too");
        expect(position(per_pass) == full * l.full_position &&
                   position(last_full) == (full - shared_full) * l.full_position &&
                   position(last_all) == position(last_full),
               "an image holds the rows of the full-attention stores the mode keeps");
        expect(per_pass.image_bytes(0) == last_full.image_bytes(0) &&
                   last_full.image_bytes(0) - last_all.image_bytes(0) == shared_rings * l.ring,
               "an image holds the rings the mode keeps");
        const std::uint64_t slots = (4096 + 1 + 255) / 256 * 256, sequences = 2;
        expect(per_pass.memory().kv_bytes - last_full.memory().kv_bytes ==
                       shared_full * slots * sequences * l.full_position &&
                   last_full.memory().kv_bytes > last_all.memory().kv_bytes,
               "the shared stores are not allocated");
    }

    const std::pair<KvCacheStorage, const char*> formats[] = {
        {KvCacheStorage::Mxfp8, "mxfp8"},
        {KvCacheStorage::Float16, "fp16"},
        {KvCacheStorage::Int8Group64, "int8"},
        {KvCacheStorage::RotatedInt8KeyInt4ValueGroup64, "rk8v4"},
        {KvCacheStorage::RotatedLloyd4KeyInt4Value, "rk4v4"},
    };
    for (const auto& [storage, name] : formats) {
        compare_calls(*model, device, storage, name, LoopKvCache::LastFull);
    }
    compare_calls(*model, device, KvCacheStorage::Mxfp8, "mxfp8", LoopKvCache::LastAll);

    std::cout << (failures == 0 ? "PASS" : "FAILED") << " Inkling 2 --loop-kv\n";
    return failures == 0 ? 0 : 1;
}
