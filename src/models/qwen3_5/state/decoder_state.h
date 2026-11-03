#pragma once

#include "core/layout.h"
#include "core/paged_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5 {

inline constexpr std::int32_t kKvInt8QuantGroup = 64;
inline constexpr std::int32_t kKvFp8QuantGroup  = 256;

struct DecoderStateSpec {
    std::uint32_t full_attention_layers     = 0;
    std::uint32_t mtp_layers                = 0;
    std::uint32_t capacity                  = 0;
    std::int32_t kv_heads                   = 0;
    std::int32_t attention_head_dim         = 0;
    KvCacheStorage kv_storage                = KvCacheStorage::BFloat16;
    bool enable_mtp                         = false;
    std::int32_t kv_table_rows              = 1;
    std::uint32_t text_physical_page_groups = 0;
    std::uint32_t mtp_physical_page_groups  = 0;
    // The rank holding each full-attention layer's KV planes; empty puts every layer on rank 0.
    std::vector<std::size_t> text_layer_rank;
    // Tensor parallelism: how many of every full-attention layer's KV heads each rank holds, in
    // rank order (ranks [0, size) each hold a non-empty contiguous run). Empty keeps every head of
    // a layer on its `text_layer_rank` rank.
    std::vector<std::int32_t> text_kv_head_shares;
    // Position-split attention (--tensor-attention positions): every full-attention layer has
    // planes on ranks [0, size), each holding every KV head of the physical pages p with
    // p % size == its rank. Zero (or one) keeps the head shares or layer ranks above.
    std::uint32_t text_kv_position_ranks = 0;
    // Tensor parallelism dividing MTP's attention: how many of its KV heads each rank holds, as
    // `text_kv_head_shares`. Empty keeps the MTP cache whole on rank 0.
    std::vector<std::int32_t> mtp_kv_head_shares;
    // Invariant attention reductions over the text cache (--invariant-attention): its attention
    // calls split their keys into fixed segments for the capacity
    // (ops::causal_attention_invariant_segment_keys).
    bool invariant_attention = false;
};

struct PagedKVCacheLayout {
    DeviceKVPagePoolLayout pages;
    // One copy of the block tables per rank that runs attention layers of this cache.
    std::vector<KVExecutionTableLayout> execution_tables;
    std::uint32_t layers      = 0;
    std::uint32_t max_context = 0;
    std::int32_t kv_heads     = 0;
    std::int32_t head_dim     = 0;
    KvCacheStorage storage    = KvCacheStorage::BFloat16;
    // Tensor parallelism: each rank's KV heads of every layer; empty when one rank holds a layer.
    std::vector<std::int32_t> head_shares;
    // Position-split attention: the ranks whose planes divide every layer's pages; 0 otherwise.
    std::uint32_t position_ranks = 0;
    // Invariant attention reductions: the keys of one split, 0 for the tuned splits.
    std::uint32_t attention_segment_keys = 0;

    [[nodiscard]] std::size_t payload_bytes() const noexcept { return pages.payload_bytes(); }
};

class PagedKVCache;

class PagedKVCacheView {
public:
    PagedKVCacheView() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return cache_ != nullptr; }

    [[nodiscard]] std::uint32_t max_context() const noexcept;
    // `rank`: a tensor-parallel rank's heads of the layer; 0 otherwise.
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer, std::size_t rank = 0) const;

private:
    friend class PagedKVCache;
    PagedKVCacheView(const PagedKVCache& cache, KVExecutionRowHandle row) noexcept;

    const PagedKVCache* cache_ = nullptr;
    // The row, not a tensor: each layer reads the copy of its row held by the layer's own rank.
    KVExecutionRowHandle row_;
};

class PagedKVCache {
public:
    // `backings[r]` is rank r's persistent device memory.
    PagedKVCache(std::span<const DeviceSpan> backings, const PagedKVCacheLayout& layout);
    // Every plane and table on rank 0.
    PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout);

    PagedKVCache(const PagedKVCache&)            = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    PagedKVCache(PagedKVCache&&)                 = delete;
    PagedKVCache& operator=(PagedKVCache&&)      = delete;

    [[nodiscard]] std::uint32_t max_context() const noexcept { return max_context_; }

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }

    [[nodiscard]] DeviceKVPagePool& page_pool() noexcept { return pages_; }

    [[nodiscard]] const DeviceKVPagePool& page_pool() const noexcept { return pages_; }

    [[nodiscard]] KVExecutionTablePool& execution_tables() noexcept { return execution_tables_; }

    [[nodiscard]] const KVExecutionTablePool& execution_tables() const noexcept {
        return execution_tables_;
    }

    [[nodiscard]] PagedKVCacheView execution_view(const KVExecutionRowLease& row) const;

    // `rank`: a tensor-parallel rank's heads of the layer; 0 otherwise.
    [[nodiscard]] PagedKVBatchLayerView batch_layer_view(std::uint32_t layer,
                                                         std::size_t rank = 0) const;
    // The rank holding this layer's KV planes, which is also the rank whose block table it reads.
    // With tensor parallelism every rank in `head_shares()` holds some of them; this is rank 0.
    [[nodiscard]] std::size_t layer_rank(std::uint32_t layer) const;

    // Tensor parallelism: each rank's KV heads of every layer; empty otherwise.
    [[nodiscard]] std::span<const std::int32_t> head_shares() const noexcept {
        return head_shares_;
    }

    // Position-split attention: the ranks whose planes divide every layer's pages (each rank's
    // views give its page stride and residue); 0 otherwise.
    [[nodiscard]] std::uint32_t position_ranks() const noexcept { return position_ranks_; }

    // Invariant attention reductions: the keys of one split every attention call over this cache
    // takes (CausalAttentionExecutionEnvelope::invariant_segment_keys); 0 for the tuned splits.
    [[nodiscard]] std::uint32_t attention_segment_keys() const noexcept {
        return attention_segment_keys_;
    }

private:
    friend class PagedKVCacheView;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer, const KVExecutionRowHandle* row,
                                              std::size_t rank) const;
    // A rank's plane of a layer: `kind` 0 K, 1 V, 2 K scales, 3 V scales.
    [[nodiscard]] std::size_t plane_index(std::uint32_t layer, std::size_t rank,
                                          std::size_t kind) const;

    DeviceKVPagePool pages_;
    KVExecutionTablePool execution_tables_;
    std::uint32_t layers_      = 0;
    std::uint32_t max_context_ = 0;
    std::int32_t kv_heads_     = 0;
    std::int32_t head_dim_     = 0;
    KvCacheStorage storage_    = KvCacheStorage::BFloat16;
    std::vector<std::int32_t> head_shares_;
    std::uint32_t position_ranks_         = 0;
    std::uint32_t attention_segment_keys_ = 0;
};

struct DecoderStateLayout {
    PagedKVCacheLayout text_kv;
    std::optional<PagedKVCacheLayout> mtp_kv;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
};

[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec);
// One layout builder per rank: each layer's planes and each rank's block-table copy are laid out in
// that rank's own backing.
[[nodiscard]] DecoderStateLayout plan_decoder_state(std::span<LayoutBuilder* const> builders,
                                                    const DecoderStateSpec& spec);

struct DecoderState {
    PagedKVCache text_kv;
    std::optional<PagedKVCache> mtp_kv;

    DecoderState(std::span<const DeviceSpan> backings, const DecoderStateLayout& layout);
    DecoderState(DeviceSpan backing, const DecoderStateLayout& layout);

    [[nodiscard]] PagedKVCache* mtp_cache() noexcept;
    [[nodiscard]] const PagedKVCache* mtp_cache() const noexcept;
};

} // namespace ninfer::models::qwen3_5
