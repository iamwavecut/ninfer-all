#include "models/qwen3_5/state/decoder_state.h"

#include "ninfer/ops/softmax_attention.h"
#include "ops/kv_cache/d256_profile.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(std::span<LayoutBuilder* const> builders,
                              std::span<const std::size_t> layer_rank,
                              std::span<const std::int32_t> head_shares,
                              std::uint32_t position_ranks, std::uint32_t layers,
                              std::uint32_t capacity, std::int32_t kv_heads, std::int32_t head_dim,
                              KvCacheStorage storage, std::int32_t table_rows,
                              std::uint32_t physical_page_groups, bool invariant_attention) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    if (head_dim != ops::kD256KVCacheHeadDim) {
        throw std::invalid_argument("Paged KV cache dtype or quantization is invalid");
    }
    const ops::D256KVCacheProfile profile = ops::d256_kv_cache_profile(storage);
    // Scale (or side record) planes: every storage but BF16's; bf16p has a side record and no group.
    const bool scaled                     = profile.scale_leading_extent != 0;

    const std::uint32_t logical_pages = page_count(capacity);
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    if (!layer_rank.empty() && layer_rank.size() != layers) {
        throw std::invalid_argument("Paged KV layer rank map does not cover every layer");
    }
    if (!head_shares.empty()) {
        std::int32_t heads = 0;
        for (const std::int32_t share : head_shares) {
            if (share <= 0) { throw std::invalid_argument("Paged KV head share is empty"); }
            heads += share;
        }
        if (heads != kv_heads || !layer_rank.empty()) {
            throw std::invalid_argument("Paged KV head shares do not cover the KV heads");
        }
    }
    const bool positions = position_ranks > 1;
    if (positions && (!head_shares.empty() || !layer_rank.empty() ||
                      position_ranks > static_cast<std::uint32_t>(builders.size()) ||
                      position_ranks > physical_page_groups)) {
        throw std::invalid_argument("Paged KV position ranks are invalid");
    }
    const auto rank_of = [&](std::uint32_t layer) {
        return layer_rank.empty() ? std::size_t{0} : layer_rank[layer];
    };

    // A layer's planes: one run of K, V (and their scales) per rank holding some of its heads, in
    // rank order, each run with that rank's heads; or, split by positions, each of K, V (and their
    // scales) as one run of the ranks' residue planes in rank order, every plane with every head,
    // since the pool keeps residue siblings adjacent.
    const std::vector<std::int32_t> one_share{kv_heads};
    const std::vector<std::int32_t> position_shares(positions ? position_ranks : 0U, kv_heads);
    const std::span<const std::int32_t> shares =
        positions ? std::span<const std::int32_t>(position_shares)
                  : (head_shares.empty() ? std::span<const std::int32_t>(one_share) : head_shares);
    KVPageGeometry geometry;
    std::vector<std::size_t> plane_rank;
    const std::size_t planes_per_share = scaled ? 4ULL : 2ULL;
    const std::size_t planes_per_layer = planes_per_share * shares.size();
    geometry.planes.reserve(static_cast<std::size_t>(layers) * planes_per_layer);
    plane_rank.reserve(static_cast<std::size_t>(layers) * planes_per_layer);
    const std::pair<DType, std::int32_t> kinds[] = {
        {profile.key_code_dtype, profile.key_leading_extent},
        {profile.value_code_dtype, profile.value_leading_extent},
        {profile.key_scale_dtype, profile.scale_leading_extent},
        {profile.value_scale_dtype, profile.value_scale_leading_extent},
    };
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        if (positions) {
            for (std::size_t kind = 0; kind < planes_per_share; ++kind) {
                for (std::uint32_t residue = 0; residue < position_ranks; ++residue) {
                    KVPlaneGeometry plane{kinds[kind].first, kinds[kind].second, kv_heads, 256};
                    plane.page_stride  = static_cast<std::int32_t>(position_ranks);
                    plane.page_residue = static_cast<std::int32_t>(residue);
                    geometry.planes.push_back(plane);
                    plane_rank.push_back(residue);
                }
            }
            continue;
        }
        for (std::size_t share = 0; share < shares.size(); ++share) {
            for (std::size_t kind = 0; kind < planes_per_share; ++kind) {
                geometry.planes.push_back(
                    KVPlaneGeometry{kinds[kind].first, kinds[kind].second, shares[share], 256});
            }
            plane_rank.insert(plane_rank.end(), planes_per_share,
                              head_shares.empty() ? rank_of(layer) : share);
        }
    }

    // A block-table copy on every rank that runs attention layers of this cache, and always on
    // rank 0, which is where callers that address the tables directly look.
    std::vector<std::size_t> table_ranks{0};
    for (const std::size_t rank : plane_rank) {
        if (std::find(table_ranks.begin(), table_ranks.end(), rank) == table_ranks.end()) {
            table_ranks.push_back(rank);
        }
    }
    std::sort(table_ranks.begin(), table_ranks.end());

    PagedKVCacheLayout out{
        .pages =
            plan_device_kv_page_pool(builders, plane_rank,
                                     DeviceKVPagePoolSpec{.page_group_count = physical_page_groups,
                                                          .geometry         = std::move(geometry)}),
        .execution_tables = {},
        .layers           = layers,
        .max_context      = capacity,
        .kv_heads         = kv_heads,
        .head_dim         = head_dim,
        .storage          = storage,
        .head_shares      = std::vector<std::int32_t>(head_shares.begin(), head_shares.end()),
        .position_ranks   = positions ? position_ranks : 0U,
        .attention_segment_keys =
            invariant_attention ? ops::causal_attention_invariant_segment_keys(capacity) : 0U,
    };
    for (const std::size_t rank : table_ranks) {
        if (rank >= builders.size() || builders[rank] == nullptr) {
            throw std::invalid_argument("Paged KV layer names a rank with no layout builder");
        }
        out.execution_tables.push_back(plan_kv_execution_tables(
            *builders[rank],
            KVExecutionTableSpec{.logical_page_capacity = logical_pages, .table_rows = table_rows},
            rank));
    }
    return out;
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    LayoutBuilder* const builders[] = {&builder};
    return plan_decoder_state(builders, spec);
}

DecoderStateLayout plan_decoder_state(std::span<LayoutBuilder* const> builders,
                                      const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builders, spec.text_layer_rank, spec.text_kv_head_shares,
                                spec.text_kv_position_ranks, spec.full_attention_layers,
                                spec.capacity, spec.kv_heads, spec.attention_head_dim,
                                spec.kv_storage, spec.kv_table_rows,
                                spec.text_physical_page_groups, spec.invariant_attention);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(spec.mtp_kv_head_shares.empty() ? builders.first(1) : builders,
                                   {}, spec.mtp_kv_head_shares, 0, spec.mtp_layers, spec.capacity,
                                   spec.kv_heads, spec.attention_head_dim, spec.kv_storage,
                                   spec.kv_table_rows, spec.mtp_physical_page_groups, false);
    }
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : PagedKVCache(std::span<const DeviceSpan>(&backing, 1), layout) {}

PagedKVCache::PagedKVCache(std::span<const DeviceSpan> backings, const PagedKVCacheLayout& layout)
    : pages_(backings, layout.pages), execution_tables_(backings, layout.execution_tables, pages_),
      layers_(layout.layers), max_context_(layout.max_context), kv_heads_(layout.kv_heads),
      head_dim_(layout.head_dim), storage_(layout.storage), head_shares_(layout.head_shares),
      position_ranks_(layout.position_ranks),
      attention_segment_keys_(layout.attention_segment_keys) {}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, KVExecutionRowHandle row) noexcept
    : cache_(&cache), row_(row) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer, std::size_t rank) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, &row_, rank);
}

PagedKVCacheView PagedKVCache::execution_view(const KVExecutionRowLease& row) const {
    if (!row.belongs_to(execution_tables_)) {
        throw std::invalid_argument("Paged KV execution row belongs to another cache");
    }
    return PagedKVCacheView(*this, row.handle());
}

std::size_t PagedKVCache::plane_index(std::uint32_t layer, std::size_t rank,
                                      std::size_t kind) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const std::size_t shares =
        position_ranks_ > 1 ? position_ranks_ : (head_shares_.empty() ? 1 : head_shares_.size());
    if (rank >= shares) { throw std::out_of_range("Paged KV rank holds no heads of the layer"); }
    const bool scaled        = ops::d256_kv_cache_profile(storage_).scale_leading_extent != 0;
    const std::size_t kinds  = scaled ? 4ULL : 2ULL;
    const std::size_t first  = static_cast<std::size_t>(layer) * shares * kinds;
    return position_ranks_ > 1 ? first + kind * shares + rank : first + rank * kinds + kind;
}

std::size_t PagedKVCache::layer_rank(std::uint32_t layer) const {
    return pages_.plane_rank(plane_index(layer, 0, 0));
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, const KVExecutionRowHandle* row,
                                          std::size_t rank) const {
    const bool scaled      = ops::d256_kv_cache_profile(storage_).scale_leading_extent != 0;
    const std::size_t base = plane_index(layer, rank, 0);
    // The layer reads the copy of the table on the rank holding its planes; a kernel cannot follow
    // a pointer into another device's memory.
    const std::size_t holder = pages_.plane_rank(base);
    const Tensor block_table = row == nullptr ? Tensor() : execution_tables_.row(*row, holder);
    return PagedKVLayerView{
        .k_pages       = pages_.plane(base),
        .v_pages       = pages_.plane(plane_index(layer, rank, 1)),
        .k_scale_pages = scaled ? pages_.plane(plane_index(layer, rank, 2)) : Tensor(),
        .v_scale_pages = scaled ? pages_.plane(plane_index(layer, rank, 3)) : Tensor(),
        .block_table   = block_table,
        .head_dim      = head_dim_,
        .num_kv_heads  = head_shares_.empty() ? kv_heads_ : head_shares_[rank],
        .storage       = storage_,
        .page_stride   = position_ranks_ > 1 ? static_cast<std::int32_t>(position_ranks_) : 1,
        .page_residue  = position_ranks_ > 1 ? static_cast<std::int32_t>(rank) : 0,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer, std::size_t rank) const {
    const PagedKVLayerView direct = layer_view(layer, nullptr, rank);
    return PagedKVBatchLayerView{
        .k_pages       = direct.k_pages,
        .v_pages       = direct.v_pages,
        .k_scale_pages = direct.k_scale_pages,
        .v_scale_pages = direct.v_scale_pages,
        .block_tables  = execution_tables_.matrix(pages_.plane_rank(plane_index(layer, rank, 0))),
        .head_dim      = direct.head_dim,
        .num_kv_heads  = direct.num_kv_heads,
        .storage       = direct.storage,
        .page_stride   = direct.page_stride,
        .page_residue  = direct.page_residue,
    };
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : DecoderState(std::span<const DeviceSpan>(&backing, 1), layout) {}

DecoderState::DecoderState(std::span<const DeviceSpan> backings, const DecoderStateLayout& layout)
    : text_kv(backings, layout.text_kv) {
    if (layout.mtp_kv) { mtp_kv.emplace(backings, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::models::qwen3_5
