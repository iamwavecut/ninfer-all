#pragma once

// The block-table metadata a prompt kernel unit passes its kernel, built from the launch's fields
// (prompt_kernel_launch.h).

#include "ops/kernel/paged_kv_address.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_kernel_launch.h"

#include <type_traits>

namespace ninfer::ops::detail {

template <PromptMetadataForm Form>
using PromptMetadata =
    std::conditional_t<Form == PromptMetadataForm::Direct, PagedKVDirectMetadata,
                       PagedKVBatchMetadata<Form == PromptMetadataForm::BatchMasked>>;

template <PromptMetadataForm Form>
[[nodiscard]] PromptMetadata<Form> prompt_metadata(const PromptLaunch& launch) {
    if constexpr (Form == PromptMetadataForm::Direct) {
        return PagedKVDirectMetadata{launch.tables};
    } else {
        return PromptMetadata<Form>{
            .tables        = launch.tables,
            .valid_columns = launch.valid_columns,
            .table_rows    = launch.table_rows,
            .table_stride  = launch.table_stride,
        };
    }
}

} // namespace ninfer::ops::detail
