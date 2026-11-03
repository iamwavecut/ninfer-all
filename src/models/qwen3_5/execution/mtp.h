#pragma once

#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t mtp_projection_workspace_bytes(const MtpProjectionParameters& parameters,
                                                         std::int32_t first, std::int32_t last);
[[nodiscard]] std::size_t mtp_kv_workspace_bytes(const MtpProjectionParameters& parameters,
                                                 const AttentionConfig& config, std::int32_t first,
                                                 std::int32_t last);
[[nodiscard]] std::size_t mtp_query_gate_workspace_bytes(const MtpProjectionParameters& parameters,
                                                         const AttentionConfig& config,
                                                         std::int32_t first, std::int32_t last);
void mtp_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                    const AttentionConfig& config, Tensor& query, Tensor& gate, Tensor& key,
                    Tensor& value, WorkspaceArena& workspace, cudaStream_t stream);
void mtp_kv_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                       const AttentionConfig& config, Tensor& key, Tensor& value,
                       WorkspaceArena& workspace, cudaStream_t stream);
void mtp_query_gate_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                               const AttentionConfig& config, Tensor& query, Tensor& gate,
                               WorkspaceArena& workspace, cudaStream_t stream);
// K and V through their own rows `key` and `value`: the rows form of mtp_kv_projection, and a
// divided attention's whole rows on rank 0 (MtpParameters::whole_key).
[[nodiscard]] std::size_t mtp_kv_rows_workspace_bytes(const LinearParameters& key,
                                                      const LinearParameters& value,
                                                      std::int32_t first, std::int32_t last);
void mtp_kv_rows_projection(const Tensor& hidden, const LinearParameters& key,
                            const LinearParameters& value, Tensor& key_out, Tensor& value_out,
                            WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
