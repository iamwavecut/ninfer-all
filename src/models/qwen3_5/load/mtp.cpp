#include "models/qwen3_5/load/bindings.h"

namespace ninfer::models::qwen3_5::loading {

MtpWeights bind_mtp(Bindings& b, const TextConfig& config, const TextWeights& target) {
    const auto h = config.hidden_size;
    MtpWeights out;
    out.input_projection = b.parameter("mtp/input_projection", {h, 2ULL * h}, {"mtp/stem_input"});
    out.embedding_norm   = b.direct("mtp/embedding_norm", {h});
    out.hidden_norm      = b.direct("mtp/hidden_norm", {h});
    out.final_norm       = b.direct("mtp/final_norm", {h});
    // Under tensor parallelism every rank holds its rows of a dense FFN, as in a text layer.
    std::vector<DenseWeights> shares;
    if (const auto* dense = std::get_if<DenseConfig>(&config.ffn);
        dense != nullptr && !target.tensor_shares.empty()) {
        shares = bind_dense_shares(b, h, dense->intermediate_size, "mtp/layers/0/",
                                   target.tensor_shares);
    }
    out.layer = bind_block(b, config, "mtp/layers/0/", MixerKind::FullAttention,
                           shares.empty() ? nullptr : &shares.front());
    if (!shares.empty()) { out.rank_ffn.assign(shares.begin() + 1, shares.end()); }
    out.token_embedding = target.token_embedding;
    out.output_head     = target.output_head;
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
