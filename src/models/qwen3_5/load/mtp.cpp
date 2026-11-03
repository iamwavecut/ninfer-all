#include "models/qwen3_5/load/bindings.h"

namespace ninfer::models::qwen3_5::loading {

MtpWeights bind_mtp(Bindings& b, const TextConfig& config, const TextWeights& target) {
    const auto h = config.hidden_size;
    MtpWeights out;
    out.input_projection = b.parameter("mtp/input_projection", {h, 2ULL * h}, {"mtp/stem_input"});
    out.embedding_norm   = b.direct("mtp/embedding_norm", {h});
    out.hidden_norm      = b.direct("mtp/hidden_norm", {h});
    out.final_norm       = b.direct("mtp/final_norm", {h});
    // Under tensor parallelism every rank holds its rows of a dense FFN, and every rank holding
    // KV heads its share of the attention, as in a text layer.
    const std::string p = "mtp/layers/0/";
    std::vector<DenseWeights> shares;
    std::vector<AttentionWeights> attention;
    if (!target.tensor_shares.empty()) {
        if (const auto* dense = std::get_if<DenseConfig>(&config.ffn)) {
            shares = bind_dense_shares(b, h, dense->intermediate_size, p, target.tensor_shares);
        }
        attention = bind_attention_shares(b, config, p, target.tensor_shares);
    }
    out.layer = bind_block(b, config, p, MixerKind::FullAttention,
                           shares.empty() ? nullptr : &shares.front(),
                           attention.empty() ? nullptr : &attention.front());
    if (!shares.empty()) { out.rank_ffn.assign(shares.begin() + 1, shares.end()); }
    if (!attention.empty()) {
        // Rank 0 projects a prompt chunk's keys and values for every rank: it holds all their
        // rows as a piece, so the object they are stored in with the queries and gates stays off.
        const auto k            = config.attention.value().key_width();
        const auto input        = std::vector<std::string>{p + "mixer_input"};
        const ParameterPiece kv = {ParameterPiece::Cut::Rows, {{0, k}}};
        out.whole_key =
            b.piece(p + "attention/key", {k, h}, input, {}, 0, kv, p + "attention/key@whole");
        out.whole_value =
            b.piece(p + "attention/value", {k, h}, input, {}, 0, kv, p + "attention/value@whole");
        out.rank_attention.assign(attention.begin() + 1, attention.end());
        for (std::size_t rank = 1; rank < target.tensor_shares.size(); ++rank) {
            out.rank_input_norm.push_back(bind_norm_replica(b, p + "input_norm", h, rank));
            out.rank_post_attention_norm.push_back(
                bind_norm_replica(b, p + "post_attention_norm", h, rank));
        }
    }
    out.token_embedding = target.token_embedding;
    out.output_head     = target.output_head;
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
