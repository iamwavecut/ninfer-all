#pragma once

#include "ninfer/engine_options.h"
#include "ninfer/model_enums.h"
#include "product/option_layers.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "none") { return SpeculativeBackend::None; }
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    if (value == "dspark") { return SpeculativeBackend::DSpark; }
    if (value == "dspine") { return SpeculativeBackend::DSpine; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    case SpeculativeBackend::DSpark:
        return "dspark";
    case SpeculativeBackend::DSpine:
        return "dspine";
    }
    return "unknown";
}

// Copy drafting rides along with every drafter but DSpark and DSpine unless the command line sets
// its width (a DSpine drafter serves models whose Engines have no copy proposals). Fifteen drafts
// is the widest a multi-request batch verifies, so the default holds at any concurrency.
inline constexpr std::uint32_t kDefaultNgramDraftTokens = 15;

inline void apply_default_ngram_draft_tokens(SpeculativeOptions& options,
                                             bool width_explicit) noexcept {
    if (!width_explicit && options.backend != SpeculativeBackend::None &&
        options.backend != SpeculativeBackend::DSpark &&
        options.backend != SpeculativeBackend::DSpine) {
        options.ngram_draft_tokens = kDefaultNgramDraftTokens;
    }
}

// Without --spec the Engine chooses (SpeculativeOptions::automatic), unless a drafting option
// names the backend's own settings.
inline void apply_automatic_speculation(SpeculativeOptions& options, bool backend_explicit) noexcept {
    options.automatic = !backend_explicit && options.backend == SpeculativeBackend::None &&
                        options.draft_tokens == 0;
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    // A value out of range is an error of its own option; the rest are combinations that a later
    // option layer resolves (product/option_layers.h). The floor applies to MTP the command line
    // names or the Engine chooses (no --spec), and to Qwen3.8-Flash-Next's DFlash2 drafter.
    if (options.draft_min_p && (!std::isfinite(*options.draft_min_p) ||
                                *options.draft_min_p < 0 || *options.draft_min_p > 1)) {
        throw std::invalid_argument("--draft-min-p takes a finite value in [0,1]");
    }
    if (options.draft_min_p && options.backend != SpeculativeBackend::Mtp &&
        options.backend != SpeculativeBackend::DFlash2 &&
        options.backend != SpeculativeBackend::DSpine && !options.automatic) {
        throw OptionConflict({"draft-min-p", "spec"},
                             "--draft-min-p requires --spec mtp, dflash2, dspine or no --spec");
    }
    if (options.backend == SpeculativeBackend::DSpine && options.draft_model.empty()) {
        throw OptionConflict({"spec", "draft-model"},
                             "--spec dspine requires --draft-model PATH, the DSpine drafter "
                             "artifact");
    }
    if (!options.draft_model.empty() && options.backend != SpeculativeBackend::DSpine) {
        throw OptionConflict({"draft-model", "spec"}, "--draft-model requires --spec dspine");
    }
    if (options.draft_confidence &&
        (!std::isfinite(*options.draft_confidence) || *options.draft_confidence < 0 ||
         *options.draft_confidence > 1)) {
        throw std::invalid_argument("--draft-confidence takes a finite value in [0,1]");
    }
    if (options.draft_confidence && options.backend != SpeculativeBackend::DSpark) {
        throw OptionConflict({"draft-confidence", "spec"}, "--draft-confidence requires --spec dspark");
    }
    if (options.draft_experts && options.backend != SpeculativeBackend::DSpark) {
        throw OptionConflict({"draft-experts", "spec"}, "--draft-experts requires --spec dspark");
    }
    if (options.ngram_archive_bytes != 0 &&
        (options.ngram_draft_tokens == 0 || options.ngram_session_bytes < (1ULL << 20) ||
         options.ngram_session_bytes > options.ngram_archive_bytes)) {
        throw OptionConflict({"ngram-archive-mib", "ngram-session-mib", "ngram-draft-tokens", "spec"},
                             "ngram archive requires ngram drafting and session capacity between "
                             "1 MiB and total archive capacity");
    }
    if (options.ngram_draft_tokens != 0 &&
        (options.ngram_draft_tokens > 63 || options.ngram_min_match < 4 ||
         options.ngram_min_match > 64)) {
        throw std::invalid_argument("ngram drafting takes drafts 1..63 and match 4..64");
    }
    if (options.ngram_draft_tokens != 0 && options.backend != SpeculativeBackend::DFlash2 &&
        options.backend != SpeculativeBackend::DFlash && options.backend != SpeculativeBackend::Mtp &&
        options.backend != SpeculativeBackend::DSpine) {
        throw OptionConflict({"ngram-draft-tokens", "spec"},
                             "ngram drafting requires --spec mtp|dflash|dflash2|dspine");
    }
    if (options.mtp_attention_window != 0 &&
        (options.backend != SpeculativeBackend::Mtp ||
         options.mtp_attention_window < options.draft_tokens + 1)) {
        throw OptionConflict(
            {"mtp-attention-window", "spec", "draft-tokens"},
            "--mtp-attention-window requires --spec mtp and at least --draft-tokens + 1 keys");
    }
    // The Qwen3.5-core models also adapt a masked drafter's rounds; the other families' Engines
    // refuse that themselves.
    if (options.mtp_policy != MtpDraftPolicy::Fixed && options.backend != SpeculativeBackend::Mtp &&
        options.backend != SpeculativeBackend::DFlash &&
        options.backend != SpeculativeBackend::DFlash2 &&
        options.backend != SpeculativeBackend::DSpine) {
        throw OptionConflict({"adaptive-mtp", "spec"},
                             "--adaptive-mtp requires --spec mtp|dflash|dflash2|dspine");
    }
    switch (options.backend) {
    case SpeculativeBackend::None:
        // Without --spec, --draft-tokens sizes a context lookup's proposals (Flint, DeepSeek-V4.1);
        // the model's Engine checks that it drafts by lookup alone and the range it verifies.
        if ((options.draft_tokens != 0 && options.lookup_ngram == 0) ||
            options.proposal_head != ProposalHead::Auto) {
            throw OptionConflict({"lm-head-draft", "draft-tokens", "lookup-ngram", "spec"},
                                 "--lm-head-draft, --full-draft-head and --draft-tokens without "
                                 "--lookup-ngram require --spec mtp|dflash|dflash2|dspark|dspine");
        }
        return;
    case SpeculativeBackend::Mtp:
    case SpeculativeBackend::DFlash:
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw OptionConflict({"spec", "draft-tokens"},
                                 std::string("--spec ") + speculative_backend_name(options.backend) +
                                     " requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DSpark:
        // Without --draft-tokens a round verifies the drafter's whole block.
        if (options.draft_tokens > 15 || options.proposal_head != ProposalHead::Auto) {
            throw OptionConflict({"spec", "draft-tokens", "lm-head-draft"},
                                 "--spec dspark takes --draft-tokens up to 15 and no proposal head");
        }
        return;
    case SpeculativeBackend::DSpine:
        // The model's Engine checks the drafts against the drafter's block.
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw OptionConflict({"spec", "draft-tokens"},
                                 "--spec dspine requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
