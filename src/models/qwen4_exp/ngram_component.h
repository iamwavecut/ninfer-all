#pragma once

// The n-gram table of a Qwen3.8-Flash-Next model is described by its artifact's `ngram`
// component: the hash constants the rows were written for, their format and the SHA-256 of the
// table's bytes. The rows themselves are stored either in the same artifact or in a separate table
// artifact (an `ngram` component alone). A model reads any table of its hash constants and
// geometry, in any row format, the one its component names or another; nothing compares the rows'
// values. The rows stay where the artifact stores them, and nothing reads them at load;
// NgramTableReader reads the rows each token addresses from there, or loads the whole table into
// RAM.

#include "artifact/reader.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/ngram_table.h"
#include "models/qwen4_exp/ngram_profile.h"
#include "ninfer/ops/ngram_rows.h"

#include <filesystem>
#include <optional>
#include <string>

namespace ninfer::models::qwen4_exp {

struct NgramTableSource {
    NgramTableLayout layout;
    ops::NgramRowFormat format = ops::NgramRowFormat::Bf16;
    std::optional<NgramProfile> hot_profile;
    // The selected table's descriptor, and the one the model's component names; they differ when
    // another table of the same shape stands in for the model's.
    std::string table_format;
    std::string table_digest;
    std::string model_format;
    std::string model_digest;
};

// Locates the table of the model artifact `reader` opened at `artifact`: the rows of the table
// artifact at `table` when one is given, otherwise the model's own rows. Refuses a table whose
// hash constants, row count or row width differ from the model's, rows the runtime cannot decode,
// and a model that stores no rows when `table` is empty. When requested, reads and validates the
// selected table artifact's `ngram.resources.hot_profile`; absent profiles remain empty for the
// caller to report.
[[nodiscard]] NgramTableSource ngram_table_source(const artifact::Reader& reader,
                                                  const std::filesystem::path& artifact,
                                                  const TextConfig& config,
                                                  const std::filesystem::path& table = {},
                                                  bool read_hot_profile = false);

// True when the artifact holds an n-gram table and no model.
[[nodiscard]] bool is_ngram_table_artifact(const artifact::Reader& reader);

} // namespace ninfer::models::qwen4_exp
