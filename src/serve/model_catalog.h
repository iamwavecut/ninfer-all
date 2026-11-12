#pragma once

// The models a router-mode server can serve, read from a models directory and a preset file in
// llama.cpp's conventions. Each model carries the layers of serve options it is started with: its
// preset's global section, then its own section, then the router's configuration files and command
// line, each overriding the ones before (product/option_layers.h) -- the command line's precedence
// over presets is llama.cpp's.

#include "product/launch_config/launch_config.h"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

struct CatalogModel {
    // Public id: the preset section name, or the artifact's stem (or its directory's name) in a
    // models directory.
    std::string id;
    std::vector<std::string> aliases;
    std::filesystem::path artifact;
    // Serve option layers for this model, lowest precedence first, without the program name and
    // artifact path.
    std::vector<product::ArgumentLayer> layers;
    bool load_on_startup = false;
};

// The .ninfer artifacts directly in `directory`, and each subdirectory holding exactly one.
[[nodiscard]] std::vector<CatalogModel> discover_models(const std::filesystem::path& directory);

// Merges the directory's models with the preset's sections. A preset (product::read_config_file) has
// `[*]` with defaults for every model; any other section is a model: `model` (or `artifact`) is its
// artifact path, `alias` a comma-separated list of further ids, `load-on-startup` a boolean; every
// other key is a serve option without its leading dashes, as in a launch configuration
// (product::append_config_option). A section whose name matches a model found in the models
// directory configures that model and needs no path. Each model's layers are `[*]`, its own
// section, then `shared` (the router's configuration files and command line). Throws
// std::invalid_argument for an unknown or duplicate id, a section without an artifact, or a
// malformed entry.
[[nodiscard]] std::vector<CatalogModel>
build_catalog(const std::optional<std::filesystem::path>& models_dir,
              const std::optional<product::ConfigFile>& preset,
              const std::vector<product::ArgumentLayer>& shared);

} // namespace ninfer::serve
