#include "serve/model_catalog.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ninfer::serve {
namespace {

std::string trim(std::string text) {
    const auto first = std::find_if_not(text.begin(), text.end(),
                                        [](unsigned char c) { return std::isspace(c) != 0; });
    const auto last  = std::find_if_not(text.rbegin(), text.rend(), [](unsigned char c) {
                          return std::isspace(c) != 0;
                       }).base();
    return first < last ? std::string(first, last) : std::string();
}

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

bool is_artifact(const std::filesystem::path& path) {
    return path.extension() == ".ninfer" && std::filesystem::is_regular_file(path);
}

std::vector<std::string> split_aliases(const std::string& value) {
    std::vector<std::string> out;
    std::stringstream stream(value);
    for (std::string item; std::getline(stream, item, ',');) {
        item = trim(item);
        if (!item.empty()) { out.push_back(std::move(item)); }
    }
    return out;
}

// Keys a preset uses that are not serve options.
bool catalog_key(const std::string& key) {
    return key == "model" || key == "artifact" || key == "alias" || key == "load-on-startup" ||
           key == "stop-timeout" || key == "version";
}

bool parse_bool(const std::string& section, const std::string& key, const std::string& value) {
    const std::string word = lower(value);
    if (word == "true" || word == "on" || word == "yes" || word == "1") { return true; }
    if (word == "false" || word == "off" || word == "no" || word == "0") { return false; }
    throw std::invalid_argument("preset [" + section + "]: " + key + " must be a boolean");
}

} // namespace

std::vector<CatalogModel> discover_models(const std::filesystem::path& directory) {
    if (!std::filesystem::is_directory(directory)) {
        throw std::invalid_argument("--models-dir is not a directory: " + directory.string());
    }
    std::vector<CatalogModel> out;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (is_artifact(entry.path())) {
            out.push_back({.id = entry.path().stem().string(), .artifact = entry.path()});
            continue;
        }
        if (!entry.is_directory()) { continue; }
        std::vector<std::filesystem::path> artifacts;
        for (const auto& inner : std::filesystem::directory_iterator(entry.path())) {
            if (is_artifact(inner.path())) { artifacts.push_back(inner.path()); }
        }
        if (artifacts.size() == 1) {
            out.push_back({.id = entry.path().filename().string(), .artifact = artifacts.front()});
        }
    }
    std::sort(out.begin(), out.end(),
              [](const CatalogModel& a, const CatalogModel& b) { return a.id < b.id; });
    return out;
}

std::vector<CatalogModel> build_catalog(const std::optional<std::filesystem::path>& models_dir,
                                        const std::optional<product::ConfigFile>& preset,
                                        const std::vector<product::ArgumentLayer>& shared) {
    std::vector<CatalogModel> models;
    if (models_dir) { models = discover_models(*models_dir); }
    product::ArgumentLayer global{.source = "preset [*]", .arguments = {}};
    if (preset) {
        for (const auto& section : preset->sections) {
            if (section.name != "*") { continue; }
            for (const auto& [key, value] : section.entries) {
                if (catalog_key(key)) { continue; }
                product::append_config_option(global.arguments, "preset [*]: " + key, key, value);
            }
        }
        for (const auto& section : preset->sections) {
            if (section.name == "*") { continue; }
            auto model = std::find_if(models.begin(), models.end(),
                                      [&](const CatalogModel& m) { return m.id == section.name; });
            if (model == models.end()) {
                models.push_back({.id = section.name});
                model = std::prev(models.end());
            }
            model->layers.push_back({.source = "preset [" + section.name + "]", .arguments = {}});
            for (const auto& [key, value] : section.entries) {
                if (key == "model" || key == "artifact") {
                    model->artifact = value;
                } else if (key == "alias") {
                    const auto aliases = split_aliases(value);
                    model->aliases.insert(model->aliases.end(), aliases.begin(), aliases.end());
                } else if (key == "load-on-startup") {
                    model->load_on_startup = parse_bool(section.name, key, value);
                } else if (!catalog_key(key)) {
                    product::append_config_option(model->layers.back().arguments,
                                                  "preset [" + section.name + "]: " + key, key,
                                                  value);
                }
            }
            if (model->artifact.empty()) {
                throw std::invalid_argument("preset [" + section.name +
                                            "] names no artifact (model = PATH) and no such model "
                                            "is in --models-dir");
            }
        }
    }
    std::set<std::string> seen;
    for (CatalogModel& model : models) {
        for (const std::string& name : [&] {
                 std::vector<std::string> names{model.id};
                 names.insert(names.end(), model.aliases.begin(), model.aliases.end());
                 return names;
             }()) {
            if (!seen.insert(name).second) {
                throw std::invalid_argument("model id or alias '" + name + "' is used twice");
            }
        }
        // Precedence, lowest first: the preset's global section, the model's own section, then the
        // router's configuration files and command line.
        std::vector<product::ArgumentLayer> layers{global};
        layers.insert(layers.end(), model.layers.begin(), model.layers.end());
        layers.insert(layers.end(), shared.begin(), shared.end());
        model.layers = std::move(layers);
    }
    if (models.empty()) {
        throw std::invalid_argument(
            "router mode found no models in --models-dir or --models-preset");
    }
    return models;
}

} // namespace ninfer::serve
