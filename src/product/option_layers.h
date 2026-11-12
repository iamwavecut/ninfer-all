#pragma once

// Layers of command-line options and how a later layer overrides an earlier one. A run's options
// come from its configuration files (`--config FILE`, in order) and then its command line; each
// layer is a list of arguments in the command line's own spelling. Every program parses the merged
// list with its usual parser after these rules:
//
//   - A setting a later layer gives replaces every option an earlier layer gave for it: a value, a
//     repeatable list as a whole, a switch and its `--no-` form, and the other spellings of the
//     setting a program names as aliases (`--device` and `--devices`).
//   - When the parser refuses a combination with an OptionConflict and the options it names come
//     from different layers, the options of the earlier layers are dropped and the merged list is
//     parsed again: a later layer that switches speculation off also takes the earlier layers'
//     draft settings away.

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::product {

// A combination of options the program refuses, naming the settings involved (see option_setting).
// A layered parse resolves it when those options come from different layers; within one layer it
// is the error it says.
class OptionConflict : public std::invalid_argument {
public:
    OptionConflict(std::initializer_list<std::string_view> settings, const std::string& message)
        : std::invalid_argument(message), settings_(settings.begin(), settings.end()) {}
    OptionConflict(std::vector<std::string> settings, const std::string& message)
        : std::invalid_argument(message), settings_(std::move(settings)) {}

    [[nodiscard]] const std::vector<std::string>& settings() const noexcept { return settings_; }

private:
    std::vector<std::string> settings_;
};

// `option` spelled as `--name` sets `set(true)`, as `--no-name` `set(false)`. Returns whether it
// was either. Every switch of every program takes both forms, so a later layer can turn off what
// an earlier one turned on.
template <class Set>
bool switch_option(std::string_view option, std::string_view name, Set&& set) {
    if (option.size() == name.size() + 2 && option.starts_with("--") && option.substr(2) == name) {
        set(true);
        return true;
    }
    if (option.size() == name.size() + 5 && option.starts_with("--no-") &&
        option.substr(5) == name) {
        set(false);
        return true;
    }
    return false;
}

// Another spelling of a setting: `option` (without dashes) sets what `setting` sets.
struct SettingAlias {
    std::string_view option;
    std::string_view setting;
};

// The setting an option sets: its name without leading dashes or a `no-` prefix, through the
// program's aliases. `--no-vision` and `--vision` both set "vision".
[[nodiscard]] inline std::string option_setting(std::string_view option,
                                                std::span<const SettingAlias> aliases) {
    while (option.starts_with('-')) { option.remove_prefix(1); }
    if (option.starts_with("no-")) { option.remove_prefix(3); }
    for (const SettingAlias& alias : aliases) {
        if (alias.option == option) { return std::string(alias.setting); }
    }
    return std::string(option);
}

// One option of one layer: its arguments (the option and its values) and the setting it sets.
struct LayeredOption {
    std::size_t layer = 0;
    std::string setting;
    std::vector<std::string> arguments;
};

// The arguments [begin, end) of one option in the list a parser was given.
struct OptionSpan {
    std::size_t begin = 0;
    std::size_t end   = 0;
};

namespace detail {

// Keeps, for every setting, only the options of the latest layer that gives it.
inline void keep_latest_settings(std::vector<LayeredOption>& options) {
    // Copies of the settings: the loop below moves the options, and with them their strings.
    std::vector<std::pair<std::string, std::size_t>> latest;
    for (const LayeredOption& option : options) {
        auto found = std::find_if(latest.begin(), latest.end(),
                                  [&](const auto& entry) { return entry.first == option.setting; });
        if (found == latest.end()) {
            latest.emplace_back(option.setting, option.layer);
        } else {
            found->second = std::max(found->second, option.layer);
        }
    }
    std::vector<LayeredOption> kept;
    kept.reserve(options.size());
    for (LayeredOption& option : options) {
        const auto found = std::find_if(latest.begin(), latest.end(), [&](const auto& entry) {
            return entry.first == option.setting;
        });
        if (option.layer == found->second) { kept.push_back(std::move(option)); }
    }
    options = std::move(kept);
}

// Drops the options a conflict names from layers before the latest one among them. False when
// they all come from one layer, which leaves the conflict to the user.
inline bool drop_superseded(std::vector<LayeredOption>& options, const OptionConflict& conflict) {
    const auto named = [&](const LayeredOption& option) {
        return std::find(conflict.settings().begin(), conflict.settings().end(), option.setting) !=
               conflict.settings().end();
    };
    std::size_t latest = 0;
    bool any           = false;
    for (const LayeredOption& option : options) {
        if (named(option)) {
            latest = std::max(latest, option.layer);
            any    = true;
        }
    }
    if (!any) { return false; }
    const auto before = options.size();
    std::erase_if(options, [&](const LayeredOption& option) {
        return named(option) && option.layer < latest;
    });
    return options.size() != before;
}

[[nodiscard]] inline std::vector<std::string> flatten(const std::vector<LayeredOption>& options) {
    std::vector<std::string> arguments;
    for (const LayeredOption& option : options) {
        arguments.insert(arguments.end(), option.arguments.begin(), option.arguments.end());
    }
    return arguments;
}

} // namespace detail

// Parses layers of arguments (lowest precedence first) with `parse`, by the rules at the top of
// this file. `parse(arguments, spans)` is the program's parser over one list of options: with
// `spans` it only splits the list into options, recording each option's span, and validates
// nothing else; without, it parses and validates. `effective`, when given, receives the merged
// arguments the result was parsed from.
template <class Parse>
auto parse_layered(const std::vector<std::vector<std::string>>& layers,
                   std::span<const SettingAlias> aliases, Parse&& parse,
                   std::vector<std::string>* effective = nullptr) {
    std::vector<LayeredOption> options;
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        std::vector<OptionSpan> spans;
        (void)parse(layers[layer], &spans);
        for (const OptionSpan& span : spans) {
            const auto first = layers[layer].begin() + static_cast<std::ptrdiff_t>(span.begin);
            const auto last  = layers[layer].begin() + static_cast<std::ptrdiff_t>(span.end);
            options.push_back({.layer     = layer,
                               .setting   = option_setting(*first, aliases),
                               .arguments = std::vector<std::string>(first, last)});
        }
    }
    detail::keep_latest_settings(options);
    for (;;) {
        std::vector<std::string> arguments = detail::flatten(options);
        try {
            auto result = parse(arguments, nullptr);
            if (effective != nullptr) { *effective = std::move(arguments); }
            return result;
        } catch (const OptionConflict& conflict) {
            if (!detail::drop_superseded(options, conflict)) { throw; }
        }
    }
}

// Records the option spans of a parser loop over `arguments`: call `begin(i)` at the top of each
// iteration with the index of the option, and `finish()` after the loop.
class OptionSpanRecorder {
public:
    explicit OptionSpanRecorder(std::vector<OptionSpan>* spans) : spans_(spans) {}

    void begin(std::size_t index) {
        if (spans_ == nullptr) { return; }
        if (open_) { spans_->push_back({.begin = start_, .end = index}); }
        start_ = index;
        open_  = true;
    }

    void finish(std::size_t end) {
        if (spans_ != nullptr && open_) { spans_->push_back({.begin = start_, .end = end}); }
        open_ = false;
    }

    // Whether the parser only splits its arguments (see parse_layered).
    [[nodiscard]] bool splitting() const noexcept { return spans_ != nullptr; }

private:
    std::vector<OptionSpan>* spans_;
    std::size_t start_ = 0;
    bool open_         = false;
};

} // namespace ninfer::product
