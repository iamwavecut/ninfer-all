#include "models/qwen3_5/frontend/inkling_output.h"

#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {
namespace {

std::int32_t single_token(const Tokenizer& tokenizer, std::string_view text) {
    const std::vector<int> ids = tokenizer.encode(text);
    return ids.size() == 1 ? ids.front() : -1;
}

// The length of the longest suffix of `text` that begins `marker` (shorter than the marker).
std::size_t partial_suffix(std::string_view text, std::string_view marker) {
    const std::size_t maximum = std::min(text.size(), marker.size() - 1);
    for (std::size_t size = maximum; size != 0; --size) {
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

using Json = nlohmann::ordered_json;

// A call's JSON for `tool`: its name, then its arguments, under the parameters' schema when the
// tool is strict and any object otherwise.
Json call_json(const ToolCallOutputContract::Tool& tool) {
    Json arguments = tool.strict ? Json::parse(strict_tool_parameters_schema(tool))
                                 : Json{{"type", "object"}};
    Json schema{{"type", "object"},
                {"properties",
                 Json{{"name", Json{{"const", tool.name}}}, {"args", std::move(arguments)}}},
                {"required", Json::array({"name", "args"})},
                {"additionalProperties", false}};
    return Json{{"type", "json_schema"}, {"json_schema", std::move(schema)}};
}

Json literal(std::string_view text) {
    return Json{{"type", "const_string"}, {"value", std::string(text)}};
}

// Any declared tool's call JSON.
Json any_call_json(const ToolCallOutputContract& contract) {
    if (contract.tools.empty()) { throw std::invalid_argument("no tools are declared"); }
    Json::array_t bodies;
    for (const auto& tool : contract.tools) { bodies.push_back(call_json(tool)); }
    return Json{{"type", "or"}, {"elements", std::move(bodies)}};
}

// Free text in which every JSON call block holds a declared tool's call, after the forced call's
// JSON and its end when the prompt opened one. The free text holds none of `excludes`.
Json calls_in_blocks(const ToolCallOutputContract& contract, Json::array_t excludes) {
    Json calls{{"type", "triggered_tags"},
               {"triggers", Json::array({kInklingCallOpen})},
               {"tags", Json::array({Json{{"type", "tag"},
                                          {"begin", kInklingCallOpen},
                                          {"content", any_call_json(contract)},
                                          {"end", kInklingEnd}}})},
               {"at_least_one", false},
               {"stop_after_first", false}};
    if (!excludes.empty()) { calls["excludes"] = std::move(excludes); }
    if (contract.forced_tool_name.empty()) { return calls; }
    const auto forced =
        std::find_if(contract.tools.begin(), contract.tools.end(),
                     [&](const auto& tool) { return tool.name == contract.forced_tool_name; });
    if (forced == contract.tools.end()) {
        throw std::invalid_argument("the forced tool is not declared");
    }
    return Json{{"type", "sequence"},
                {"elements", Json::array({call_json(*forced), literal(kInklingEnd),
                                          std::move(calls)})}};
}

} // namespace

std::string inkling_tool_call_text_format(const ToolCallOutputContract& contract) {
    return calls_in_blocks(contract, {}).dump();
}

std::string inkling_blocks_before_text_format(const ToolCallOutputContract& contract) {
    return calls_in_blocks(contract, {kInklingText}).dump();
}

std::string inkling_calls_turn_format(const ToolCallOutputContract& contract) {
    // A forced call is the turn's first call. Otherwise the blocks before the first call, which
    // the triggered tags' at_least_one would not admit, then that call and the blocks after it.
    if (!contract.forced_tool_name.empty()) {
        return calls_in_blocks(contract, {kInklingText}).dump();
    }
    return Json{{"type", "sequence"},
                {"elements",
                 Json::array({Json{{"type", "any_text"},
                                   {"excludes", Json::array({kInklingCallOpen, kInklingText})}},
                              literal(kInklingCallOpen), any_call_json(contract),
                              literal(kInklingEnd), calls_in_blocks(contract, {kInklingText})})}}
        .dump();
}

InklingMarkers InklingMarkers::from(const Tokenizer& tokenizer) {
    InklingMarkers out;
    out.header     = single_token(tokenizer, kInklingHeader);
    out.thinking   = single_token(tokenizer, kInklingThinking);
    out.text       = single_token(tokenizer, kInklingText);
    out.tool_error = single_token(tokenizer, kInklingToolError);
    out.tool_text  = single_token(tokenizer, kInklingToolText);
    out.call       = single_token(tokenizer, kInklingCallOpen);
    out.end        = single_token(tokenizer, kInklingEnd);
    return out;
}

bool InklingMarkers::complete() const noexcept {
    return header >= 0 && thinking >= 0 && text >= 0 && tool_error >= 0 && tool_text >= 0 &&
           call >= 0 && end >= 0;
}

bool InklingMarkers::content(std::int32_t token) const noexcept {
    return token == thinking || token == text || token == tool_error || token == tool_text ||
           token == call;
}

bool parse_inkling_call(std::string_view region, std::string_view forced, GeneratedToolCall& out) {
    const nlohmann::ordered_json call = nlohmann::ordered_json::parse(region, nullptr, false);
    if (call.is_discarded() || !call.is_object()) { return false; }
    const auto name = call.find("name");
    if (name == call.end() || !name->is_string() || name->get_ref<const std::string&>().empty()) {
        return false;
    }
    const auto arguments = call.find("args");
    if ((arguments != call.end() && !arguments->is_object()) ||
        call.size() != (arguments == call.end() ? 1U : 2U)) {
        return false;
    }
    if (!forced.empty() && name->get_ref<const std::string&>() != forced) { return false; }
    out = GeneratedToolCall{.name           = name->get<std::string>(),
                            .arguments_json = arguments == call.end() ? std::string("{}")
                                                                      : arguments->dump()};
    return true;
}

InklingToolCallDecoder::InklingToolCallDecoder(
    std::shared_ptr<const ToolCallOutputContract> contract)
    : contract_(std::move(contract)) {
    if (contract_ == nullptr || contract_->forced_tool_name.empty()) { return; }
    // The generation prompt ends with the call's opener, so generation resumes in its JSON.
    in_call_                 = true;
    forced_                  = true;
    diagnostics_.marker_seen = true;
}

std::string InklingToolCallDecoder::feed(std::string_view text) {
    std::string published;
    pending_.append(text);
    for (;;) {
        if (!in_call_) {
            const std::size_t open = pending_.find(kInklingCallOpen);
            if (open == std::string::npos) {
                const std::size_t hold = partial_suffix(pending_, kInklingCallOpen);
                published.append(pending_, 0, pending_.size() - hold);
                pending_.erase(0, pending_.size() - hold);
                return published;
            }
            published.append(pending_, 0, open);
            pending_.erase(0, open + kInklingCallOpen.size());
            in_call_                 = true;
            diagnostics_.marker_seen = true;
            region_.clear();
        }
        const std::size_t close = pending_.find(kInklingEnd);
        if (close == std::string::npos) {
            const std::size_t hold = partial_suffix(pending_, kInklingEnd);
            region_.append(pending_, 0, pending_.size() - hold);
            pending_.erase(0, pending_.size() - hold);
            return published;
        }
        region_.append(pending_, 0, close);
        pending_.erase(0, close + kInklingEnd.size());
        in_call_ = false;
        GeneratedToolCall call;
        if (parse_inkling_call(region_, forced_ ? contract_->forced_tool_name : "", call)) {
            calls_.push_back(std::move(call));
            ++diagnostics_.structured_call_count;
        } else {
            diagnostics_.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
            // Only what the model produced: a forced call's opener came from the prompt.
            if (!forced_) { published.append(kInklingCallOpen); }
            published.append(region_);
            published.append(kInklingEnd);
        }
        region_.clear();
        forced_ = false;
    }
}

InklingToolCallDecoder::Terminal InklingToolCallDecoder::finish() {
    Terminal out;
    if (in_call_) {
        region_.append(pending_);
        GeneratedToolCall call;
        if (parse_inkling_call(region_, forced_ ? contract_->forced_tool_name : "", call)) {
            calls_.push_back(std::move(call));
            ++diagnostics_.structured_call_count;
            // A forced turn that ends after its JSON is a complete call missing only its closing
            // marker; anything less complete does not parse and stays text.
            if (forced_) { diagnostics_.forced_call_closed = true; }
        } else {
            diagnostics_.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
            if (!forced_) { out.content.append(kInklingCallOpen); }
            out.content.append(region_);
        }
    } else {
        out.content = pending_;
    }
    pending_.clear();
    region_.clear();
    forced_         = false;
    in_call_        = false;
    out.tool_calls  = std::move(calls_);
    out.diagnostics = diagnostics_;
    return out;
}

} // namespace ninfer::models::qwen3_5::frontend
