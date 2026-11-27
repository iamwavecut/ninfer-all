#include "serve/openai_responses.h"

#include "serve/generation_service.h"
#include "serve/openai_common.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

constexpr char kReasoningSummaryPlaceholder[] = "Reasoning summary is not supported. (Ninfer: OpenAI Responses API)";

std::string response_status(ninfer::FinishReason reason) {
    switch (reason) {
    case ninfer::FinishReason::OutputLimit:
    case ninfer::FinishReason::ContextCapacity:
        return "incomplete";
    case ninfer::FinishReason::Cancelled:
        return "cancelled";
    case ninfer::FinishReason::None:
    case ninfer::FinishReason::StopToken:
    case ninfer::FinishReason::StopString:
        return "completed";
    }
    return "failed";
}

// Response.error.code is a closed enum on the wire. A failure whose code is outside it is reported
// as server_error, and its own code moves into the message.
Json response_error(const ApiError& error) {
    static constexpr std::string_view kWireCodes[] = {"server_error",
                                                      "rate_limit_exceeded",
                                                      "invalid_prompt",
                                                      "data_residency_mismatch",
                                                      "bio_policy",
                                                      "misalignment_policy_violation",
                                                      "vector_store_timeout",
                                                      "invalid_image",
                                                      "invalid_image_format",
                                                      "invalid_base64_image",
                                                      "invalid_image_url",
                                                      "image_too_large",
                                                      "image_too_small",
                                                      "image_parse_error",
                                                      "image_content_policy_violation",
                                                      "invalid_image_mode",
                                                      "image_file_too_large",
                                                      "unsupported_image_media_type",
                                                      "empty_image_file",
                                                      "failed_to_download_image",
                                                      "image_file_not_found"};
    if (std::find(std::begin(kWireCodes), std::end(kWireCodes), error.code) !=
        std::end(kWireCodes)) {
        return Json{{"code", error.code}, {"message", error.message}};
    }
    return Json{
        {"code", "server_error"},
        {"message", error.code.empty() ? error.message : error.message + " (" + error.code + ")"}};
}

struct ItemIds {
    std::string reasoning;
    std::string message;
    std::vector<std::string> function_calls;
    std::vector<std::string> call_ids;
};

void add_wire_function_identity(Json& object, const OpenAIResponsesCreateRequest& request,
                                std::string_view engine_name) {
    const auto position = request.tool_identities.find(std::string(engine_name));
    if (position == request.tool_identities.end()) {
        object["name"] = engine_name;
        return;
    }
    object["name"] = position->second.name;
    if (position->second.wire_namespace) { object["namespace"] = *position->second.wire_namespace; }
}

// Build the display-only reasoning summary requested by the client.
Json reasoning_summary(const OpenAIResponsesCreateRequest& request) {
    if (!request.prompt.reasoning_summary) { return Json::array(); }
    return Json::array({Json{{"type", "summary_text"}, {"text", kReasoningSummaryPlaceholder}}});
}

// Mirror raw reasoning into the opaque field requested by Harness. This is deliberately not
// encryption; replayed prompt and cache semantics continue to come only from reasoning_text.
void add_reasoning_encrypted_content(Json& item, const OpenAIResponsesCreateRequest& request,
                                     const std::string& reasoning) {
    if (request.include_reasoning_encrypted_content) {
        item["encrypted_content"] = reasoning;
    }
}

Json response_common(const std::string& id, std::int64_t created_at,
                     const OpenAIResponsesCreateRequest& request,
                     const OpenAIResponsesRuntimeValues& runtime) {
    const Json reasoning = {
        {"effort",
         request.prompt.generation.reasoning_effort
             ? Json(requested_reasoning_effort_name(*request.prompt.generation.reasoning_effort))
             : Json(nullptr)},
        {"summary", request.prompt.reasoning_summary ? Json(*request.prompt.reasoning_summary)
                                                     : Json(nullptr)}};
    return Json{
        {"id", id},
        {"object", "response"},
        {"created_at", created_at},
        {"access_programs", nullptr},
        {"background", false},
        {"instructions",
         request.prompt.instructions ? Json(*request.prompt.instructions) : Json(nullptr)},
        {"max_output_tokens", request.requested_max_output_tokens
                                  ? Json(*request.requested_max_output_tokens)
                                  : Json(nullptr)},
        {"max_tool_calls", request.max_tool_calls ? Json(*request.max_tool_calls) : Json(nullptr)},
        {"metadata", request.metadata},
        {"model", request.prompt.model},
        {"parallel_tool_calls", request.parallel_tool_calls},
        {"previous_response_id", request.prompt.previous_response_id
                                     ? Json(*request.prompt.previous_response_id)
                                     : Json(nullptr)},
        {"reasoning", reasoning},
        {"service_tier", "default"},
        {"store", request.store},
        {"temperature", runtime.temperature},
        {"text", Json{{"format", request.prompt.text_format}}},
        {"tool_choice", request.tool_choice},
        {"tools", request.tools},
        {"top_logprobs", request.prompt.generation.top_logprobs},
        {"top_p", runtime.top_p},
        {"truncation", "disabled"}};
}

// One output Item a stream published: a reasoning or a message Item with the text of its deltas, in
// output_index order. Reasoning the model writes after content gets an Item of its own after the
// message, and content after that a new message Item, so every Item holds what the model wrote
// there; a client that returns them gets the reasoning and the content back by concatenation.
struct StreamedItem {
    bool reasoning   = false;
    std::string id;
    int output_index = -1;
    std::string text;
    std::vector<ninfer::TokenLogprob> logprobs;
    std::string status;
};

Json reasoning_item_json(const std::string& id, const std::string& status,
                         const OpenAIResponsesCreateRequest& request, const std::string& text) {
    Json item = {{"id", id},
                 {"type", "reasoning"},
                 {"status", status},
                 {"summary", reasoning_summary(request)},
                 {"content", Json::array({Json{{"type", "reasoning_text"}, {"text", text}}})}};
    add_reasoning_encrypted_content(item, request, text);
    return item;
}

Json message_item_json(const std::string& id, const std::string& status, const std::string& text,
                       Json logprobs) {
    return Json{{"id", id},
                {"type", "message"},
                {"status", status},
                {"role", "assistant"},
                {"content", Json::array({Json{{"type", "output_text"},
                                              {"annotations", Json::array()},
                                              {"text", text},
                                              {"logprobs", std::move(logprobs)}}})}};
}

bool needs_message_item(const GenerationOutcome& outcome, const std::string& status) {
    if (!outcome.text.empty()) { return true; }
    if (!outcome.tool_calls.empty() || !outcome.reasoning.empty()) { return false; }
    return status == "completed";
}

BuiltOpenAIResponse build_response(const std::string& id, std::int64_t created_at,
                                   const OpenAIResponsesCreateRequest& request,
                                   const OpenAIResponsesRuntimeValues& runtime,
                                   const GenerationOutcome& outcome, ItemIds ids,
                                   const std::vector<StreamedItem>& streamed = {}) {
    BuiltOpenAIResponse built;
    const std::string status      = response_status(outcome.finish_reason);
    const std::string item_status = status == "completed" ? "completed" : "incomplete";

    if (!streamed.empty()) {
        for (const StreamedItem& item : streamed) {
            built.output_items.push_back(
                item.reasoning
                    ? reasoning_item_json(item.id, item.status, request, item.text)
                    : message_item_json(item.id, item.status, item.text,
                                        openai_token_logprobs_json(
                                            item.logprobs, request.prompt.generation.top_logprobs,
                                            true)));
        }
    } else {
        if (!outcome.reasoning.empty()) {
            if (ids.reasoning.empty()) { ids.reasoning = new_openai_response_item_id("rs"); }
            const char* reasoning_status = (!outcome.text.empty() || !outcome.tool_calls.empty())
                                               ? "completed"
                                               : item_status.c_str();
            built.output_items.push_back(
                reasoning_item_json(ids.reasoning, reasoning_status, request, outcome.reasoning));
        }
        if (needs_message_item(outcome, status)) {
            if (ids.message.empty()) { ids.message = new_openai_response_item_id("msg"); }
            built.output_items.push_back(message_item_json(
                ids.message, item_status, outcome.text,
                openai_token_logprobs_json(outcome.content_logprobs,
                                           request.prompt.generation.top_logprobs, true)));
        }
    }

    ids.function_calls.resize(outcome.tool_calls.size());
    ids.call_ids.resize(outcome.tool_calls.size());
    for (std::size_t index = 0; index < outcome.tool_calls.size(); ++index) {
        if (ids.function_calls[index].empty()) {
            ids.function_calls[index] = new_openai_response_item_id("fc");
        }
        if (ids.call_ids[index].empty()) {
            ids.call_ids[index] = new_openai_response_item_id("call");
        }
        const ninfer::GeneratedToolCall& call = outcome.tool_calls[index];
        Json item                             = {{"id", ids.function_calls[index]},
                                                 {"type", "function_call"},
                                                 {"status", "completed"},
                                                 {"call_id", ids.call_ids[index]},
                                                 {"arguments", call.arguments_json}};
        add_wire_function_identity(item, request, call.name);
        built.output_items.push_back(std::move(item));
    }

    if (!outcome.reasoning.empty() || !outcome.text.empty() || !outcome.tool_calls.empty() ||
        status == "completed") {
        ChatTurn history;
        history.role              = ChatRole::Assistant;
        history.reasoning_content = outcome.reasoning;
        history.tool_calls.reserve(outcome.tool_calls.size());
        for (std::size_t index = 0; index < outcome.tool_calls.size(); ++index) {
            const ninfer::GeneratedToolCall& call = outcome.tool_calls[index];
            history.tool_calls.push_back(ToolCall{.id             = ids.call_ids[index],
                                                  .name           = call.name,
                                                  .arguments_json = call.arguments_json});
        }
        if (!outcome.text.empty()) {
            ContentPart part;
            part.kind     = ContentKind::Text;
            part.type_raw = "output_text";
            part.text     = outcome.text;
            history.content.push_back(std::move(part));
        }
        built.output_history.push_back(std::move(history));
    }

    Json response            = response_common(id, created_at, request, runtime);
    response["status"]       = status;
    response["completed_at"] = status == "completed" ? Json(unix_time_now()) : Json(nullptr);
    response["error"]        = nullptr;
    response["output"]       = built.output_items;
    response["incomplete_details"] =
        status == "incomplete" ? Json{{"reason", "max_output_tokens"}} : Json(nullptr);
    const int observed_cached = std::max(runtime.cached_input_tokens,
                                         static_cast<int>(outcome.metrics.prefix_cache_hit_tokens));
    const int cached_tokens   = std::clamp(observed_cached, 0, outcome.prompt_tokens);
    response["usage"]         = Json{
        {"input_tokens", outcome.prompt_tokens},
        {"input_tokens_details", Json{{"cached_tokens", cached_tokens}, {"cache_write_tokens", 0}}},
        {"output_tokens", outcome.completion_tokens},
        {"output_tokens_details", Json{{"reasoning_tokens", outcome.reasoning_tokens}}},
        {"total_tokens", outcome.prompt_tokens + outcome.completion_tokens}};
    built.body = std::move(response);
    return built;
}

std::string sse(const Json& event) {
    return "event: " + event.at("type").get<std::string>() + "\n" + "data: " + event.dump() +
           "\n\n";
}

Json in_progress_response(const std::string& id, std::int64_t created_at,
                          const OpenAIResponsesCreateRequest& request,
                          const OpenAIResponsesRuntimeValues& runtime) {
    Json response                  = response_common(id, created_at, request, runtime);
    response["status"]             = "in_progress";
    response["completed_at"]       = nullptr;
    response["error"]              = nullptr;
    response["incomplete_details"] = nullptr;
    response["output"]             = Json::array();
    response["usage"]              = nullptr;
    return response;
}

} // namespace

BuiltOpenAIResponse make_openai_response_object(const std::string& id, std::int64_t created_at,
                                                const OpenAIResponsesCreateRequest& request,
                                                const OpenAIResponsesRuntimeValues& runtime,
                                                const GenerationOutcome& outcome) {
    return build_response(id, created_at, request, runtime, outcome, {});
}

std::string make_openai_response_input_tokens_body(int input_tokens) {
    return Json{{"object", "response.input_tokens"}, {"input_tokens", input_tokens}}.dump();
}

class OpenAIResponsesEventStream::Impl {
public:
    Impl(std::string response_id, std::int64_t created_at_, OpenAIResponsesCreateRequest request_,
         OpenAIResponsesRuntimeValues runtime_)
        : id(std::move(response_id)), created_at(created_at_), request(std::move(request_)),
          runtime(runtime_) {}

    Json event(std::string type, Json fields = Json::object()) {
        fields["type"]            = std::move(type);
        fields["sequence_number"] = sequence++;
        return fields;
    }

    // The Item receiving deltas, if one is open: Items open and close one after another.
    StreamedItem* open_item() { return open < 0 ? nullptr : &items[static_cast<std::size_t>(open)]; }

    std::vector<std::string> open_reasoning() {
        StreamedItem item;
        item.reasoning                  = true;
        item.id                         = new_openai_response_item_id("rs");
        item.output_index               = next_output_index++;
        const Json summary              = reasoning_summary(request);
        const Json added                = {{"id", item.id},
                                           {"type", "reasoning"},
                                           {"status", "in_progress"},
                                           {"summary", summary},
                                           {"content", Json::array()}};
        const Json part                 = {{"type", "reasoning_text"}, {"text", ""}};
        std::vector<std::string> events = {
            sse(event("response.output_item.added",
                      Json{{"output_index", item.output_index}, {"item", added}}))};
        if (!summary.empty()) {
            const Json added_summary_part = {{"type", "summary_text"}, {"text", ""}};
            const Json& done_summary_part = summary.at(0);
            events.push_back(sse(event("response.reasoning_summary_part.added",
                                       Json{{"item_id", item.id},
                                            {"output_index", item.output_index},
                                            {"summary_index", 0},
                                            {"part", added_summary_part}})));
            events.push_back(sse(event("response.reasoning_summary_text.delta",
                                       Json{{"item_id", item.id},
                                            {"output_index", item.output_index},
                                            {"summary_index", 0},
                                            {"delta", kReasoningSummaryPlaceholder}})));
            events.push_back(sse(event("response.reasoning_summary_text.done",
                                       Json{{"item_id", item.id},
                                            {"output_index", item.output_index},
                                            {"summary_index", 0},
                                            {"text", kReasoningSummaryPlaceholder}})));
            events.push_back(sse(event("response.reasoning_summary_part.done",
                                       Json{{"item_id", item.id},
                                            {"output_index", item.output_index},
                                            {"summary_index", 0},
                                            {"part", done_summary_part}})));
        }
        events.push_back(
            sse(event("response.content_part.added", Json{{"item_id", item.id},
                                                          {"output_index", item.output_index},
                                                          {"content_index", 0},
                                                          {"part", part}})));
        open = static_cast<int>(items.size());
        items.push_back(std::move(item));
        return events;
    }

    std::vector<std::string> open_message() {
        StreamedItem item;
        item.id           = new_openai_response_item_id("msg");
        item.output_index = next_output_index++;
        const Json added  = {{"id", item.id},
                             {"type", "message"},
                             {"status", "in_progress"},
                             {"role", "assistant"},
                             {"content", Json::array()}};
        const Json part   = {{"type", "output_text"},
                             {"annotations", Json::array()},
                             {"text", ""},
                             {"logprobs", Json::array()}};
        std::vector<std::string> events = {
            sse(event("response.output_item.added",
                      Json{{"output_index", item.output_index}, {"item", added}})),
            sse(event("response.content_part.added", Json{{"item_id", item.id},
                                                          {"output_index", item.output_index},
                                                          {"content_index", 0},
                                                          {"part", part}}))};
        open = static_cast<int>(items.size());
        items.push_back(std::move(item));
        return events;
    }

    std::vector<std::string> close_open(const std::string& item_status) {
        StreamedItem* item = open_item();
        if (item == nullptr) { return {}; }
        open         = -1;
        item->status = item_status;
        if (item->reasoning) {
            const Json part = {{"type", "reasoning_text"}, {"text", item->text}};
            return {sse(event("response.reasoning_text.done", Json{{"item_id", item->id},
                                                                   {"output_index", item->output_index},
                                                                   {"content_index", 0},
                                                                   {"text", item->text}})),
                    sse(event("response.content_part.done", Json{{"item_id", item->id},
                                                                 {"output_index", item->output_index},
                                                                 {"content_index", 0},
                                                                 {"part", part}})),
                    sse(event("response.output_item.done",
                              Json{{"output_index", item->output_index},
                                   {"item", reasoning_item_json(item->id, item_status, request,
                                                                item->text)}}))};
        }
        // Streamed events carry the event shape of a record, without bytes; the response object
        // carries them.
        const Json aggregate = openai_token_logprobs_json(
            item->logprobs, request.prompt.generation.top_logprobs, false);
        const Json part = {{"type", "output_text"},
                           {"annotations", Json::array()},
                           {"text", item->text},
                           {"logprobs", aggregate}};
        return {sse(event("response.output_text.done", Json{{"item_id", item->id},
                                                            {"output_index", item->output_index},
                                                            {"content_index", 0},
                                                            {"text", item->text},
                                                            {"logprobs", aggregate}})),
                sse(event("response.content_part.done", Json{{"item_id", item->id},
                                                             {"output_index", item->output_index},
                                                             {"content_index", 0},
                                                             {"part", part}})),
                sse(event("response.output_item.done",
                          Json{{"output_index", item->output_index},
                               {"item", message_item_json(item->id, item_status, item->text,
                                                          aggregate)}}))};
    }

    // Reasoning goes to the open reasoning Item; after content it closes the message Item and
    // opens a reasoning Item of its own.
    std::vector<std::string> append_reasoning(const std::string& text) {
        std::vector<std::string> events;
        if (open_item() != nullptr && !open_item()->reasoning) { events = close_open("completed"); }
        if (open_item() == nullptr) {
            std::vector<std::string> added = open_reasoning();
            events.insert(events.end(), std::make_move_iterator(added.begin()),
                          std::make_move_iterator(added.end()));
        }
        StreamedItem& item = *open_item();
        item.text += text;
        reasoning_text += text;
        events.push_back(sse(event("response.reasoning_text.delta",
                                   Json{{"item_id", item.id},
                                        {"output_index", item.output_index},
                                        {"content_index", 0},
                                        {"delta", text}})));
        return events;
    }

    std::vector<std::string> append_content(const std::string& text,
                                            std::span<const ninfer::TokenLogprob> logprobs) {
        std::vector<std::string> events;
        if (open_item() != nullptr && open_item()->reasoning) { events = close_open("completed"); }
        if (open_item() == nullptr) {
            std::vector<std::string> added = open_message();
            events.insert(events.end(), std::make_move_iterator(added.begin()),
                          std::make_move_iterator(added.end()));
        }
        StreamedItem& item = *open_item();
        item.text += text;
        item.logprobs.insert(item.logprobs.end(), logprobs.begin(), logprobs.end());
        content_text += text;
        events.push_back(sse(event(
            "response.output_text.delta",
            Json{{"item_id", item.id},
                 {"output_index", item.output_index},
                 {"content_index", 0},
                 {"delta", text},
                 {"logprobs", openai_token_logprobs_json(
                                  logprobs, request.prompt.generation.top_logprobs, false)}})));
        return events;
    }

    [[nodiscard]] bool has_message() const {
        return std::any_of(items.begin(), items.end(),
                           [](const StreamedItem& item) { return !item.reasoning; });
    }

    std::string id;
    std::int64_t created_at = 0;
    OpenAIResponsesCreateRequest request;
    OpenAIResponsesRuntimeValues runtime;
    std::uint64_t sequence = 0;
    int next_output_index  = 0;
    int open               = -1;
    bool started           = false;
    bool finish_built      = false;
    bool terminal_emitted  = false;
    // The concatenations of the streamed reasoning and content: the outcome's prefixes.
    std::string reasoning_text;
    std::string content_text;
    std::vector<StreamedItem> items;
    ItemIds ids;
};

OpenAIResponsesEventStream::OpenAIResponsesEventStream(std::string response_id,
                                                       std::int64_t created_at,
                                                       OpenAIResponsesCreateRequest request,
                                                       OpenAIResponsesRuntimeValues runtime)
    : impl_(std::make_unique<Impl>(std::move(response_id), created_at, std::move(request),
                                   runtime)) {}

OpenAIResponsesEventStream::~OpenAIResponsesEventStream() = default;
OpenAIResponsesEventStream::OpenAIResponsesEventStream(OpenAIResponsesEventStream&&) noexcept =
    default;
OpenAIResponsesEventStream&
OpenAIResponsesEventStream::operator=(OpenAIResponsesEventStream&&) noexcept = default;

std::vector<std::string> OpenAIResponsesEventStream::start() {
    if (impl_->started) { throw std::logic_error("Responses event stream already started"); }
    impl_->started = true;
    const Json response =
        in_progress_response(impl_->id, impl_->created_at, impl_->request, impl_->runtime);
    return {sse(impl_->event("response.created", Json{{"response", response}})),
            sse(impl_->event("response.in_progress", Json{{"response", response}}))};
}

std::vector<std::string> OpenAIResponsesEventStream::reasoning_delta(const std::string& text) {
    if (!impl_->started || impl_->finish_built) {
        throw std::logic_error("invalid reasoning delta event state");
    }
    if (text.empty()) { return {}; }
    return impl_->append_reasoning(text);
}

std::vector<std::string>
OpenAIResponsesEventStream::content_delta(const std::string& text,
                                          std::span<const ninfer::TokenLogprob> logprobs) {
    if (!impl_->started || impl_->finish_built) {
        throw std::logic_error("invalid content delta event state");
    }
    if (text.empty() && logprobs.empty()) { return {}; }
    return impl_->append_content(text, logprobs);
}

OpenAIResponsesStreamFinish OpenAIResponsesEventStream::finish(const GenerationOutcome& outcome) {
    if (!impl_->started || impl_->finish_built) {
        throw std::logic_error("invalid Responses stream finish state");
    }
    impl_->finish_built = true;
    OpenAIResponsesStreamFinish finished;
    const std::string status = response_status(outcome.finish_reason);
    const char* item_status  = status == "completed" ? "completed" : "incomplete";

    auto append = [&](std::vector<std::string> events) {
        finished.events_before_terminal.insert(finished.events_before_terminal.end(),
                                               std::make_move_iterator(events.begin()),
                                               std::make_move_iterator(events.end()));
    };
    if (!outcome.reasoning.starts_with(impl_->reasoning_text)) {
        throw std::logic_error("streamed reasoning does not match terminal reasoning");
    }
    if (!outcome.text.starts_with(impl_->content_text)) {
        throw std::logic_error("streamed content does not match terminal content");
    }
    const std::string reasoning_suffix = outcome.reasoning.substr(impl_->reasoning_text.size());
    const std::string content_suffix   = outcome.text.substr(impl_->content_text.size());
    // The open Item's tail first, so text the output held back joins the Item it belongs to.
    const bool content_first = impl_->open_item() != nullptr && !impl_->open_item()->reasoning;
    if (content_first && !content_suffix.empty()) {
        append(impl_->append_content(content_suffix, {}));
    }
    if (!reasoning_suffix.empty()) { append(impl_->append_reasoning(reasoning_suffix)); }
    if (!content_first && !content_suffix.empty()) {
        append(impl_->append_content(content_suffix, {}));
    }
    if (needs_message_item(outcome, status) && !impl_->has_message()) {
        append(impl_->close_open("completed"));
        append(impl_->open_message());
    }
    if (const StreamedItem* item = impl_->open_item()) {
        const char* reasoning_status =
            (!outcome.text.empty() || !outcome.tool_calls.empty()) ? "completed" : item_status;
        append(impl_->close_open(item->reasoning ? reasoning_status : item_status));
    }

    impl_->ids.function_calls.reserve(outcome.tool_calls.size());
    impl_->ids.call_ids.reserve(outcome.tool_calls.size());
    for (const ninfer::GeneratedToolCall& call : outcome.tool_calls) {
        const std::string item_id = new_openai_response_item_id("fc");
        const std::string call_id = new_openai_response_item_id("call");
        impl_->ids.function_calls.push_back(item_id);
        impl_->ids.call_ids.push_back(call_id);
        const int output_index = impl_->next_output_index++;
        Json added_item        = {{"id", item_id},
                                  {"type", "function_call"},
                                  {"status", "in_progress"},
                                  {"call_id", call_id},
                                  {"arguments", ""}};
        add_wire_function_identity(added_item, impl_->request, call.name);
        finished.events_before_terminal.push_back(
            sse(impl_->event("response.output_item.added",
                             Json{{"output_index", output_index}, {"item", added_item}})));
        if (!call.arguments_json.empty()) {
            finished.events_before_terminal.push_back(sse(impl_->event(
                "response.function_call_arguments.delta", Json{{"item_id", item_id},
                                                               {"output_index", output_index},
                                                               {"delta", call.arguments_json}})));
        }
        Json arguments_done = {{"item_id", item_id},
                               {"output_index", output_index},
                               {"arguments", call.arguments_json}};
        add_wire_function_identity(arguments_done, impl_->request, call.name);
        finished.events_before_terminal.push_back(
            sse(impl_->event("response.function_call_arguments.done", std::move(arguments_done))));
        Json done_item = {{"id", item_id},
                          {"type", "function_call"},
                          {"status", "completed"},
                          {"call_id", call_id},
                          {"arguments", call.arguments_json}};
        add_wire_function_identity(done_item, impl_->request, call.name);
        finished.events_before_terminal.push_back(
            sse(impl_->event("response.output_item.done",
                             Json{{"output_index", output_index}, {"item", done_item}})));
    }

    finished.response = build_response(impl_->id, impl_->created_at, impl_->request, impl_->runtime,
                                       outcome, impl_->ids, impl_->items);
    return finished;
}

std::string OpenAIResponsesEventStream::terminal(const BuiltOpenAIResponse& response) {
    if (!impl_->finish_built || impl_->terminal_emitted) {
        throw std::logic_error("invalid Responses terminal event state");
    }
    impl_->terminal_emitted  = true;
    const std::string status = response.body.at("status").get<std::string>();
    const std::string type   = status == "completed"    ? "response.completed"
                               : status == "incomplete" ? "response.incomplete"
                                                        : "response.failed";
    return sse(impl_->event(type, Json{{"response", response.body}}));
}

std::string OpenAIResponsesEventStream::failed(const ApiError& error) {
    if (!impl_->started || impl_->terminal_emitted) {
        throw std::logic_error("invalid Responses failed event state");
    }
    impl_->terminal_emitted = true;
    Json response =
        in_progress_response(impl_->id, impl_->created_at, impl_->request, impl_->runtime);
    response["status"]       = "failed";
    response["completed_at"] = nullptr;
    response["error"]        = response_error(error);
    return sse(impl_->event("response.failed", Json{{"response", response}}));
}

} // namespace ninfer::serve
