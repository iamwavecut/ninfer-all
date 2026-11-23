#pragma once
#include "serve/request.h"
#include "serve/request_json.h"

#include <optional>

namespace ninfer::serve {
// Chat nests schema metadata under json_schema; Responses and Anthropic use a flat format.
StructuredOutputOptions parse_structured_output(const RequestJson& format, bool nested,
                                                const std::string& param);

// The constrained-decoding fields vLLM and llama.cpp add to OpenAI's request: `structured_outputs`
// (vLLM), the earlier `guided_json`, `guided_regex`, `guided_choice` and `guided_grammar` (vLLM),
// and `grammar`, a GBNF grammar (llama.cpp; empty means none). At most one of them may constrain a
// request, and not beside a response format already in `request`. Sets the request's
// structured_output and the field that carried it.
void parse_constrained_decoding_extensions(const RequestJson& body, GenerationRequest& request);

// A server started without --structured-output reserves no grammar masks. There, a request's
// response format and strict tools are refused, unless --unconstrained-response-format lets them go
// unenforced; a Chat Completions strict flag is advisory and always does. Banned words (bad_words)
// are refused there in any case. Returns the request with the dropped constraints, or nothing when
// the request runs as given.
std::optional<GenerationRequest> drop_unavailable_constraints(const GenerationRequest& request,
                                                              bool structured_output,
                                                              bool unconstrained_response_format);
} // namespace ninfer::serve
