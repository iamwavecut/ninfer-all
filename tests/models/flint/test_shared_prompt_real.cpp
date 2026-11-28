// Flint: a shared prompt pass on a real artifact (skipped without one), as
// tests/models/shared_prompt_check.h checks it:
//   NINFER_FLINT_ARTIFACT  a .ninfer of the family
#include "artifact/reader.h"
#include "core/device.h"
#include "models/flint/executor.h"
#include "models/flint/model.h"
#include "../shared_prompt_check.h"

#include <cstdlib>
#include <iostream>

int main() {
    using namespace ninfer;
    const char* artifact_path = std::getenv("NINFER_FLINT_ARTIFACT");
    if (artifact_path == nullptr) {
        std::cout << "SKIP Flint shared prompt pass: NINFER_FLINT_ARTIFACT unset\n";
        return 77;
    }
    DeviceContext device(0);
    device.bind_to_current_thread();
    const artifact::Reader reader(artifact_path);
    auto model = models::flint::load_model(reader, {.artifact = artifact_path}, device);
    const auto domain            = std::int32_t(model->resources().public_token_count);
    const std::uint32_t history  = model->config().sliding_window + 100;
    models::flint::ExecutorOptions options;
    options.max_context   = history + 512;
    options.prefill_chunk = 512;
    options.sequences     = 4;
    models::flint::Executor executor(*model, device, options);
    executor.warm_up();
    const int failures = test::shared_prompt_check<models::flint::Executor, models::flint::PromptPiece>(
        executor, domain, history, options.prefill_chunk);
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " Flint shared prompt pass\n";
    return failures == 0 ? 0 : 1;
}
