// DeepSeek: a shared prompt pass on a real artifact (skipped without one), as
// tests/models/shared_prompt_check.h checks it, its experts read from the artifact's files, under
// the exact prefill tail, with the Engram table (each piece's rows continue its own sequence's
// history) and the DSpark drafter when the artifact holds one (each piece's last rows tap and seed
// its own sequence's rings):
//   NINFER_DEEPSEEK_ARTIFACT      a .ninfer of the family
//   NINFER_DEEPSEEK_ENGRAM_TABLE  the Engram table artifact, unless the model stores its rows
#include "artifact/reader.h"
#include "core/device.h"
#include "models/deepseek_v41/engram.h"
#include "models/deepseek_v41/executor.h"
#include "models/deepseek_v41/model.h"
#include "../shared_prompt_check.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>

int main() {
    using namespace ninfer;
    const char* artifact_path = std::getenv("NINFER_DEEPSEEK_ARTIFACT");
    if (artifact_path == nullptr) {
        std::cout << "SKIP DeepSeek shared prompt pass: NINFER_DEEPSEEK_ARTIFACT unset\n";
        return 77;
    }
    const char* table_path = std::getenv("NINFER_DEEPSEEK_ENGRAM_TABLE");
    DeviceContext device(0);
    device.bind_to_current_thread();
    const artifact::Reader reader(artifact_path);
    const bool dspark = models::deepseek_v41::has_dspark(reader);
    auto model        = models::deepseek_v41::load_model(
        reader, {.artifact = artifact_path, .experts = ExpertResidency::Disk, .dspark = dspark}, device);
    const auto domain           = std::int32_t(model->resources().public_token_count);
    const std::uint32_t history = model->config().sliding_window + 100;
    models::deepseek_v41::ExecutorOptions options;
    options.max_context   = history + 512;
    options.prefill_chunk = 512;
    options.sequences     = 4;
    // The Engine's default tail: these prompts' rows all run the decoder blocks under it.
    options.prefill_tail  = PrefillTail::Exact;
    options.engram_table  = models::deepseek_v41::engram_table_source(
                                reader, artifact_path, model->config(),
                                table_path != nullptr ? std::filesystem::path(table_path)
                                                      : std::filesystem::path())
                                .layout;
    models::deepseek_v41::Executor executor(*model, device, options);
    executor.warm_up();
    std::cout << "DSpark drafter " << (dspark ? "bound" : "absent") << '\n';
    const int failures =
        test::shared_prompt_check<models::deepseek_v41::Executor, models::deepseek_v41::PromptPiece>(
            executor, domain, history, options.prefill_chunk);
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " DeepSeek shared prompt pass\n";
    return failures == 0 ? 0 : 1;
}
