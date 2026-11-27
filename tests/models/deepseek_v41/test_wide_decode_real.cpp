// DeepSeek: a decode step of 9 to 16 rows on a real artifact (skipped without one), as
// tests/models/wide_decode_check.h checks it:
//   NINFER_DEEPSEEK_ARTIFACT  a .ninfer of the family
#include "artifact/reader.h"
#include "core/device.h"
#include "models/deepseek_v41/executor.h"
#include "models/deepseek_v41/model.h"
#include "../wide_decode_check.h"

#include <cstdlib>
#include <iostream>

int main() {
    using namespace ninfer;
    const char* artifact_path = std::getenv("NINFER_DEEPSEEK_ARTIFACT");
    if (artifact_path == nullptr) {
        std::cout << "SKIP DeepSeek wide decode: NINFER_DEEPSEEK_ARTIFACT unset\n";
        return 77;
    }
    DeviceContext device(0);
    device.bind_to_current_thread();
    const artifact::Reader reader(artifact_path);
    auto model = models::deepseek_v41::load_model(reader, {.artifact = artifact_path}, device);
    const auto domain = std::int32_t(model->resources().public_token_count);
    constexpr std::uint32_t kSequences = 16;
    models::deepseek_v41::ExecutorOptions options;
    options.max_context   = 512;
    options.prefill_chunk = 512;
    options.sequences     = kSequences;
    models::deepseek_v41::Executor executor(*model, device, options);
    executor.warm_up();
    const int failures = test::wide_decode_check(executor, kSequences, domain);
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " DeepSeek wide decode\n";
    return failures == 0 ? 0 : 1;
}
