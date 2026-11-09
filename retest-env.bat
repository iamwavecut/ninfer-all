@echo off
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "VCPKG_ROOT=%~dp0vcpkg"
set "VCPKG_TARGET_TRIPLET=x64-windows"
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "http_proxy="
set "https_proxy="
set "ALL_PROXY="
set "all_proxy="
ctest --test-dir build-win11 --output-on-failure -R "ninfer_device_profile_test|ninfer_prompt_input_test|ninfer_serve_options_test|ninfer_qwen3_5_frontend_test|ninfer_expert_profile_test|ninfer_qwen3_5_loading_test|ninfer_model_registry_test|ninfer_expert_delivery_test|ninfer_request_log_test|ninfer_qwen3_5_graft_loader_test|ninfer_qwen3_5_steering_loader_test|ninfer_qwen4_exp_ngram_component_test|ninfer_qwen4_exp_expert_stream_test|ninfer_tma_shared_card_test" > retest.log 2>&1
echo RETEST_EXIT=%ERRORLEVEL%
