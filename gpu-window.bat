@echo off
setlocal EnableExtensions
REM ============================================================
REM gpu-window.bat - unattended M2.2 + M4 GPU-window suite
REM
REM Run from its deployment dir (F:\gsoft\gw). Phases:
REM   1  full test suite over the F: test exes, 600 s per-test timeout
REM   2  FP8 A8 plain-mma schedule sweep, five shapes, T=1..2048
REM   3  A16/A8 T-curve, five shapes
REM   4  perplexity A/B, full 1m corpus (A8 build vs A16 build, M4 gate)
REM
REM Prereq: NO ninfer-serve running. Stop the trial with
REM F:\gsoft\ninfer-fp8dev\trial-stop.bat and production with
REM D:\gsoft\ninfer\stop.bat first. The GPU must be idle.
REM
REM Usage:  gpu-window.bat          all phases (roughly 2-4 h)
REM         gpu-window.bat 2 3      only the listed phases (1-4)
REM
REM Logs:   logs\yyyy-MM-dd_HHmmss\ next to this script.
REM
REM Parse-safe on purpose: single-line ifs, gotos and subroutine calls
REM only; multi-line blocks contain no parenthesis characters at all.
REM Messages ASCII-only.
REM ============================================================
cd /d "%~dp0"

set "BENCH=%~dp0bench\ninfer_benches.exe"
set "PPL_A8=%~dp0ppl-a8\ninfer-perplexity.exe"
set "PPL_A16=%~dp0ppl-a16\ninfer-perplexity.exe"
set "NINFER_TEST_ARTIFACT=G:\AI_Models\Qwen3.8-27B-FP8-NInfer-v3\Qwen3.8-27B-FP8-v3-full.ninfer"
set "ARTIFACT=G:\AI_Models\Qwen3.8-27B-FP8-NInfer-v3\Qwen3.8-27B-FP8-v3-full.ninfer"
set "CORPUS=D:\msys64\home\z\src\dsh\t03\ninfer-all-fork\eval\corpora\perplexity-1m\manifest.json"
REM TSWEEP: 6/7/12 pin the small-T A8/A16 crossover (dflash2 verify is T=6); 384/768 pin the
REM mid-range schedule handoffs.
set "TSWEEP=1,2,4,5,6,7,8,12,16,17,20,22,24,25,32,48,64,96,128,192,256,384,512,768,1024,2048"
set "FAILURES=0"

REM ---- phase selection: no args = all phases; args pick a subset ----
set "RUN1="
set "RUN2="
set "RUN3="
set "RUN4="
if "%~1"=="" goto :setall
for %%a in (%*) do call :pickphase %%a
goto :havephases
:setall
set "RUN1=1"
set "RUN2=1"
set "RUN3=1"
set "RUN4=1"
:havephases

REM ---- prereqs ----
tasklist /fi "IMAGENAME eq ninfer-serve.exe" 2>nul | findstr /i "ninfer-serve.exe" >nul
if not errorlevel 1 goto :err_serve
netstat -ano | findstr "LISTENING" | findstr /r ":10999 " >nul
if not errorlevel 1 goto :err_trialport
netstat -ano | findstr "LISTENING" | findstr /r ":10111 " >nul
if not errorlevel 1 goto :err_prodport
if not exist "%BENCH%" goto :err_bench
if not exist "%PPL_A8%" goto :err_ppla8
if not exist "%PPL_A16%" goto :err_ppla16
if not exist "%~dp0tests\ninfer_tests.exe" goto :err_tests

REM ---- run stamp + log dir ----
set "STAMP="
for /f "delims=" %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "STAMP=%%t"
if "%STAMP%"=="" set "STAMP=manual"
set "GW=%~dp0logs\%STAMP%"
if not exist "%GW%" mkdir "%GW%"
echo === GPU window suite, logs: %GW%
nvidia-smi > "%GW%\0-gpu-start.txt" 2>&1

:phase1
if not defined RUN1 goto :phase2
echo.
echo === phase 1/4: full test suite over F: exes ===
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tests-runner.ps1" -LogDir "%GW%" > "%GW%\1-ctest.log" 2>&1
set "P1=%ERRORLEVEL%"
echo phase1 exit=%P1%  log: %GW%\1-ctest.log
if %P1%==0 goto :phase1ok
set /a FAILURES+=1
:phase1ok
goto :phase2

:phase2
if not defined RUN2 goto :phase3
echo.
echo === phase 2/4: fp8 a8 plain-mma schedule sweep ===
"%BENCH%" ninfer_fp8_a8_schedule_bench 14336x5120 --tokens %TSWEEP% > "%GW%\2-sweep-14336x5120.log" 2>&1
if errorlevel 1 (
    echo sweep 14336x5120 FAILED
    set /a FAILURES+=1
)
"%BENCH%" ninfer_fp8_a8_schedule_bench 16384x5120 --tokens %TSWEEP% > "%GW%\2-sweep-16384x5120.log" 2>&1
if errorlevel 1 (
    echo sweep 16384x5120 FAILED
    set /a FAILURES+=1
)
"%BENCH%" ninfer_fp8_a8_schedule_bench 34816x5120 --tokens %TSWEEP% > "%GW%\2-sweep-34816x5120.log" 2>&1
if errorlevel 1 (
    echo sweep 34816x5120 FAILED
    set /a FAILURES+=1
)
"%BENCH%" ninfer_fp8_a8_schedule_bench 5120x6144 --tokens %TSWEEP% > "%GW%\2-sweep-5120x6144.log" 2>&1
if errorlevel 1 (
    echo sweep 5120x6144 FAILED
    set /a FAILURES+=1
)
"%BENCH%" ninfer_fp8_a8_schedule_bench 5120x17408 --tokens %TSWEEP% > "%GW%\2-sweep-5120x17408.log" 2>&1
if errorlevel 1 (
    echo sweep 5120x17408 FAILED
    set /a FAILURES+=1
)
goto :phase3

:phase3
if not defined RUN3 goto :phase4
echo.
echo === phase 3/4: a16/a8 t-curve ===
"%BENCH%" ninfer_fp8_linear_add_bench --k 6144 --rows 5120 --policy a16 --t-sweep %TSWEEP% --csv-out "%GW%\3-a16-5120x6144.csv" > "%GW%\3-a16-5120x6144.log" 2>&1
if errorlevel 1 (
    echo tcurve 5120x6144 a16 FAILED
    set /a FAILURES+=1
)
"%BENCH%" ninfer_fp8_linear_add_bench --k 6144 --rows 5120 --policy a8 --t-sweep %TSWEEP% --csv-out "%GW%\3-a8-5120x6144.csv" > "%GW%\3-a8-5120x6144.log" 2>&1
if errorlevel 1 (
    echo tcurve 5120x6144 a8 FAILED
    set /a FAILURES+=1
)
"%BENCH%" ninfer_fp8_linear_add_bench --k 17408 --rows 5120 --policy a16 --t-sweep %TSWEEP% --csv-out "%GW%\3-a16-5120x17408.csv" > "%GW%\3-a16-5120x17408.log" 2>&1
if errorlevel 1 (
    echo tcurve 5120x17408 a16 FAILED
    set /a FAILURES+=1
)
"%BENCH%" ninfer_fp8_linear_add_bench --k 17408 --rows 5120 --policy a8 --t-sweep %TSWEEP% --csv-out "%GW%\3-a8-5120x17408.csv" > "%GW%\3-a8-5120x17408.log" 2>&1
if errorlevel 1 (
    echo tcurve 5120x17408 a8 FAILED
    set /a FAILURES+=1
)
"%BENCH%" ninfer_fp8_linear_add_bench --k 5120 --rows 14336 --policy a16 --t-sweep %TSWEEP% --csv-out "%GW%\3-a16-14336x5120.csv" > "%GW%\3-a16-14336x5120.log" 2>&1
if errorlevel 1 (
    echo tcurve 14336x5120 a16 FAILED
    set /a FAILURES+=1
)
"%BENCH%" ninfer_fp8_linear_add_bench --k 5120 --rows 14336 --policy a8 --t-sweep %TSWEEP% --csv-out "%GW%\3-a8-14336x5120.csv" > "%GW%\3-a8-14336x5120.log" 2>&1
if errorlevel 1 (
    echo tcurve 14336x5120 a8 FAILED
    set /a FAILURES+=1
)
"%BENCH%" ninfer_fp8_linear_add_bench --k 5120 --rows 16384 --policy a16 --t-sweep %TSWEEP% --csv-out "%GW%\3-a16-16384x5120.csv" > "%GW%\3-a16-16384x5120.log" 2>&1
if errorlevel 1 (
    echo tcurve 16384x5120 a16 FAILED
    set /a FAILURES+=1
)
"%BENCH%" ninfer_fp8_linear_add_bench --k 5120 --rows 16384 --policy a8 --t-sweep %TSWEEP% --csv-out "%GW%\3-a8-16384x5120.csv" > "%GW%\3-a8-16384x5120.log" 2>&1
if errorlevel 1 (
    echo tcurve 16384x5120 a8 FAILED
    set /a FAILURES+=1
)
"%BENCH%" ninfer_fp8_linear_add_bench --k 5120 --rows 34816 --policy a16 --t-sweep %TSWEEP% --csv-out "%GW%\3-a16-34816x5120.csv" > "%GW%\3-a16-34816x5120.log" 2>&1
if errorlevel 1 (
    echo tcurve 34816x5120 a16 FAILED
    set /a FAILURES+=1
)
"%BENCH%" ninfer_fp8_linear_add_bench --k 5120 --rows 34816 --policy a8 --t-sweep %TSWEEP% --csv-out "%GW%\3-a8-34816x5120.csv" > "%GW%\3-a8-34816x5120.log" 2>&1
if errorlevel 1 (
    echo tcurve 34816x5120 a8 FAILED
    set /a FAILURES+=1
)
goto :phase4

:phase4
if not defined RUN4 goto :summary
echo.
echo === phase 4/4: perplexity A/B, full 1m corpus ===
echo   a8 quick ...
"%PPL_A8%" "%ARTIFACT%" --corpus "%CORPUS%" --kv-dtype int8 --quick --output "%GW%\ppl-a8-quick" > "%GW%\4-ppl-a8-quick.log" 2>&1
echo   a16 quick ...
"%PPL_A16%" "%ARTIFACT%" --corpus "%CORPUS%" --kv-dtype int8 --quick --output "%GW%\ppl-a16-quick" > "%GW%\4-ppl-a16-quick.log" 2>&1
echo   a8 full ...
"%PPL_A8%" "%ARTIFACT%" --corpus "%CORPUS%" --kv-dtype int8 --output "%GW%\ppl-a8" > "%GW%\4-ppl-a8.log" 2>&1
if errorlevel 1 set /a FAILURES+=1
echo   a16 full ...
"%PPL_A16%" "%ARTIFACT%" --corpus "%CORPUS%" --kv-dtype int8 --output "%GW%\ppl-a16" > "%GW%\4-ppl-a16.log" 2>&1
if errorlevel 1 set /a FAILURES+=1
goto :summary

:summary
echo.
echo === GPU window suite summary ===
echo failures=%FAILURES%
echo logs: %GW%
dir /b "%GW%"
if %FAILURES%==0 goto :done
echo GPU_WINDOW_FAILED - inspect the logs in %GW%
goto :end
:done
echo GPU_WINDOW_DONE
:end
pause
exit /b 0

REM ---- error paths ----
:err_serve
echo [error] a ninfer-serve.exe process is still running.
echo         Stop the trial with F:\gsoft\ninfer-fp8dev\trial-stop.bat
echo         and production with D:\gsoft\ninfer\stop.bat first.
pause
exit /b 1
:err_trialport
echo [error] port 10999 is still listening.
pause
exit /b 1
:err_prodport
echo [error] port 10111 is still listening.
pause
exit /b 1
:err_bench
echo [error] bench\ninfer_benches.exe missing next to this script.
pause
exit /b 1
:err_ppla8
echo [error] ppl-a8\ninfer-perplexity.exe missing next to this script.
pause
exit /b 1
:err_ppla16
echo [error] ppl-a16\ninfer-perplexity.exe missing next to this script.
pause
exit /b 1
:err_tests
echo [error] tests\ninfer_tests.exe missing next to this script.
pause
exit /b 1

REM ---- subroutine: phase picker (no args) ----
:pickphase
if "%~1"=="1" set "RUN1=1"
if "%~1"=="2" set "RUN2=1"
if "%~1"=="3" set "RUN3=1"
if "%~1"=="4" set "RUN4=1"
goto :eof
