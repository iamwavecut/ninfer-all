@echo off
setlocal
rem ---------------------------------------------------------------------------------------------
rem fp8-dev trial serve: the FORK build (sm_89, FP8 A8 route ON) with the production's flags,
rem on its own port and its own disk-KV path. Double-click it to run.
rem
rem Production (D:\gsoft\ninfer, port 10111) must be STOPPED first: this box runs one engine at a
rem time, and the model needs the ~46 GB VRAM the production engine holds.
rem
rem This window: starts the serve (minimized window "fp8dev-serve") and the nvidia-smi sampler
rem (minimized window "fp8dev-smi"), waits for port 10999, checks the activation-policy line, then
rem WAITS here. Run trial-load.bat from a second window for the test workload; stop with
rem trial-stop.bat. The serve's stdout/stderr go to logs\trial-<stamp>\serve.log.
rem ---------------------------------------------------------------------------------------------

set "SERVER_BIN=%~dp0ninfer-serve.exe"
rem The converted full artifact (same model as production)
set "MODEL_PATH=G:\AI_Models\Qwen3.8-27B-FP8-NInfer-v3\Qwen3.8-27B-FP8-v3-full.ninfer"
set "MODEL_NAME=qwen3.8-27b-fp8-trial"
set "KV_PATH=G:\.cache\ninfer-kv-fp8dev"
set "KV_GB=256"
set "PORT=10999"
set "LOGDIR=%~dp0logs"

rem ---- run dir: one folder per start, logs never mix ----
set "STAMP="
for /f "delims=" %%a in ('powershell -NoProfile -Command "[DateTime]::Now.ToString(\"yyyyMMdd-HHmmss\")"') do set "STAMP=%%a"
if not defined STAMP (
    echo [error] cannot get a timestamp from powershell
    pause
    exit /b 1
)
set "RUN=%LOGDIR%\trial-%STAMP%"
set "LOG=%RUN%\serve.log"

rem ---- KV pool budget (tokens): the production's, 4 lanes ----
rem long conversation 256K (prompt at the ceiling) + short conversation 32K (incl. output)
rem + 2 decisions at 8K each (state + question). 311296 tokens.
set "LONG_K=256"
set "SHORT_K=2"
set "DEC_K=2"
set /a KV_CAPACITY=(%LONG_K%+%SHORT_K%+2*%DEC_K%)*1024
set /a MAX_CONTEXT=%LONG_K%*1024
if %MAX_CONTEXT% GTR %KV_CAPACITY% set /a MAX_CONTEXT=%KV_CAPACITY%
set "PREFILL_CHUNK=32768"
set "MAX_CONCURRENCY=4"

set "CUDA_MODULE_LOADING=LAZY"

echo === fp8-dev trial serve [port %PORT%, FP8 A8 build] ===
echo Model: %MODEL_PATH%
echo "KV pool: %KV_CAPACITY% tokens | max-context: %MAX_CONTEXT% | concurrency: %MAX_CONCURRENCY%"
echo "Run dir: %RUN%"
echo =========================================================

if not exist "%SERVER_BIN%" (
  echo Missing %SERVER_BIN% - run installtest.bat first
  pause
  exit /b 1
)
if not exist "%MODEL_PATH%" (
  echo Missing model: %MODEL_PATH%
  pause
  exit /b 1
)
if not exist "%RUN%" mkdir "%RUN%"

rem ---- nvidia-smi sampler: 2 s cadence, stops when %RUN%\smi-stop.flag appears ----
del "%RUN%\smi-stop.flag" >nul 2>&1
start "fp8dev-smi" /min cmd /c ""%~dp0trial-smi.bat" 2 "%RUN%""

rem ---- the serve itself: production's flag set, own port and disk-KV path ----
rem Launched through serve-launch.py (a small stdlib wrapper): it runs the serve and mirrors
rem stdout+stderr to %LOG%. The naive `start ... cmd /c "serve ... > log 2>&1"` form is fragile
rem here: `^` continuations inside the quoted string break at odd quote parity, and `start`
rem re-tokenises the argument list. The launcher is deterministic, and the serve still owns its
rem console, so trial-stop.bat's console-ctrl stop (AttachConsole + CTRL_C on the port's PID)
rem works exactly like the production stop.
set "PY=C:\Users\z\AppData\Local\Programs\Python\Python314\python.exe"
if not exist "%PY%" set "PY=python"
start "fp8dev-serve" /min "%PY%" "%~dp0serve-launch.py" "%LOG%" "%SERVER_BIN%" "%MODEL_PATH%" --model-id %MODEL_NAME% --max-context %MAX_CONTEXT% --kv-capacity %KV_CAPACITY% --prefill-chunk %PREFILL_CHUNK% --prefill-cublas --max-concurrency %MAX_CONCURRENCY% --disk-kv-path "%KV_PATH%" --disk-kv-gib %KV_GB% --disk-kv-restore --disk-kv-directstorage --kv-dtype fp8 --fast-prefill-kernel --concurrent-prefill --wddm-evictable-budget --gdn-state-fp16 --host 127.0.0.1 --port %PORT% --vision --vision-residency overlay --default-reasoning-effort xhigh --auto-long-anchors --auto-prefix-grid --derive-session-keys --spec dflash2 --draft-tokens 5

rem ---- wait for the port (up to ~8 min: model load) ----
echo Waiting for port %PORT% (model load takes a few minutes)...
set /a TRIES=0
:wait_loop
netstat -ano | findstr "LISTENING" | findstr /r ":%PORT% " >nul && goto :port_up
set /a TRIES+=1
if %TRIES% geq 240 goto :port_fail
timeout /t 2 /nobreak >nul
goto :wait_loop

:port_fail
echo.
echo The port never opened. Last 40 serve log lines:
powershell -NoProfile -Command "if (Test-Path '%LOG%') { Get-Content '%LOG%' -Tail 40 }"
pause
exit /b 1

:port_up
echo.
echo Port %PORT% is up.
echo.
rem The policy line is printed during model materialization (before the port opens), but the
rem serve's C stdio block-buffers stdout when it is a file, so poll a little for the flush.
echo --- activation policy line from the serve log (the A8 route must be KEPT, not demoted): ---
set /a POL_TRIES=0
:policy_loop
findstr /c:"activation policy" "%LOG%" >nul && goto :policy_seen
set /a POL_TRIES+=1
if %POL_TRIES% geq 15 (
    echo not flushed to the log yet - it appears once the serve stdout buffer flushes
    goto :policy_done
)
timeout /t 2 /nobreak >nul
goto :policy_loop
:policy_seen
findstr /c:"activation policy" "%LOG%"
:policy_done
echo.
echo === Trial serve is READY. ===
echo Next:
echo   1. Double-click trial-load.bat (test workload; results in %RUN% )
echo   2. When done, double-click trial-stop.bat (graceful stop).
echo The serve runs in its own minimized window (fp8dev-serve); this window may be closed.
pause
