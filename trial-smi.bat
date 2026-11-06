@echo off
setlocal
rem ---------------------------------------------------------------------------------------------
rem nvidia-smi sampler for the fp8-dev trial run.
rem
rem Usage: trial-smi.bat [interval_seconds] [run_dir]
rem   run_dir receives smi.csv (one row per interval) and smi-stop.flag (stop signal).
rem Started minimized by trial-start.bat; stopped by trial-stop.bat creating the flag file,
rem or closes itself after 4 hours as a safety net.
rem ---------------------------------------------------------------------------------------------

set "INTERVAL=%~1"
set "RUN=%~2"
if "%INTERVAL%"=="" set "INTERVAL=2"
if "%RUN%"=="" (
    echo usage: trial-smi.bat [interval_seconds] [run_dir]
    exit /b 1
)

> "%RUN%\smi.csv" echo time,util_gpu,util_mem,memory_used_mib,power_w,sm_mhz,mem_mhz,temp_c,hw_slowdown,sw_power_cap,sw_thermal_slowdown
set /a ELAPSED=0
:loop
if exist "%RUN%\smi-stop.flag" goto :done
for /f "delims=" %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-ddTHH:mm:ss"') do set "T=%%t"
rem No redirection inside the for/f string (this host's cmd re-parses it and rejects it; the
rem production stop.bat follows the same rule). The clocks_event_reasons fields are the throttle
rem bitfield this driver exposes ("Not Active" means no throttle). A failed sample yields no row.
for /f "tokens=1,2,3,4,5,6,7,8,9,10 delims=," %%a in ('nvidia-smi "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown" "--format=csv,noheader,nounits"') do (
    echo %T%,%%a,%%b,%%c,%%d,%%e,%%f,%%g,%%h,%%i,%%j >> "%RUN%\smi.csv"
)
set /a ELAPSED+=%INTERVAL%
if %ELAPSED% geq 14400 goto :done
rem powershell sleep instead of timeout: timeout.exe needs a console input handle and refuses to
rem run (and the loop would then spin) when the sampler is ever detached from one.
powershell -NoProfile -Command "Start-Sleep -Seconds %INTERVAL%"
goto :loop
:done
exit /b 0
