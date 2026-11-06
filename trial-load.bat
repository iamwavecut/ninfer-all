@echo off
setlocal
rem ---------------------------------------------------------------------------------------------
rem fp8-dev trial workload: sends the test load (smoke / prefill / decode / mixed, 4 concurrent)
rem to the trial serve on port 10999 and writes <newest logs\trial-<stamp>>\load.csv.
rem
rem Parse-safe on purpose: no for /f command strings, no powershell. cmd on this host re-parses
rem for /f '...' regions in ways that break piped powershell one-liners (literal ^ reaches the
rem interpreter); the newest run dir is resolved by trial-load.py itself.
rem
rem prereq: the trial serve is up (trial-start.bat finished its port wait).
rem ---------------------------------------------------------------------------------------------

set "PORT=10999"

rem the serve must be listening
netstat -ano | findstr "LISTENING" | findstr /r ":%PORT% " >nul
if errorlevel 1 (
    echo Port %PORT% is not listening - the trial serve is not up, run trial-start.bat first
    pause
    exit /b 1
)

echo Trial workload -> port %PORT%
echo (smoke, then 4-concurrent prefill / decode / mixed phases; a few minutes total)
echo  Optional: phase[:workers] items (space-separated), e.g.  trial-load.bat prefill:1 decode:4
rem "" keeps python's run_dir slot empty (it self-resolves the newest trial dir);
rem without it the first phase would land in the run_dir slot.
python "%~dp0trial-load.py" %PORT% "" %1 %2 %3 %4 %5
set "RC=%ERRORLEVEL%"
echo.
if %RC%==0 (
    echo Workload finished OK. Stop the trial with trial-stop.bat, then restart production.
) else (
    echo Workload finished with status %RC% - check load.csv and serve.log.
)
pause
