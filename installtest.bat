@echo off
setlocal EnableExtensions

rem ============================================================
rem installtest.bat - fp8-dev trial deployment (messages ASCII-only)
rem
rem Deploys the FORK build (build-win11\apps) to F:\gsoft\ninfer-fp8dev,
rem beside the production deployment D:\gsoft\ninfer, and copies the trial
rem scripts. It never touches the production directory: no rename, no
rem backup of D:\gsoft\ninfer, no copy into it.
rem
rem Safe to re-run: robocopy /IS overwrites the deployed binaries in place.
rem ============================================================

rem switch to this script's own directory so relative paths are reliable
cd /d "%~dp0"

set "DEST=F:\gsoft\ninfer-fp8dev"
set "BUILD=build-win11\apps"
set "RC=0"
set "RB1LOG=%TEMP%\ninfer_installtest_rb1.log"
set "RB2LOG=%TEMP%\ninfer_installtest_rb2.log"

rem ---- 0. the trial serve itself must not be running (its exe would be in use).
rem The production engine (also named ninfer-serve.exe) may run: it is never touched.
for /f "delims=" %%p in ('powershell -NoProfile -Command "Get-Process ninfer-serve -ErrorAction SilentlyContinue | Where-Object { $_.Path -like 'F:\gsoft\*' } | ForEach-Object { [string]$_.Id }"') do (
    echo [error] the trial serve is running, PID %%p; run trial-stop.bat first
    pause & exit /b 1
)

rem ---- 1. target directory (fresh, or from a previous trial deployment) ----
if not exist "F:\gsoft" mkdir "F:\gsoft"
rem mkdir is a single line on purpose: the nested if-not-mkdir form is not
rem valid cmd syntax, and cmd pre-parses a whole block, so an invalid line
rem inside it kills the script even when the outer condition is false.
if not exist "%DEST%" mkdir "%DEST%"
if not exist "%DEST%" (
    echo [error] cannot create %DEST%
    pause & exit /b 1
)
echo [1/3] target dir ok: %DEST%

rem ---- 2. sync the serve deployment, same file list as install.bat ----
if not exist "%BUILD%"  goto :err_build
if not exist "%BUILD%\ninfer-serve.exe" (
    echo [error] ninfer-serve.exe missing from %BUILD%; build the fork first
    pause & exit /b 1
)
robocopy "%BUILD%" "%DEST%" ninfer-serve.* /IS /NFL /NDL /NJH /NJS /NP > "%RB1LOG%" 2>&1
set /a RC=%errorlevel%
if %RC% geq 8          goto :err_robocopy1
echo [2/3] ninfer-serve.exe + .pdb deployed, robocopy code = %RC%
robocopy "%BUILD%" "%DEST%" *.dll /IS /NFL /NDL /NJH /NJS /NP > "%RB2LOG%" 2>&1
set /a RC=%errorlevel%
if %RC% geq 8          goto :err_robocopy2
echo [2/3] all .dll deployed, robocopy code = %RC%

rem ---- 3. the trial scripts ----
set "COPYERR="
for %%f in (trial-start.bat trial-stop.bat trial-smi.bat trial-load.bat trial-load.py serve-launch.py) do (
    if not exist "%%f" (
        echo [error] missing %%f next to this script
        pause & exit /b 1
    )
    copy /y "%%f" "%DEST%\" >nul 2>&1 || set "COPYERR=%%f"
)
if defined COPYERR goto :err_copy
echo [3/3] trial scripts deployed

echo.
echo DONE. trial deployment ready: %DEST%
echo.
echo Next steps:
echo   1. Stop the production engine (D:\gsoft\ninfer\stop.bat) to free the GPU.
echo   2. Double-click trial-start.bat   (serve on port 10999, logs under logs\).
echo   3. Double-click trial-load.bat    (test workload against the trial serve).
echo   4. Double-click trial-stop.bat    (graceful stop, disk KV flushed).
echo   5. Restart production (D:\gsoft\ninfer\full-start.bat).
pause
exit /b 0

:err_build
echo [error] build dir not found: %BUILD%
pause & exit /b 1
:err_robocopy1
echo [error] ninfer-serve exe/pdb sync failed, robocopy code=%RC% (>=8 means error)
if exist "%RB1LOG%" type "%RB1LOG%"
pause & exit /b 1
:err_robocopy2
echo [error] .dll sync failed, robocopy code=%RC% (>=8 means error)
if exist "%RB2LOG%" type "%RB2LOG%"
pause & exit /b 1
:err_copy
echo [error] copy of %COPYERR% failed, check that %DEST% is writable
pause & exit /b 1
