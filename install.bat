@echo off
setlocal EnableExtensions

rem ============================================================
rem hardened install.bat  (messages are ASCII-only, encoding-proof)
rem ============================================================

rem switch to this script's own directory so relative paths are reliable
cd /d "%~dp0"

set "SRC=D:\gsoft\ninfer"
set "BUILD=build-win11\apps"
set "STAMP="
set "RC=0"
set "MOVEERR=%TEMP%\ninfer_install_move.err"
set "RB1LOG=%TEMP%\ninfer_install_rb1.log"
set "RB2LOG=%TEMP%\ninfer_install_rb2.log"

rem ---- 1. timestamp (PowerShell, locale independent) ----
for /f "delims=" %%a in ('powershell -NoProfile -Command "[DateTime]::Now.ToString(\"yyyyMMddTHHmm\")"') do set "STAMP=%%a"
if not defined STAMP goto :err_time
set "BACKUP=%SRC%_%STAMP%"
echo [1/4] timestamp = %STAMP%

rem ---- 2. rename old dir to backup ----
rem a rename only fails when the dir is in use (engine still running, or an
rem Explorer/terminal window parked inside it), the backup name exists, or a
rem permission is missing. Check the exit code AND both post-conditions, and on
rem failure print the real OS error before exiting non-zero.
if not exist "%SRC%"    goto :err_src
if exist "%BACKUP%"     goto :err_backup_exists
set "OSERR="
move /y "%SRC%" "%BACKUP%" > "%MOVEERR%" 2>&1
set /a RC=%errorlevel%
if not "%RC%"=="0"      goto :err_move
if not exist "%BACKUP%" goto :err_backup_missing
if exist "%SRC%"        goto :err_src_still
echo [2/4] rename ok: %SRC% -^> %BACKUP%

rem ---- 3. sync the serve deployment, and nothing else ----
rem copy by explicit file name instead of mirroring the whole tree, so the CMakeFiles/
rem directory, the other executables and cmake_install.cmake are never touched:
rem ninfer-serve.* (the one .exe plus its .pdb for crash stacks) and every .dll.
if not exist "%BUILD%"  goto :err_build
robocopy "%BUILD%" "%SRC%" ninfer-serve.* /IS /NFL /NDL /NJH /NJS /NP > "%RB1LOG%" 2>&1
set /a RC=%errorlevel%
if %RC% geq 8          goto :err_robocopy1
echo [3/4] ninfer-serve.exe + .pdb copied, robocopy code = %RC%
robocopy "%BUILD%" "%SRC%" *.dll /IS /NFL /NDL /NJH /NJS /NP > "%RB2LOG%" 2>&1
set /a RC=%errorlevel%
if %RC% geq 8          goto :err_robocopy2
echo [3/4] all .dll copied, robocopy code = %RC%

rem ---- 4. restore .bat scripts ----
if exist "%BACKUP%\*.bat" (
    copy /y "%BACKUP%\*.bat" "%SRC%\" >nul 2>&1
    if errorlevel 1 goto :err_copy
    echo [4/4] .BAT restored from backup
) else (
    echo [warn] no .bat in backup, skipped
    echo [hint] if start scripts live in build-win11\apps they were copied in step 3
)

echo.
echo DONE. backup dir: %BACKUP%
pause
exit /b 0

:err_time
echo [error] cannot create timestamp
pause & exit /b 1
:err_src
echo [error] source dir not found: %SRC%
pause & exit /b 1
:err_backup_exists
echo [error] backup dir already exists: %BACKUP%
pause & exit /b 1
:err_move
echo [error] rename failed, errorlevel=%RC%
set "OSERR="
if exist "%MOVEERR%" set /p OSERR= < "%MOVEERR%"
echo          OS said: %OSERR%
echo          likely: dir in use - stop the engine and close any Explorer/terminal
echo          window whose current folder is %SRC% - or no permission
pause & exit /b 1
:err_backup_missing
echo [error] move reported ok but backup missing: %BACKUP%
pause & exit /b 1
:err_src_still
echo [error] after move the source still exists: %SRC%
pause & exit /b 1
:err_build
echo [error] build dir not found: %BUILD%
echo          backup kept at: %BACKUP%
pause & exit /b 1
:err_robocopy1
echo [error] ninfer-serve exe/pdb sync failed, robocopy code=%RC% (>=8 means error)
if exist "%RB1LOG%" type "%RB1LOG%"
echo          backup kept at: %BACKUP%
pause & exit /b 1
:err_robocopy2
echo [error] .dll sync failed, robocopy code=%RC% (>=8 means error)
if exist "%RB2LOG%" type "%RB2LOG%"
echo          backup kept at: %BACKUP%
pause & exit /b 1
:err_copy
echo [error] copy .bat failed, check that %SRC% is writable
pause & exit /b 1
