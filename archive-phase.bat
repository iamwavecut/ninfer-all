@echo off
setlocal EnableExtensions
REM ---------------------------------------------------------------------------
REM archive-phase.bat - archive one phase's serve deliverables to F:\gsoft\archive.
REM
REM Usage: archive-phase.bat ^<phase^> [evidence_dir]
REM   phase         e.g. p1, p2 (directory name becomes ninfer-fp8dev-<phase>-<YYYYMMDD>)
REM   evidence_dir  optional; its *.txt / *.log / *.csv files are copied to evidence\
REM
REM Copies build-win11\apps (the five exes + runtime DLLs) into bin\, writes
REM sha256-manifest.txt and build-info.txt. Re-runnable: robocopy /IS overwrites
REM in place and both text files are rewritten.
REM ---------------------------------------------------------------------------

cd /d "%~dp0"

set "PHASE=%~1"
set "EV=%~2"
if not "%PHASE%"=="" goto have_phase
echo usage: archive-phase.bat ^<phase^> [evidence_dir]
exit /b 1
:have_phase

set "STAMP="
for /f "delims=" %%a in ('powershell -NoProfile -Command "[DateTime]::Now.ToString('yyyyMMdd')"') do set "STAMP=%%a"
if defined STAMP goto have_stamp
echo [error] could not get a timestamp
exit /b 1
:have_stamp

set "DEST=F:\gsoft\archive\ninfer-fp8dev-%PHASE%-%STAMP%"
set "SRC=build-win11\apps"
echo archiving %PHASE% into %DEST%

robocopy "%SRC%" "%DEST%\bin" ninfer-serve.exe ninfer.exe ninfer-perplexity.exe ninfer-calibrate.exe ninfer-ngram-profile.exe *.dll /IS /NFL /NDL /NJH /NJS
set "RC=%ERRORLEVEL%"
if %RC% LSS 8 goto have_bin
echo [error] robocopy exited %RC%
exit /b 1
:have_bin

if not "%EV%"=="" goto have_ev
echo [note] no evidence dir given; skipping evidence copy
goto write_manifest
:have_ev
robocopy "%EV%" "%DEST%\evidence" *.txt *.log *.csv /IS /NFL /NDL /NJH /NJS
set "RC=%ERRORLEVEL%"
if %RC% LSS 8 goto have_ev2
echo [error] evidence robocopy exited %RC%
exit /b 1
:have_ev2

:write_manifest
powershell -NoProfile -Command "Get-ChildItem '%DEST%\bin' -File | Sort-Object Name | ForEach-Object { (Get-FileHash $_.FullName -Algorithm SHA256).Hash + '  ' + $_.Name } | Set-Content -Path '%DEST%\sha256-manifest.txt' -Encoding ASCII"
if errorlevel 1 goto manifest_fail
set "GITHEAD="
for /f "delims=" %%a in ('git rev-parse HEAD') do set "GITHEAD=%%a"
set "GITDESC="
for /f "delims=" %%a in ('git describe --always') do set "GITDESC=%%a"
set "GITBRANCH="
for /f "delims=" %%a in ('git rev-parse --abbrev-ref HEAD') do set "GITBRANCH=%%a"
> "%DEST%\build-info.txt" echo ninfer-fp8dev phase %PHASE% archive, %STAMP%
echo git HEAD: %GITHEAD% (%GITDESC%) >> "%DEST%\build-info.txt"
echo branch: %GITBRANCH% >> "%DEST%\build-info.txt"
echo build dir: build-win11 (Release, sm_89) >> "%DEST%\build-info.txt"
echo cmake: CMAKE_BUILD_TYPE=Release CMAKE_CUDA_ARCHITECTURES=89 NINFER_SM89_FP8=ON NINFER_DIRECTSTORAGE=ON >> "%DEST%\build-info.txt"
echo toolchain: MSVC 14.51.36231 (VS 18 Community) + CUDA 13.4.59 + CMake 4.4.4 + Ninja >> "%DEST%\build-info.txt"
echo card: RTX 4090 48 GB, unlocked (ssdiag readback required before perf runs) >> "%DEST%\build-info.txt"
echo artifact: G:\AI_Models\Qwen3.8-27B-FP8-NInfer-v3\Qwen3.8-27B-FP8-v3-full.ninfer >> "%DEST%\build-info.txt"
echo.
echo ARCHIVE_DONE %DEST%
exit /b 0
:manifest_fail
echo [error] manifest generation failed
exit /b 1
