@echo off
REM cdb capture for the ninfer_expert_delivery_test hang. The test hangs deterministically in
REM its first disk-expert run (worker never sets prepared; main spins in wait_host), a few
REM seconds after the fixture is written. cdb starts the test, sleeps 90 s for the hang to
REM settle, dumps every thread's stack, and quits.
REM
REM Run with the GPU free (production + trial stopped) from its deployment dir (F:).
REM Symbols: the A8 dev build tree (build-win11) that produced the F: test exe.
cd /d "%~dp0"
set "TMP=D:\msys64\home\z\src\dsh\t03\ninfer-all-fork\build-win11\test-tmp"
set "TEMP=%TMP%"
if not exist "%TMP%" mkdir "%TMP%"
set "SYMS=D:\msys64\home\z\src\dsh\t03\ninfer-all-fork\build-win11;D:\msys64\home\z\src\dsh\t03\ninfer-all-fork\build-win11\tests;D:\msys64\home\z\src\dsh\t03\ninfer-all-fork\build-win11\src\core;D:\msys64\home\z\src\dsh\t03\ninfer-all-fork\build-win11\src\ops;D:\msys64\home\z\src\dsh\t03\ninfer-all-fork\build-win11\src\models;D:\msys64\home\z\src\dsh\t03\ninfer-all-fork\build-win11\src\runtime"
"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\cdb.exe" -y "%SYMS%" -c "s 90; ~*kv 40; q" "F:\gsoft\gw\tests\ninfer_expert_delivery_test.exe" > "%~dp0cdb-expert-delivery.log" 2>&1
echo CDB_EXIT=%ERRORLEVEL%
