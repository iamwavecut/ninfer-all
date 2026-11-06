r''' 2>nul
@echo off
setlocal
rem ---------------------------------------------------------------------------------------------
rem Stop the fp8-dev TRIAL API server (port 10999). Double-click it to run.
rem
rem Windows port of stop.bat, adapted to the trial deployment: same graceful double CTRL+C
rem (the disk KV flush needs it), force-kill after 15 s, plus the trial extras: it stops the
rem nvidia-smi sampler (smi-stop.flag) and prints an smi.csv summary and the VRAM state.
rem
rem This file is a cmd/python polyglot. For the interpreter, everything from the four-quote first
rem line to the closing triple-quote line is one string literal (a no-op); the batch logic sits
rem inside it and the real python sits below it. Consequences: no three consecutive quotes may
rem appear anywhere in the batch region, and every batch path must exit before the python region
rem so cmd never parses it.
rem
rem Requires python on PATH (checked below).
rem ---------------------------------------------------------------------------------------------

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found on PATH; it is required to stop the service.
  pause
  exit /b 1
)

set "PORT=10999"
set "PID="

echo === Stopping the fp8-dev TRIAL service (port %PORT%) ===

rem 1. The PID owning the port. The port alone is not proof it is ours.
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "LISTENING" ^| findstr /r ":%PORT% "') do set "PID=%%a"
if defined PID (
  tasklist /fi "PID eq %PID%" /fo csv /nh | findstr /i "ninfer-serve" >nul
  if errorlevel 1 set "PID="
)

rem 2. No process-name fallback on purpose: the production engine shares the name. Only the
rem trial's port may act; a name match would kill production.
if not defined PID goto :not_running

echo Found trial serve (PID %PID%). Pressing CTRL+C, again in one second, then waiting up to 15 s...

python "%~f0" %PID%
set "STOP_STATUS=%ERRORLEVEL%"

if "%STOP_STATUS%"=="0" (
  echo Service stopped. A "STOPPED n" line above means a graceful shutdown with the disk KV flushed; "GONE" means it had already exited.
) else (
  echo Service stopped by force; the disk KV may not be flushed.
)
goto :smi

:not_running
echo No trial service found listening on port %PORT%.
goto :smi

:smi
echo ======================================
rem Stop the sampler and summarize its capture (the newest trial-* run dir).
rem for /d (not for /f + powershell): cmd re-parses for /f '...' regions and a piped
rem powershell one-liner breaks (a literal ^ reaches the interpreter); and a plain
rem wildcard for skips directories on this host's cmd, while for /d enumerates them
rem in name order, so the last match is the newest stamp. %%~fdd is the full path.
set "RUN="
for /d %%d in ("%~dp0logs\trial-*") do set "RUN=%%~fdd"
rem Goto-style on purpose: the powershell summary line below is full of
rem parentheses, which a multi-line cmd block cannot tolerate (the block
rem terminates at the first ) character on a body line).
if not defined RUN goto :smi_end
echo Trial run dir: %RUN%
> "%RUN%\smi-stop.flag" echo stopped
if not exist "%RUN%\smi.csv" goto :smi_end
echo SMI sample summary:
powershell -NoProfile -Command "$r = Import-Csv '%RUN%\smi.csv'; $n = @($r).Count; $sm = @($r | Where-Object {$_.sm_mhz -match '^\d+$'} | ForEach-Object {[int]$_.sm_mhz}); $pw = @($r | Where-Object {$_.power_w -match '^\d'} | ForEach-Object {[double]$_.power_w}); $mem = @($r | Where-Object {$_.memory_used_mib -match '^\d+$'} | ForEach-Object {[int]$_.memory_used_mib}); $s = 'rows=' + $n; if ($sm.Count) { $m = $sm | Measure-Object -Minimum -Maximum; $s += ' | sm=' + $m.Minimum + '..' + $m.Maximum + ' MHz' }; if ($pw.Count) { $s += ' | power_avg=' + [math]::Round(($pw | Measure-Object -Average).Average, 0) + ' W' }; if ($mem.Count) { $s += ' | mem_max=' + ($mem | Measure-Object -Maximum).Maximum + ' MiB' }; $s"
:smi_end
rem pause
exit /b 0
'''
# The python part. cmd never reaches here: every batch path above exits first. The PID arrives as
# argv[1]; the batch region above is this module's docstring.
import ctypes, subprocess, sys, time
from ctypes import wintypes
k = ctypes.windll.kernel32
k.OpenProcess.restype = wintypes.HANDLE
k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
k.CloseHandle.argtypes = [wintypes.HANDLE]
k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
k.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.DWORD]
pid = int(sys.argv[1])
k.FreeConsole()
sender = "import ctypes\nk = ctypes.windll.kernel32\nk.FreeConsole()\nk.AttachConsole(" + str(pid) + ")\nk.GenerateConsoleCtrlEvent(0, 0)"
STILL_ACTIVE = 259
def dead():
    h = k.OpenProcess(1024, False, pid)
    if not h:
        return True
    code = wintypes.DWORD()
    ok = k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return not ok or code.value != STILL_ACTIVE
def press():
    subprocess.run([sys.executable, "-c", sender])
if dead():
    print("GONE")
    raise SystemExit(0)
press()
time.sleep(1.0)
press()
waited = 0
while waited < 15 and not dead():
    time.sleep(1.0)
    waited = waited + 1
if dead():
    print("STOPPED " + str(waited))
    raise SystemExit(0)
h = k.OpenProcess(1, False, pid)
if h:
    k.TerminateProcess(h, 1)
    k.CloseHandle(h)
print("FORCED")
raise SystemExit(1)
