# tests-runner.ps1 - F: test suite runner for the gpu-window suite.
#
# Runs the ctest-equivalent suite without ctest: parses the CTestTestfile.cmake
# that sits in tests\, remaps every D: build-tree executable to the F: copy
# next to this script, and runs each test with a per-test timeout so one
# wedged test can no longer hold the GPU window.
#
# Semantics kept from ctest:
#   - exit code 77                          -> SKIPPED (SKIP_RETURN_CODE)
#   - stdout+stderr matches the test's
#     SKIP_REGULAR_EXPRESSION               -> SKIPPED
#   - exit 0                                -> PASSED
#   - timeout (default 600 s, or the test's
#     TIMEOUT property when smaller)        -> FAILED(timeout), process tree killed
#   - the test's ENVIRONMENT property       -> applied around its launch
#
# Usage: tests-runner.ps1 -LogDir <dir> [-TimeoutSec 600] [-Only name1,name2]
# Exit code: 0 when nothing failed, 1 otherwise.

param(
    [Parameter(Mandatory)][string]$LogDir,
    [int]$TimeoutSec = 600,
    [string]$Only = ''
)
$ErrorActionPreference = 'Stop'

$Here     = Split-Path -Parent $MyInvocation.MyCommand.Path
$TestsDir = Join-Path $Here 'tests'
$Manifest = Join-Path $TestsDir 'CTestTestfile.cmake'
if (-not (Test-Path $Manifest)) {
    # The manifest is path text only; fall back to the D: build tree copy so a
    # tests dir deployed without it still works.
    $Manifest = 'D:\msys64\home\z\src\dsh\t03\ninfer-all-fork\build-win11\tests\CTestTestfile.cmake'
    if (-not (Test-Path $Manifest)) { Write-Error "manifest not found"; exit 1 }
}

# The D: build tree prefix every test executable (and the python helper exes)
# is registered with; remap it to the F: deployment dir.
$DTests  = 'D:\msys64\home\z\src\dsh\t03\ninfer-all-fork\build-win11\tests\'
$DTestsS = 'D:/msys64/home/z/src/dsh/t03/ninfer-all-fork/build-win11/tests/'

function Remap-Path([string]$p) {
    if ($p.StartsWith($DTests))  { return (Join-Path $TestsDir ($p.Substring($DTests.Length))) }
    if ($p.StartsWith($DTestsS)) { return (Join-Path $TestsDir ($p.Substring($DTestsS.Length))) }
    return $p
}

# ---- parse the manifest ----
$tests = [ordered]@{}
$order = New-Object System.Collections.Generic.List[string]
$skipRe = @{}
$envProps = @{}
$timeouts = @{}
foreach ($line in (Get-Content $Manifest)) {
    if ($line -match '^add_test\("([^"]+)"\s+(.+)\)\s*$') {
        $name = $Matches[1]
        $toks = [regex]::Matches($Matches[2], '"((?:[^"]|\\.)*)"') | ForEach-Object { $_.Groups[1].Value }
        foreach ($t in $toks) {
            if ($t -like '*;*') { Write-Error "test $name has a list argument: $t" }
        }
        if (-not $tests.Contains($name)) { $order.Add($name) }
        $tests[$name] = [string[]]$toks
    }
    # Independent matches on purpose: one line carries several properties at
    # once, so an elseif chain would drop the later ones.
    if ($line -match '^set_tests_properties\("([^"]+)" PROPERTIES[^)]*SKIP_REGULAR_EXPRESSION "([^"]*)"') {
        $skipRe[$Matches[1]] = $Matches[2]
    }
    if ($line -match '^set_tests_properties\("([^"]+)" PROPERTIES[^)]*ENVIRONMENT "([^"]*)"') {
        $envProps[$Matches[1]] = $Matches[2]
    }
    if ($line -match '^set_tests_properties\("([^"]+)" PROPERTIES[^)]*TIMEOUT "(\d+)"') {
        $timeouts[$Matches[1]] = [int]$Matches[2]
    }
}
Write-Host ("parsed {0} tests" -f $order.Count)

# NOTE: $OnlyList on purpose (PowerShell variables are case-insensitive; a
# variable named $only would silently clobber the -Only parameter).
$OnlyList = New-Object System.Collections.Generic.List[string]
if ($Only -ne '') { foreach ($o in ($Only -split ',')) { $OnlyList.Add($o.Trim()) } }

# ---- environment for the test children ----
# TMP/TEMP must stay inside the D: tree (Low Mandatory Level): the venv python
# used by the interop tests carries a Low-labeled image, so it runs at Low
# integrity no matter who starts it and cannot write F: (Medium) or
# C:\Users\z\AppData\Local\Temp - python's tempfile then finds no usable
# directory. The D: build-tree test-tmp (the ctest redirect target) is
# writable by both the Low python and the Medium F: exes.
$tmp = 'D:\msys64\home\z\src\dsh\t03\ninfer-all-fork\build-win11\test-tmp'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$env:TMP  = $tmp
$env:TEMP = $tmp
# NINFER_TEST_ARTIFACT is inherited from the caller (gpu-window.bat sets it).
Write-Host ("NINFER_TEST_ARTIFACT={0}" -f $(if ($env:NINFER_TEST_ARTIFACT) { $env:NINFER_TEST_ARTIFACT } else { '<not set: real-artifact tests will skip>' }))
Write-Host ("tmp={0}  timeout={1}s" -f $tmp, $TimeoutSec)

# Launch a test executable and report its exit code reliably.
#
# The DSH sandbox leaves Process.ExitCode unreadable ($null) for processes
# started here, so each test runs under a cmd wrapper: the wrapper redirects
# the test's stdout/stderr to the per-test logs and echoes the child's exit
# code (delayed expansion of !ERRORLEVEL!) to its own stdout, which the
# caller reads from the code file. The wrapper stays alive for the whole test,
# so the per-test timeout and taskkill /T still cover the test process tree.
#
# Start-Process also trips over case-duplicated environment keys in DSH
# sandbox jobs (e.g. NO_PROXY + no_proxy); the loop deletes the offending
# keys and retries.
function Start-TestProcess {
    param([string]$Exe, [string[]]$TestArgs, [string]$Log, [string]$CodeFile)
    $errFile = $Log + '.err'
    # Quote every argument; escape ! and % for cmd /V:ON delayed expansion.
    $parts = @()
    foreach ($a in $TestArgs) {
        $q = $a -replace '!', '^!'
        $q = $q -replace '%', '%%'
        $parts += '"' + $q + '"'
    }
    $argPart = ''
    if ($parts.Count -gt 0) { $argPart = ' ' + ($parts -join ' ') }
    # The exe must be the first token of the /c line WITHOUT quotes: cmd /c
    # strips the first and last quote of a line that begins with a quote,
    # which would orphan the closing quote of the 2> target (123). All
    # deployed exe/python paths are space-free; quote it otherwise and fail
    # loudly instead of silently misquoting.
    if ($Exe -match '\s') { throw "cmd wrapper cannot quote the exe path (contains spaces): $Exe" }
    $qe = $Exe -replace '!', '^!'
    $inner = $qe + $argPart + ' > "' + $Log + '" 2> "' + $errFile + '" & echo TESTEXIT=!ERRORLEVEL!'
    $cmdArgs = @('/V:ON', '/c', $inner)
    if ($env:RUNNER_DEBUG -ne $null) {
        $inner | Out-File -FilePath ($CodeFile + '.inner') -Encoding utf8
        ($cmdArgs | ForEach-Object { "[" + $_ + "]" }) -join "`n" | Out-File -FilePath ($CodeFile + '.args') -Encoding utf8
        "exe=[$Exe]" | Out-File -FilePath ($CodeFile + '.exe') -Encoding utf8
    }
    for ($attempt = 0; $attempt -lt 32; $attempt++) {
        try {
            return (Start-Process -FilePath 'cmd.exe' -ArgumentList $cmdArgs -WorkingDirectory $TestsDir `
                -RedirectStandardOutput $CodeFile -RedirectStandardError ($CodeFile + '.err') -NoNewWindow -PassThru)
        } catch {
            $msg = $_.Exception.Message
            if ($msg -match "Key in dictionary: '([^']+)'") {
                $k1 = $Matches[1]
            } else { throw }
            $k2 = $null
            if ($msg -match "Key being added: '([^']+)'") { $k2 = $Matches[1] }
            [System.Environment]::SetEnvironmentVariable($k1, $null)
            if ($k2 -ne $null) { [System.Environment]::SetEnvironmentVariable($k2, $null) }
            Write-Host ("  env: removed duplicate keys {0} {1}, retrying" -f $k1, $k2)
        }
    }
    throw "Start-Process failed after 32 env-healing retries: $Exe"
}

# Read the TESTEXIT=<code> line the wrapper wrote; $null when absent.
function Read-TestExitCode([string]$CodeFile) {
    if (-not (Test-Path $CodeFile)) { return $null }
    $ct = Get-Content $CodeFile -ErrorAction SilentlyContinue
    foreach ($ln in $ct) {
        if ($ln -match 'TESTEXIT=(\d+)') { return [int]$Matches[1] }
    }
    return $null
}

# ---- run ----
$t0 = Get-Date
$passed = 0; $failed = 0; $skipped = 0
$runDir = Join-Path $LogDir 'tests'
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$i = 0
foreach ($name in $order) {
    $i++
    if ($OnlyList.Count -gt 0) {
        $matchOnly = $false
        foreach ($o in $OnlyList) { if ($o -ceq $name) { $matchOnly = $true; break } }
        if (-not $matchOnly) { continue }
    }
    $cmd = $tests[$name]
    $exe = Remap-Path $cmd[0]
    $testArgs = @()
    for ($k = 1; $k -lt $cmd.Count; $k++) { $testArgs += ,(Remap-Path $cmd[$k]) }
    $log = Join-Path $runDir ("{0:D3}-{1}.log" -f $i, $name)
    $codeFile = Join-Path $runDir ("{0:D3}-{1}.code" -f $i, $name)
    # Per-test ENVIRONMENT property (ctest "K=V;K2=V2" form). Children inherit
    # the runner's environment at spawn, so the values are set around the
    # launch and restored afterwards.
    $envPairs = @()
    if ($envProps.ContainsKey($name)) {
        foreach ($pair in ($envProps[$name] -split ';')) {
            if ($pair -match '^(.*?)=(.*)$') { $envPairs += [pscustomobject]@{ K = $Matches[1]; V = $Matches[2] } }
        }
    }
    $tmo = $TimeoutSec
    if ($timeouts.ContainsKey($name) -and $timeouts[$name] -lt $tmo) { $tmo = $timeouts[$name] }
    $prevEnv = @{}
    foreach ($e in $envPairs) {
        if ([System.Environment]::GetEnvironmentVariable($e.K) -ne $null) { $prevEnv[$e.K] = [System.Environment]::GetEnvironmentVariable($e.K) }
        Set-Item -Path ("Env:" + $e.K) -Value $e.V
    }
    $s0  = Get-Date
    try {
        $proc = Start-TestProcess -Exe $exe -TestArgs $testArgs -Log $log -CodeFile $codeFile
        $done = $proc.WaitForExit($tmo * 1000)
        $secs = [math]::Round(((Get-Date) - $s0).TotalSeconds, 2)
        if (-not $done) {
            try { taskkill /PID $proc.Id /T /F | Out-Null; $proc.WaitForExit(10000) | Out-Null } catch {}
            $result = 'FAILED(timeout)'
            $failed++
        } else {
            $code = Read-TestExitCode $codeFile
            $out = $null
            if (Test-Path $log) { $out = Get-Content $log -Raw -ErrorAction SilentlyContinue }
            $err = $null
            if (Test-Path ($log + '.err')) { $err = Get-Content ($log + '.err') -Raw -ErrorAction SilentlyContinue }
            $text = $out + "`n" + $err
            if ($code -eq $null) { $result = 'FAILED(no exit code)' }
            elseif ($code -eq 77) { $result = 'SKIPPED' }
            elseif ($skipRe.ContainsKey($name) -and $skipRe[$name] -ne '' -and ($text -match $skipRe[$name])) { $result = 'SKIPPED' }
            elseif ($code -eq 0) { $result = 'PASSED' }
            else { $result = "FAILED(code $code)" }
            if ($result -eq 'PASSED') { $passed++ }
            elseif ($result -eq 'SKIPPED') { $skipped++ }
            else { $failed++ }
        }
    } catch {
        $result = 'FAILED(launch: ' + $_.Exception.Message + ')'
        $failed++
        $secs = [math]::Round(((Get-Date) - $s0).TotalSeconds, 2)
    } finally {
        foreach ($e in $envPairs) {
            if ($prevEnv.ContainsKey($e.K)) { Set-Item -Path ("Env:" + $e.K) -Value $prevEnv[$e.K] }
            else { Remove-Item -Path ("Env:" + $e.K) -ErrorAction SilentlyContinue }
        }
    }
    Write-Host ("{0,3}/{1,3} {2,-55} {3,-18} {4,8}s" -f $i, $order.Count, $name, $result, $secs)
}

$total = [math]::Round(((Get-Date) - $t0).TotalSeconds / 60, 1)
$sum = "tests={0} passed={1} skipped={2} failed={3} total={4} min" -f $order.Count, $passed, $skipped, $failed, $total
Write-Host $sum
$sum | Out-File -FilePath (Join-Path $LogDir '1-testresults.txt') -Encoding utf8
if ($failed -gt 0) { exit 1 }
exit 0
