# Build NInfer-3090 on Windows with the toolchain this project actually needs.
#
# Three things are not the defaults on a typical machine, and getting any of them wrong produces
# an error that does not name the real cause:
#
#   MSVC 14.4x from VS 2022 BuildTools. VS 2026 ships MSVC 14.50 (_MSC_VER 1950), which CUDA
#   12.8's host_config.h rejects outright - it accepts 1910-1949. If a VS 2026 cl.exe wins on
#   PATH you get a wall of host_config errors that say nothing about the compiler version.
#
#   The selected CUDA toolkit is forced through CUDACXX, avoiding a different nvcc on PATH.
#   The existing default prefers CUDA 12.8/12.9; -CudaVersion selects a newer installed toolkit
#   explicitly without silently changing the toolchain of existing builds.
#
#   The Ninja generator. MSBuild's CUDA integration needs CUDA_PATH_V12_8, which the CUDA
#   installer does not always set; without it CudaToolkitDir resolves empty and every .cu fails.
#
# Running this from a plain PowerShell prompt is fine: it imports the BuildTools environment
# itself rather than requiring a Developer Prompt.
#
#   .\scripts\build.ps1                  configure + build into build-ninja
#   .\scripts\build.ps1 -Test            ... then run the test suite
#   .\scripts\build.ps1 -Package         ... then build the release archive for VERSION
#   .\scripts\build.ps1 -Benchmarks      ... include bench\ (implied by -Package)
#   .\scripts\build.ps1 -Clean           delete the build directory first
#   .\scripts\build.ps1 -Target ninfer-serve
#   .\scripts\build.ps1 -Arch 89 -CudaVersion 13.4 -Benchmarks -BuildDir build-rtx4090
[CmdletBinding()]
param(
    [switch]$Test,
    [switch]$Package,
    [switch]$Clean,
    [switch]$Benchmarks,
    [switch]$ExperimentalFastSfu,
    [string]$Target,
    [ValidateRange(0, 256)][int]$Jobs = 0,
    [string]$BuildDir,
    [ValidateSet('12.8', '12.9', '13.1', '13.2', '13.3', '13.4')][string]$CudaVersion,
    [ValidateSet('75', '80', '86', '89')][string]$Arch = '86'
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $BuildDir) { $BuildDir = Join-Path $RepoRoot 'build-ninja' }

# The packager ships bench\ninfer_bench.exe alongside the CLI and the server, but
# benchmarks are an opt-in subdirectory (NINFER_BUILD_BENCHMARKS defaults to OFF). Configuring
# without them and then packaging fails late, after the whole tree has been built, with
# "Missing release product: ...\bench\ninfer_bench.exe" - so make -Package imply the option rather
# than leaving the two settings to be kept consistent by hand.
$BuildBenchmarks = $Benchmarks -or $Package

# --- locate the toolchain ---------------------------------------------------------------------

$VcVarsCandidates = @(
    'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
    'C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
    'C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat',
    'C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat',
    'C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat'
)
$VcVars = $VcVarsCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $VcVars) {
    throw @"
No Visual Studio 2022 x64 build environment found. Looked in:
$($VcVarsCandidates -join "`n")
Install "Desktop development with C++" from VS 2022 Build Tools. VS 2026 will not work: its
MSVC 14.50 is rejected by CUDA 12.8.
"@
}

$CudaCandidates = @(
    'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe',
    'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe'
)
if ($CudaVersion) {
    $CudaCandidates = @("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$CudaVersion\bin\nvcc.exe")
}
$Nvcc = $CudaCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $Nvcc) {
    throw @"
No suitable CUDA toolkit found. Looked in:
$($CudaCandidates -join "`n")
CMakeLists requires CUDA >= 12.8. Choose an installed version with -CudaVersion
(or install CUDA 12.8/12.9 to use the existing default).
"@
}

# vcvars64.bat only exports into its own cmd process, so run it and copy the result back.
Write-Host "toolchain: $VcVars"
Write-Host "toolchain: $Nvcc"
$VcEnv = cmd /c "`"$VcVars`" >nul 2>&1 && set"
# Windows may expose both PATH and Path. Importing the latter last discards
# vcvars' compiler directory in PowerShell's case-insensitive environment.
$VcPath = $VcEnv | Where-Object {
    $_ -match '^PATH=' -and $_ -match 'MSVC\\[^;]+\\bin\\HostX64\\x64'
} | Select-Object -First 1
if (-not $VcPath) { throw 'vcvars64 did not return PATH' }
$VcEnv | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$' -and $Matches[1] -ine 'PATH') {
        Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
    }
}
$env:PATH = $VcPath.Substring(5)
$env:CUDACXX = $Nvcc

$VcpkgRoot = if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { 'C:\vcpkg' }
$VcpkgToolchain = Join-Path $VcpkgRoot 'scripts\buildsystems\vcpkg.cmake'
if (-not (Test-Path -LiteralPath $VcpkgToolchain -PathType Leaf)) {
    throw "vcpkg toolchain not found: $VcpkgToolchain (set VCPKG_ROOT)"
}

$cl = (Get-Command cl.exe -ErrorAction SilentlyContinue)
if ($cl) { Write-Host "toolchain: $($cl.Source)" }
$rc = (Get-Command rc.exe -ErrorAction SilentlyContinue)
if (-not $rc) { throw 'Windows SDK rc.exe was not found in the VS 2022 build environment' }
Write-Host "toolchain: $($rc.Source)"
$RcCmake = $rc.Source.Replace('\', '/')

# --- configure and build ----------------------------------------------------------------------

if ($Clean -and (Test-Path -LiteralPath $BuildDir)) {
    Write-Host "removing $BuildDir"
    Remove-Item -LiteralPath $BuildDir -Recurse -Force
}

Push-Location $RepoRoot
try {
    # Quote the -D arguments: PowerShell does not reliably expand a variable inside a bare token
    # that begins with "-D", and cmake then sees the literal "$Arch".
    $BenchmarksOption = if ($BuildBenchmarks) { 'ON' } else { 'OFF' }
    $TestsOption = if ($Test) { 'ON' } else { 'OFF' }
    $SfuOption = if ($ExperimentalFastSfu) { 'ON' } else { 'OFF' }
    cmake -S . -B $BuildDir -G Ninja '-DCMAKE_BUILD_TYPE=Release' "-DCMAKE_CUDA_ARCHITECTURES=$Arch" `
          "-DCMAKE_RC_COMPILER=$RcCmake" `
          "-DCMAKE_TOOLCHAIN_FILE=$VcpkgToolchain" '-DVCPKG_TARGET_TRIPLET=x64-windows' `
          "-DNINFER_BUILD_BENCHMARKS=$BenchmarksOption" "-DBUILD_TESTING=$TestsOption" `
          "-DNINFER_SFU_SIGMOID_SILU=$SfuOption" "-DNINFER_SFU_SOFTPLUS=$SfuOption"
    if ($LASTEXITCODE -ne 0) { throw "configure failed ($LASTEXITCODE)" }

    $buildArgs = @('--build', $BuildDir, '-j')
    if ($Jobs) { $buildArgs += $Jobs }
    if ($Target) { $buildArgs += @('--target', $Target) }
    cmake @buildArgs
    if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }

    if ($Test -and -not $Target) {
        # One GPU, so keep the parallelism low: unrelated CUDA tests contend for memory and
        # produce failures that do not reproduce when the test is run on its own.
        ctest --test-dir $BuildDir -j2 --output-on-failure
        if ($LASTEXITCODE -ne 0) { throw "tests failed ($LASTEXITCODE)" }
    }

    if ($Package) {
        $packager = Join-Path $PSScriptRoot 'package-release.ps1'
        # The packager defaults to build-ninja\; point it at the tree we actually built, so
        # -BuildDir and -Package agree. build.sh already does this for the Linux packager.
        $PreviousBuildRoot = $env:NINFER_BUILD_ROOT
        $env:NINFER_BUILD_ROOT = $BuildDir
        try {
            & $packager
            if ($LASTEXITCODE -ne 0) { throw "packaging failed ($LASTEXITCODE)" }
        } finally {
            $env:NINFER_BUILD_ROOT = $PreviousBuildRoot
        }
    }
} finally {
    Pop-Location
}

Write-Host ''
Write-Host "built into $BuildDir"
Write-Host "  server : $(Join-Path $BuildDir 'apps\ninfer-serve.exe')"
Write-Host "  cli    : $(Join-Path $BuildDir 'apps\ninfer.exe')"
