# Build the native Ada sm_89 RTX 4090 target with the public Engine benchmark.
# This deliberately does not change the project's default sm_86 build.
[CmdletBinding()]
param(
    [ValidateSet('12.8', '12.9', '13.1', '13.2', '13.3', '13.4')]
    [string]$CudaVersion = '13.4',
    [string]$BuildDir,
    [string]$Target,
    [ValidateRange(0, 256)][int]$Jobs = 0,
    [switch]$Clean,
    [switch]$Test,
    [switch]$ExperimentalFastSfu,
    [switch]$ExperimentalVectorSilu,
    [switch]$ExperimentalGgufIq3sT6Loop
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $BuildDir) {
    $suffix = if ($ExperimentalGgufIq3sT6Loop) { '-iq3s-t6-loop' }
              elseif ($ExperimentalFastSfu -and $ExperimentalVectorSilu) { '-sfu-vector' }
              elseif ($ExperimentalFastSfu) { '-sfu' }
              elseif ($ExperimentalVectorSilu) { '-vector' }
              else { '' }
    $BuildDir = Join-Path $RepoRoot "build-rtx4090$suffix"
}

$BuildArgs = @{
    Arch = '89'
    CudaVersion = $CudaVersion
    BuildDir = $BuildDir
    Benchmarks = $true
    ExperimentalFastSfu = [bool]$ExperimentalFastSfu
    ExperimentalVectorSilu = [bool]$ExperimentalVectorSilu
    ExperimentalGgufIq3sT6Loop = [bool]$ExperimentalGgufIq3sT6Loop
}
if ($Clean) { $BuildArgs['Clean'] = $true }
if ($Test) { $BuildArgs['Test'] = $true }
if ($Target) { $BuildArgs['Target'] = $Target }
if ($Jobs) { $BuildArgs['Jobs'] = $Jobs }
& (Join-Path $PSScriptRoot 'build.ps1') @BuildArgs

Write-Host ''
Write-Host 'RTX 4090 (sm_89) build prepared. Run the baseline benchmark with:'
Write-Host '  .\scripts\benchmark-rtx4090.ps1 -Model C:\path\to\model.ninfer'
