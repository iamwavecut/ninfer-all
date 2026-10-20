# Build the native Ada sm_89 RTX 4090 target with the public Engine benchmark.
# This deliberately does not change the project's default sm_86 build.
[CmdletBinding()]
param(
    [ValidateSet('12.8', '12.9', '13.1', '13.2', '13.3', '13.4')]
    [string]$CudaVersion = '13.4',
    [string]$BuildDir,
    [switch]$Clean,
    [switch]$Test
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $BuildDir) { $BuildDir = Join-Path $RepoRoot 'build-rtx4090' }

$BuildArgs = @{
    Arch = '89'
    CudaVersion = $CudaVersion
    BuildDir = $BuildDir
    Benchmarks = $true
}
if ($Clean) { $BuildArgs['Clean'] = $true }
if ($Test) { $BuildArgs['Test'] = $true }
& (Join-Path $PSScriptRoot 'build.ps1') @BuildArgs

Write-Host ''
Write-Host 'RTX 4090 (sm_89) build prepared. Run the baseline benchmark with:'
Write-Host '  .\scripts\benchmark-rtx4090.ps1 -Model C:\path\to\model.ninfer'
