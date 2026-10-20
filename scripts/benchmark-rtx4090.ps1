# Reproducible RTX 4090 baseline using the repository's public-Engine benchmark matrix.
# Run only on a quiet GPU. Does not modify runtime profiles, weights or kernel routing.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [ValidateSet('smoke', 'core', 'full')][string]$Preset = 'core',
    [ValidateRange(0, 7)][int]$Device = 0,
    [string]$BuildDir,
    [string]$OutputDir,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $BuildDir) { $BuildDir = Join-Path $RepoRoot 'build-rtx4090' }
$Bench = Join-Path $BuildDir 'bench\ninfer_bench.exe'
$Runner = Join-Path $RepoRoot 'tools\bench\run_ninfer_bench_matrix.py'
$Corpus = Join-Path $RepoRoot 'bench\fixtures\bench_corpus.ids'
if (-not (Test-Path -LiteralPath $Bench -PathType Leaf)) {
    throw "Missing benchmark executable: $Bench. First run .\scripts\build-rtx4090.ps1"
}
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "Model not found: $Model" }
if (-not (Test-Path -LiteralPath $Corpus -PathType Leaf)) { throw "Benchmark corpus not found: $Corpus" }
$Model = (Resolve-Path -LiteralPath $Model).Path
$Bench = (Resolve-Path -LiteralPath $Bench).Path
$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) { throw 'Python is required to run tools/bench/run_ninfer_bench_matrix.py' }
$Smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $Smi) { throw 'nvidia-smi is required to record the GPU configuration' }
$Gpus = @(& $Smi.Source '--query-gpu=index,name' '--format=csv,noheader')
if ($LASTEXITCODE -ne 0 -or $Gpus.Count -eq 0) { throw 'Could not query installed GPUs with nvidia-smi' }
if (-not @($Gpus | Where-Object { $_ -match 'GeForce RTX 4090' }).Count) {
    throw "No GeForce RTX 4090 reported by nvidia-smi: $($Gpus -join '; ')"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $RepoRoot ('profiles\bench\rtx4090-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing to overwrite existing results directory: $OutputDir" }
$null = New-Item -ItemType Directory -Path $OutputDir -Force
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path
$Commit = $null
if (Get-Command git -ErrorAction SilentlyContinue) {
    $Commit = (& git -C $RepoRoot rev-parse HEAD 2>$null)
}
$WeightsInfo = Get-Item -LiteralPath $Model
$Manifest = [ordered]@{
    timestamp_utc = (Get-Date).ToUniversalTime().ToString('o')
    repository_commit = $Commit
    benchmark_binary = $Bench
    model_path = $Model
    model_bytes = $WeightsInfo.Length
    model_modified_utc = $WeightsInfo.LastWriteTimeUtc.ToString('o')
    selected_cuda_device = $Device
    note = 'nvidia-smi index ordering may differ from CUDA device ordering; verify with ninfer-multi-gpu-probe on mixed systems'
    nvidia_smi_gpus = $Gpus
    nvidia_smi_driver = @(& $Smi.Source '--query-gpu=driver_version,power.limit' '--format=csv,noheader')
    preset = $Preset
    dry_run = [bool]$DryRun
}
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDir 'rtx4090-host.json') -Encoding utf8
$Arguments = @(
    $Runner, '--preset', $Preset, '--bench', $Bench, '--weights', $Model,
    '--corpus', $Corpus, '--device', [string]$Device,
    '--output-dir', $OutputDir, '--no-build'
)
if ($DryRun) { $Arguments += '--dry-run' }
Write-Host "RTX 4090 benchmark: $Preset on CUDA device $Device"
Write-Host "Results: $OutputDir"
Push-Location $RepoRoot
try {
    & $Python.Source @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed with exit code $LASTEXITCODE (logs: $OutputDir)" }
} finally {
    Pop-Location
}
Write-Host "Benchmark complete: $OutputDir"
