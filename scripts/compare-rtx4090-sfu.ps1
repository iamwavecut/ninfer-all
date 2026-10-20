# Compare the baseline RTX 4090 CUDA kernels to the opt-in fast SFU build.
# The existing A/B runner requires both EXEs in the same directory for dependent DLLs.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$BaseBuildDir,
    [string]$SfuBuildDir,
    [ValidateRange(2, 30)][int]$Repetitions = 6,
    [ValidateSet('none', 'mtp', 'dflash2')][string]$Spec = 'none',
    [ValidateRange(1, 15)][int]$DraftTokens = 3
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
if (-not $BaseBuildDir) { $BaseBuildDir = Join-Path $Repo 'build-rtx4090' }
if (-not $SfuBuildDir) { $SfuBuildDir = Join-Path $Repo 'build-rtx4090-sfu' }
$A = Join-Path $BaseBuildDir 'apps\ninfer.exe'
$SourceB = Join-Path $SfuBuildDir 'apps\ninfer.exe'
$B = Join-Path $BaseBuildDir 'apps\ninfer-sfu-compare.exe'
foreach ($File in @($Model, $A, $SourceB)) {
    if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { throw "Missing file: $File" }
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'python is required' }
if ((Resolve-Path -LiteralPath $A).Path -eq (Resolve-Path -LiteralPath $SourceB).Path) {
    throw 'The baseline and SFU executables must be built separately'
}
# Do not overwrite a pre-existing comparison executable; it could be from another build.
if (Test-Path -LiteralPath $B) { throw "Remove or rename an older comparison executable: $B" }
Copy-Item -LiteralPath $SourceB -Destination $B
$Model = (Resolve-Path -LiteralPath $Model).Path
$A = (Resolve-Path -LiteralPath $A).Path
$B = (Resolve-Path -LiteralPath $B).Path
$Prompt = 'Explain how CUDA Graphs and GPU memory bandwidth influence local LLM decoding.'
$SpecFlags = ''
if ($Spec -ne 'none') {
    $SpecFlags = "--spec $Spec --draft-tokens $DraftTokens"
}
$Config = 'sfu:{exe} "' + $Model + '" --prompt "' + $Prompt +
          '" --max-new 256 --max-context 8192 --kv-dtype rk4v4 --greedy ' + $SpecFlags
Write-Host 'Baseline versus SFU: paired and order-alternating decode benchmark'
Write-Host 'The test measures speed only; run correctness checks on both builds as well.'
try {
    Push-Location $Repo
    try {
        python tools/bench/run_interleaved_ab.py --arm-a $A --arm-b $B --reps $Repetitions --config $Config
        if ($LASTEXITCODE -ne 0) { throw "A/B comparison exited $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
} finally {
    # Clean up the compare-only copy, leaving both build products untouched.
    Remove-Item -LiteralPath $B -Force -ErrorAction SilentlyContinue
}
