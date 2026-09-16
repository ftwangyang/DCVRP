param(
    [Parameter(Mandatory = $true)]
    [int]$InitialSupervisorPid,
    [string]$PythonExe = "C:\Users\wy\anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$CheckpointDir = "checkpoints/strict_env_dvnda_10ep_1000_s42"
$TenEpochCsv = "results/strict_env_dvnda_10ep_1000_best_validation/table1_results.csv"

Wait-Process -Id $InitialSupervisorPid -ErrorAction SilentlyContinue

if (-not (Test-Path -LiteralPath $TenEpochCsv)) {
    throw "The 10-epoch evaluation CSV was not produced"
}

& $PythonExe scripts/verify_table1_gate.py $TenEpochCsv --max-absolute-gap 5
if ($LASTEXITCODE -eq 0) {
    Write-Output "The 10-epoch checkpoint passed all four cells; no continuation required."
    exit 0
}

Write-Output "The 10-epoch checkpoint failed the strict gate; resuming to the manuscript's 100-epoch budget."
& $PythonExe -u train.py `
    --method DVNDA `
    --customer-count 20 `
    --vehicle-count 4 `
    --epochs 100 `
    --steps-per-epoch 1000 `
    --batch-size 100 `
    --val-size 100 `
    --rollouts 3 `
    --seed 42 `
    --device cuda `
    --resume "$CheckpointDir/DVNDA.pt" `
    --output-dir $CheckpointDir

if ($LASTEXITCODE -ne 0) {
    throw "Paper-budget continuation failed with exit code $LASTEXITCODE"
}

& $PythonExe eval.py `
    --method DVNDA `
    --customer-count 20 `
    --instances 100 `
    --device cuda `
    --checkpoint "$CheckpointDir/DVNDA_best.pt" `
    --compare-table1 `
    --save-dir "results/strict_env_dvnda_100ep_1000_best_validation"

if ($LASTEXITCODE -ne 0) {
    throw "Paper-budget evaluation failed with exit code $LASTEXITCODE"
}

& $PythonExe scripts/verify_table1_gate.py `
    "results/strict_env_dvnda_100ep_1000_best_validation/table1_results.csv" `
    --max-absolute-gap 5
exit $LASTEXITCODE
