param(
    [string]$PythonExe = "C:\Users\wy\anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$CheckpointDir = "checkpoints/strict_env_dvnda_10ep_1000_s42"
$Checkpoint = "$CheckpointDir/DVNDA.pt"

function Invoke-Training([int]$Epochs) {
    & $PythonExe -u train.py `
        --method DVNDA `
        --customer-count 20 `
        --vehicle-count 4 `
        --epochs $Epochs `
        --steps-per-epoch 1000 `
        --batch-size 100 `
        --val-size 100 `
        --rollouts 3 `
        --seed 42 `
        --device cuda `
        --resume $Checkpoint `
        --output-dir $CheckpointDir
    if ($LASTEXITCODE -ne 0) {
        throw "Training to epoch $Epochs failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Evaluation([string]$OutputDir) {
    & $PythonExe eval.py `
        --method DVNDA `
        --customer-count 20 `
        --instances 100 `
        --seed 20260821 `
        --device cuda `
        --checkpoint "$CheckpointDir/DVNDA_best.pt" `
        --compare-table1 `
        --save-dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed with exit code $LASTEXITCODE"
    }
}

Invoke-Training 10
$TenEpochResults = "results/strict_env_dvnda_10ep_1000_fixed"
Invoke-Evaluation $TenEpochResults
& $PythonExe scripts/verify_table1_gate.py "$TenEpochResults/table1_results.csv" --max-absolute-gap 5
if ($LASTEXITCODE -eq 0) {
    Write-Output "PASS: all four dynamic rates passed after 10 epochs."
    exit 0
}

Write-Output "10 epochs did not pass all four cells; continuing to the paper's 100 epochs."
Invoke-Training 100
$PaperBudgetResults = "results/strict_env_dvnda_100ep_1000_fixed"
Invoke-Evaluation $PaperBudgetResults
& $PythonExe scripts/verify_table1_gate.py "$PaperBudgetResults/table1_results.csv" --max-absolute-gap 5
exit $LASTEXITCODE
