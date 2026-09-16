param(
    [string]$PythonExe = "C:\Users\wy\anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$CheckpointDir = "checkpoints/strict_env_dvnda_10ep_1000_s42"

& $PythonExe -u train.py `
    --method DVNDA `
    --customer-count 20 `
    --vehicle-count 4 `
    --epochs 10 `
    --steps-per-epoch 1000 `
    --batch-size 100 `
    --val-size 100 `
    --rollouts 3 `
    --seed 42 `
    --device cuda `
    --output-dir $CheckpointDir

if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}

& $PythonExe eval.py `
    --method DVNDA `
    --customer-count 20 `
    --instances 100 `
    --device cuda `
    --checkpoint "$CheckpointDir/DVNDA.pt" `
    --compare-table1 `
    --save-dir "results/strict_env_dvnda_10ep_1000_final"

if ($LASTEXITCODE -ne 0) {
    throw "Final-checkpoint evaluation failed with exit code $LASTEXITCODE"
}

& $PythonExe eval.py `
    --method DVNDA `
    --customer-count 20 `
    --instances 100 `
    --device cuda `
    --checkpoint "$CheckpointDir/DVNDA_best.pt" `
    --compare-table1 `
    --save-dir "results/strict_env_dvnda_10ep_1000_best_validation"

if ($LASTEXITCODE -ne 0) {
    throw "Best-validation evaluation failed with exit code $LASTEXITCODE"
}

& $PythonExe scripts/verify_table1_gate.py `
    "results/strict_env_dvnda_10ep_1000_best_validation/table1_results.csv" `
    --max-absolute-gap 5

if ($LASTEXITCODE -ne 0) {
    throw "Strict Table-I gate failed: at least one dynamic rate exceeds 5%"
}
