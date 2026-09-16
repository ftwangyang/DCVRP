param(
    [string]$PythonExe = "C:\Users\wy\anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$CheckpointDir = "checkpoints/dvnda_n20_s42"
$Checkpoint = "$CheckpointDir/DVNDA.pt"

function Train-To([int]$Epochs) {
    & $PythonExe -u train.py --method DVNDA --customer-count 20 `
        --vehicle-count 4 --epochs $Epochs --steps-per-epoch 1000 `
        --batch-size 100 --val-size 100 --rollouts 3 --seed 42 `
        --device cuda --resume $Checkpoint --output-dir $CheckpointDir
    if ($LASTEXITCODE -ne 0) { throw "Training failed at target epoch $Epochs" }
}

function Evaluate-And-Gate([string]$ResultDir) {
    & $PythonExe eval.py --method DVNDA --customer-count 20 --instances 100 `
        --seed 20260821 --device cuda --checkpoint "$CheckpointDir/DVNDA_best.pt" `
        --compare-table1 --save-dir $ResultDir
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed" }
    & $PythonExe scripts/verify_table1_gate.py "$ResultDir/table1_results.csv" `
        --max-absolute-gap 5
    return $LASTEXITCODE
}

Train-To 10
$gate = Evaluate-And-Gate "results/epoch10"
if ($gate -eq 0) {
    Write-Output "PASS at epoch 10"
    exit 0
}

Write-Output "Epoch 10 failed the four-cell gate; continuing to the paper budget."
Train-To 100
$gate = Evaluate-And-Gate "results/epoch100"
exit $gate
