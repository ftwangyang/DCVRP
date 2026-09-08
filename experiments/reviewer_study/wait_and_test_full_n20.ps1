param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingPid
)

$ErrorActionPreference = 'Stop'
$repoRoot = 'C:\Users\wy\Desktop\DCVRP-main'
$checkpointRoot = Join-Path $repoRoot 'experiments\checkpoints\paper_faithful_n20_corrected_full'
$resultRoot = Join-Path $repoRoot 'experiments\results\paper_faithful_n20_corrected_full'
$statusPath = Join-Path $checkpointRoot 'postprocess.status.txt'

Set-Location -LiteralPath $repoRoot
"Waiting for training PID $TrainingPid at $(Get-Date -Format o)" | Set-Content -LiteralPath $statusPath -Encoding utf8
Wait-Process -Id $TrainingPid
"Training exited; checking acceptance at $(Get-Date -Format o)" | Add-Content -LiteralPath $statusPath -Encoding utf8

$acceptedCheckpoint = Join-Path $checkpointRoot 'best_accepted.pt'
if (-not (Test-Path -LiteralPath $acceptedCheckpoint)) {
    "No checkpoint passed the trend and Greedy criteria; final testing was not run." | Add-Content -LiteralPath $statusPath -Encoding utf8
    exit 2
}

$acceptedRelative = 'experiments\checkpoints\paper_faithful_n20_corrected_full\best_accepted.pt'
$finalRelative = 'experiments\results\paper_faithful_n20_corrected_full\final_accepted'

"Accepted checkpoint found; starting paired Greedy comparison at $(Get-Date -Format o)" | Add-Content -LiteralPath $statusPath -Encoding utf8
& python -u -m experiments.reviewer_study.run_original_uncertainty_n20 `
    --mode compare `
    --device cuda `
    --test-size 100 `
    --deterministic-rollouts 100 `
    --paired-comparison `
    --checkpoint $acceptedRelative `
    --output-dir (Join-Path $finalRelative 'greedy_comparison')
if ($LASTEXITCODE -ne 0) {
    throw "Final paired Greedy comparison failed with exit code $LASTEXITCODE"
}

"Starting deterministic reproduction at $(Get-Date -Format o)" | Add-Content -LiteralPath $statusPath -Encoding utf8
& python -u -m experiments.reviewer_study.run_original_uncertainty_n20 `
    --mode reproduce `
    --device cuda `
    --test-size 100 `
    --deterministic-rollouts 100 `
    --checkpoint $acceptedRelative `
    --output-dir (Join-Path $finalRelative 'reproduction')
if ($LASTEXITCODE -ne 0) {
    throw "Final deterministic reproduction failed with exit code $LASTEXITCODE"
}

"Starting stochastic uncertainty evaluation at $(Get-Date -Format o)" | Add-Content -LiteralPath $statusPath -Encoding utf8
& python -u -m experiments.reviewer_study.run_original_uncertainty_n20 `
    --mode test `
    --device cuda `
    --test-size 100 `
    --noise-replications 100 `
    --uncertainty-policy sampling `
    --checkpoint $acceptedRelative `
    --output-dir (Join-Path $finalRelative 'uncertainty')
if ($LASTEXITCODE -ne 0) {
    throw "Final uncertainty evaluation failed with exit code $LASTEXITCODE"
}

"All accepted-checkpoint tests completed at $(Get-Date -Format o)" | Add-Content -LiteralPath $statusPath -Encoding utf8
