param(
    [string]$DataPath = "D:\PyCharm Project\ACCUP + EATA\data\Dataset",
    [int]$Start = 30,
    [int]$End = 200,
    [string]$Scenario = "1->0",
    [int]$Seed = 42,
    [string]$Dataset = "FD",
    [string]$SaveDir = ".\results\tta_experiments_logs",
    [string]$Python = "python"
)

$resultsDir = ".\results"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}

$bestF1 = -1.0
$bestBatch = -1
$rows = @()

for ($bs = $Start; $bs -le $End; $bs++) {
    Write-Host ">>> batch_size=$bs"
    $expName = "bs_$bs"

    $output = & $Python "trainers/tta_trainer.py" `
        --dataset $Dataset `
        --da_method "ACCUP" `
        --scenario $Scenario `
        --override "batch_size=$bs" `
        --num_runs 1 `
        --seed $Seed `
        --exp_name $expName `
        --save_dir $SaveDir `
        --data-path $DataPath 2>&1

    $f1 = [double]::NaN
    $line = ($output | Select-String -Pattern "Average current f1_scores::" | Select-Object -Last 1)
    if ($line -and $line.Line -match "Average current f1_scores::\s*([0-9.]+)") {
        $f1 = [double]$Matches[1]
    }

    $rows += [pscustomobject]@{
        batch_size = $bs
        f1_score = $f1
    }

    if (-not [double]::IsNaN($f1) -and $f1 -gt $bestF1) {
        $bestF1 = $f1
        $bestBatch = $bs
    }
}

$outCsv = ".\results\fd10_batch_search.csv"
$rows | Sort-Object batch_size | Export-Csv -NoTypeInformation $outCsv
Write-Host "Best batch_size=$bestBatch  f1=$bestF1"
Write-Host "Saved results to $outCsv"
