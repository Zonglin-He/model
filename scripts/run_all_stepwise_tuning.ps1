param(
    [string]$Python = "D:\PyCharm Project\ACCUP + EATA\.venv311\Scripts\python.exe",
    [string]$DataPath = "D:\PyCharm Project\ACCUP + EATA\data\Dataset",
    [string]$Device = "cuda",
    [string]$Seeds = "41,42,43",
    [string]$Workspace = "D:\PyCharm Project\ACCUP + EATA"
)

$ErrorActionPreference = "Stop"

Set-Location $Workspace

$logDir = Join-Path $Workspace "results\tta_experiments_logs\stepwise_all"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$masterLog = Join-Path $logDir "run_all_stepwise_tuning.log"
$statusPath = Join-Path $logDir "run_all_stepwise_tuning_status.json"

$runs = @(
    @{
        Dataset = "EEG"
        SaveDir = Join-Path $Workspace "results\tta_experiments_logs\eeg_stepwise_runs"
        OutputDir = Join-Path $Workspace "results\tta_experiments_logs\eeg_stepwise_summary"
    },
    @{
        Dataset = "HAR"
        SaveDir = Join-Path $Workspace "results\tta_experiments_logs\har_stepwise_runs"
        OutputDir = Join-Path $Workspace "results\tta_experiments_logs\har_stepwise_summary"
    },
    @{
        Dataset = "FD"
        SaveDir = Join-Path $Workspace "results\tta_experiments_logs\fd_stepwise_runs"
        OutputDir = Join-Path $Workspace "results\tta_experiments_logs\fd_stepwise_summary"
    }
)

$status = [ordered]@{
    started_at = (Get-Date).ToString("s")
    finished_at = $null
    device = $Device
    seeds = $Seeds
    workspace = $Workspace
    runs = @()
}

foreach ($run in $runs) {
    $dataset = $run.Dataset
    $saveDir = $run.SaveDir
    $outputDir = $run.OutputDir

    if (-not (Test-Path $saveDir)) {
        New-Item -ItemType Directory -Path $saveDir | Out-Null
    }
    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir | Out-Null
    }

    $runStatus = [ordered]@{
        dataset = $dataset
        started_at = (Get-Date).ToString("s")
        finished_at = $null
        exit_code = $null
        summary_json = Join-Path $outputDir "summary.json"
        summary_csv = Join-Path $outputDir "summary.csv"
    }
    $status.runs += $runStatus
    $status | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $statusPath

    Add-Content -Path $masterLog -Value ("[{0}] START {1}" -f (Get-Date).ToString("s"), $dataset)

    & $Python "scripts/tune_stepwise.py" `
        --dataset $dataset `
        --da-method "ACCUP" `
        --backbone "CNN" `
        --data-path $DataPath `
        --device $Device `
        --seeds $Seeds `
        --save-dir $saveDir `
        --output-dir $outputDir `
        --pretrain-cache-dir (Join-Path $Workspace "results\pretrain_cache") `
        *>> $masterLog

    $runStatus.exit_code = $LASTEXITCODE
    $runStatus.finished_at = (Get-Date).ToString("s")
    $status | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $statusPath

    Add-Content -Path $masterLog -Value ("[{0}] END {1} exit={2}" -f (Get-Date).ToString("s"), $dataset, $LASTEXITCODE)

    if ($LASTEXITCODE -ne 0) {
        throw "Stepwise tuning failed for $dataset with exit code $LASTEXITCODE"
    }
}

$status.finished_at = (Get-Date).ToString("s")
$status | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $statusPath
Add-Content -Path $masterLog -Value ("[{0}] ALL_DONE" -f (Get-Date).ToString("s"))
