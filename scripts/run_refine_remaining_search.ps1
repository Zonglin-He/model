param(
    [string]$Python = "D:\PyCharm Project\ACCUP + EATA\.venv311\Scripts\python.exe",
    [string]$DataPath = "D:\PyCharm Project\ACCUP + EATA\data\Dataset",
    [string]$Device = "cuda",
    [string]$Seeds = "41,42,43",
    [int]$Trials = 12,
    [string]$Workspace = "D:\PyCharm Project\ACCUP + EATA"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

Set-Location $Workspace

$logDir = Join-Path $Workspace "results\tta_experiments_logs\refine_search_summary"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$logPath = Join-Path $logDir "run_refine_remaining_search.log"

& $Python "scripts/refine_remaining_scenarios.py" `
    --datasets "EEG,HAR,FD" `
    --data-path $DataPath `
    --device $Device `
    --seeds $Seeds `
    --n-trials $Trials `
    --write-updates `
    --save-dir (Join-Path $Workspace "results\tta_experiments_logs\refine_search_runs") `
    --pretrain-cache-dir (Join-Path $Workspace "results\pretrain_cache") `
    --output-dir $logDir *>> $logPath

exit $LASTEXITCODE
