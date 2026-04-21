$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv311\Scripts\python.exe"
$dataPath = Join-Path $repoRoot "data\Dataset"

$tuneLog = Join-Path $repoRoot "results\tta_experiments_logs\component_gate_research_tuning.log"
$gateLog = Join-Path $repoRoot "results\tta_experiments_logs\component_gate_research_gate.log"
$statusPath = Join-Path $repoRoot "results\tta_experiments_logs\component_gate_research_status.json"

$status = [ordered]@{
    started_at = (Get-Date).ToString("s")
    stage = "tuning"
    tuning = [ordered]@{
        started = $null
        finished = $null
        exit_code = $null
        log = $tuneLog
    }
    gate_diagnostics = [ordered]@{
        started = $null
        finished = $null
        exit_code = $null
        log = $gateLog
    }
}

$status | ConvertTo-Json -Depth 5 | Set-Content -Path $statusPath -Encoding UTF8

$tuneArgs = @(
    "scripts\tune_component_gates_stepwise.py",
    "--datasets", "EEG,HAR,FD",
    "--backbone", "CNN",
    "--da-method", "ACCUP",
    "--data-path", $dataPath,
    "--device", "cuda",
    "--seeds", "41,42,43",
    "--save-dir", (Join-Path $repoRoot "results\tta_experiments_logs\component_gate_tuning_runs"),
    "--output-dir", (Join-Path $repoRoot "results\tta_experiments_logs\component_gate_tuning_summary"),
    "--pretrain-cache-dir", (Join-Path $repoRoot "results\pretrain_cache"),
    "--write-overrides",
    "--adv-sigma-step", "0.05",
    "--adv-sigma-points", "9",
    "--adv-num-span", "16",
    "--adv-num-step", "4",
    "--adv-num-max", "64",
    "--cons-step", "0.1",
    "--cons-points", "13",
    "--sem-step", "0.1",
    "--sem-points", "15",
    "--proto-step", "0.1",
    "--proto-points", "9"
)

$status.stage = "tuning"
$status.tuning.started = (Get-Date).ToString("s")
$status | ConvertTo-Json -Depth 5 | Set-Content -Path $statusPath -Encoding UTF8

& $python @tuneArgs *>&1 | Tee-Object -FilePath $tuneLog
$status.tuning.finished = (Get-Date).ToString("s")
$status.tuning.exit_code = $LASTEXITCODE
$status | ConvertTo-Json -Depth 5 | Set-Content -Path $statusPath -Encoding UTF8
if ($LASTEXITCODE -ne 0) {
    throw "Component/gate tuning failed with exit code $LASTEXITCODE"
}

$gateArgs = @(
    "scripts\run_gate_diagnostics.py",
    "--data_path", $dataPath,
    "--device", "cuda",
    "--seeds", "41,42,43",
    "--backbone", "CNN"
)

$status.stage = "gate_diagnostics"
$status.gate_diagnostics.started = (Get-Date).ToString("s")
$status | ConvertTo-Json -Depth 5 | Set-Content -Path $statusPath -Encoding UTF8

& $python @gateArgs *>&1 | Tee-Object -FilePath $gateLog
$status.gate_diagnostics.finished = (Get-Date).ToString("s")
$status.gate_diagnostics.exit_code = $LASTEXITCODE
$status.stage = "completed"
$status.completed_at = (Get-Date).ToString("s")
$status | ConvertTo-Json -Depth 5 | Set-Content -Path $statusPath -Encoding UTF8
if ($LASTEXITCODE -ne 0) {
    throw "Gate diagnostics failed with exit code $LASTEXITCODE"
}
