$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$taskUrl = 'http://127.0.0.1:8766'
$taskExisting = $false
try {
    $taskCatalog = Invoke-RestMethod -Uri "$taskUrl/api/catalog" -TimeoutSec 2
    $taskExisting = ($taskCatalog.mode -eq 'saved_replay' -and $taskCatalog.records -eq 15000)
} catch { $taskExisting = $false }
if ($taskExisting) {
    Start-Process $taskUrl
    exit 0
}
Set-Location -LiteralPath $taskRoot
Write-Host '先等待轨迹校验完成，再打开 http://127.0.0.1:8766；Ctrl+C 停止平台。'
& 'D:\PYTHON\python.exe' -B -m project.src.webapp.server_v1 --port 8766 --open-browser
