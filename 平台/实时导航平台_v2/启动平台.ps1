$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$taskUrl = 'http://127.0.0.1:8767'
$taskExisting = $false
try {
    $taskMeta = Invoke-RestMethod -Uri "$taskUrl/api/meta" -TimeoutSec 2
    $taskExisting = ($taskMeta.mode -eq '实时策略/视觉头推理；冻结图像编码缓存；已消费开发区')
} catch { $taskExisting = $false }
if ($taskExisting) {
    Start-Process $taskUrl
    exit 0
}
Set-Location -LiteralPath $taskRoot
Write-Host '先等待输入/源码绑定校验完成，再打开 http://127.0.0.1:8767；Ctrl+C 停止平台。'
$taskOpener = Start-Job -ArgumentList $taskUrl {
    param($url)
    for ($i = 0; $i -lt 240; $i++) {
        try {
            Invoke-RestMethod -Uri "$url/api/meta" -TimeoutSec 2 | Out-Null
            Start-Process $url
            return
        } catch { Start-Sleep -Milliseconds 500 }
    }
}
& 'D:\PYTHON\python.exe' -B -m project.src.webapp.server_v3 --port 8767
Remove-Job $taskOpener -Force -ErrorAction SilentlyContinue
