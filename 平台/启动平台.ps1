param([string]$PythonExe='', [int]$Port=8769, [switch]$NoBrowser)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:NO_PROXY='127.0.0.1,localhost'

function Invoke-LocalJson([string]$Url) {
    $req = [System.Net.HttpWebRequest]::Create($Url)
    $req.Proxy = $null
    $req.Timeout = 2000
    $req.ReadWriteTimeout = 2000
    $resp = [System.Net.HttpWebResponse]$req.GetResponse()
    $reader = New-Object System.IO.StreamReader($resp.GetResponseStream(), [System.Text.Encoding]::UTF8)
    $raw = $reader.ReadToEnd()
    $reader.Close()
    $resp.Close()
    return ($raw | ConvertFrom-Json)
}

function Test-LocalPage([string]$Url) {
    try {
        $req = [System.Net.HttpWebRequest]::Create($Url)
        $req.Proxy = $null
        $req.Timeout = 2000
        $resp = [System.Net.HttpWebResponse]$req.GetResponse()
        $ok = ([int]$resp.StatusCode -eq 200)
        $resp.Close()
        return $ok
    } catch {
        return $false
    }
}

$taskRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $PythonExe) {
    if ($env:GEONAV_PYTHON) { $PythonExe=$env:GEONAV_PYTHON }
    elseif (Test-Path -LiteralPath 'D:\PYTHON\python.exe') { $PythonExe='D:\PYTHON\python.exe' }
    else { $taskPython=Get-Command python -ErrorAction SilentlyContinue; if ($taskPython) { $PythonExe=$taskPython.Source } }
}
if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw '未找到Python。请通过 -PythonExe 指定已安装Python；搬迁包不含运行时。' }
$taskUrl="http://127.0.0.1:$Port"

# Reuse the exact current version before the preserved free-port checker.
$taskExisting=$null
try { $taskExisting=Invoke-LocalJson "$taskUrl/api/meta" } catch { }
if ($taskExisting -and $taskExisting.version -eq 'geonav-workbench-v5.0' -and (Test-LocalPage "$taskUrl/")) {
    if (-not $NoBrowser) { Start-Process "$taskUrl/" }
    Write-Host '已使用现有GeoNav5.0一体化平台服务'
    return
}

Push-Location -LiteralPath $taskRoot
try {
    $taskProbeRaw=& $PythonExe -B -m project.src.tools.release_controller_workbench_v1 probe --port $Port
    if ($LASTEXITCODE -ne 0) { throw '端口检查失败，未启动服务。' }
} finally { Pop-Location }
$taskProbe=$taskProbeRaw | ConvertFrom-Json
if ($taskProbe.state -ne 'free') { throw "端口已有服务或无法核实空闲；原服务保留。状态：$($taskProbe.state)/$($taskProbe.reason)。可通过 -Port 另选端口。" }

Push-Location -LiteralPath $taskRoot
try {
    # Keep native Python arguments free of nested quotes (Windows PowerShell 5.1).
    & $PythonExe -B -c 'import numpy, PIL, torch'
    if ($LASTEXITCODE -ne 0) { throw '依赖不齐；请查看搬迁包 requirements-workbench.txt。启动器不会自动安装或下载。' }
    Write-Host 'Python与本地导航依赖检查通过，正在预热 GeoNav V5.0 一体化工作台...'
    $taskLog=Join-Path $taskRoot 'DATA/processed_data/平台实验/工作台_v1'
    New-Item -ItemType Directory -Path $taskLog -Force | Out-Null
    $taskStamp=Get-Date -Format 'yyyyMMdd-HHmmss'
    $taskProcess=Start-Process -FilePath $PythonExe -ArgumentList @('-B','-m','project.src.webapp.workbench_server_v28','--port',"$Port") -WorkingDirectory $taskRoot -RedirectStandardOutput (Join-Path $taskLog "server-$taskStamp.log") -RedirectStandardError (Join-Path $taskLog "server-$taskStamp.error.log") -WindowStyle Hidden -PassThru
    for ($taskTry=0; $taskTry -lt 100; $taskTry++) {
        if ($taskProcess.HasExited) { throw '工作台启动失败，请检查工作台目录中的error.log。' }
        try {
            $taskMeta=Invoke-LocalJson "$taskUrl/api/meta"
            if ($taskMeta.version -eq 'geonav-workbench-v5.0' -and (Test-LocalPage "$taskUrl/")) {
                if (-not $NoBrowser) { Start-Process "$taskUrl/" }
                Write-Host "GeoNav V5.0一体化工作台已启动：$taskUrl/ · PID $($taskProcess.Id)"
                return
            }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    throw '服务仍在初始化，请检查日志；不要盲目启动第二份。'
} finally { Pop-Location }
