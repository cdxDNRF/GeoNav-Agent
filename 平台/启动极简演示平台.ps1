param(
    [string]$PythonExe = '',
    [int]$Port = 8768,
    [switch]$NoBrowser,
    [switch]$Detach
)

# GeoNav Lite 极简演示平台启动脚本 (默认端口 8768)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:NO_PROXY = "127.0.0.1,localhost"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

function Test-GeoNavHttpReady([string]$Url) {
    try {
        $req = [System.Net.HttpWebRequest]::Create($Url)
        $req.Proxy = $null
        $req.Timeout = 1500
        $req.ReadWriteTimeout = 1500
        $resp = [System.Net.HttpWebResponse]$req.GetResponse()
        $code = [int]$resp.StatusCode
        $reader = New-Object System.IO.StreamReader($resp.GetResponseStream(), [System.Text.Encoding]::UTF8)
        $body = $reader.ReadToEnd()
        $reader.Close()
        $resp.Close()
        return ($code -eq 200 -and $body.Length -gt 20)
    } catch {
        return $false
    }
}

# 1. 探测 Python 解释器
if (-not $PythonExe) {
    $PyCandidates = @(
        $env:GEONAV_PYTHON,
        "D:\PYTHON\python.exe",
        "python.exe"
    )
    foreach ($cand in $PyCandidates) {
        if (-not $cand) { continue }
        if (Test-Path -LiteralPath $cand -PathType Leaf) {
            $PythonExe = $cand
            break
        } else {
            $found = Get-Command $cand -ErrorAction SilentlyContinue
            if ($found -and (Test-Path -LiteralPath $found.Source -PathType Leaf)) {
                $PythonExe = $found.Source
                break
            }
        }
    }
}

if (-not $PythonExe) {
    Write-Host "[错误] 未能找到 Python 解释器，请先确认 Python 环境！" -ForegroundColor Red
    if (-not $Detach) { pause }
    exit 1
}

$BaseUrl = "http://127.0.0.1:$Port"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   GeoNav Lite · 视觉地理定位极简演示平台" -ForegroundColor Green
Write-Host "   (单屏清爽推演 · 192独立航拍底图960题 + 192地面题)" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "[环境] Python: $PythonExe" -ForegroundColor Gray
Write-Host "[地址] $BaseUrl/" -ForegroundColor Green

# 2. 若已有就绪的 Lite 服务在监听，直接复用并打开浏览器
if ((Test-GeoNavHttpReady "$BaseUrl/api/meta") -and (Test-GeoNavHttpReady "$BaseUrl/")) {
    Write-Host "[就绪] 检测到端口 $Port 服务已在线就绪，直接唤起浏览器..." -ForegroundColor Green
    if (-not $NoBrowser) { Start-Process "$BaseUrl/" }
    exit 0
}

$occupied = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($occupied) {
    Write-Host "[提示] 端口 $Port 已被其它进程占用，正在等待其完成初始化或请改用 -Port 指定新端口..." -ForegroundColor Yellow
}

# 3. 启动后台服务，严格等待 HTTP 200 就绪后再打开浏览器（彻底消除白屏/空屏）
$scriptPath = Join-Path $RepoRoot "project\src\webapp\lite_server.py"
$logDir = Join-Path $RepoRoot "DATA\processed_data\平台实验\工作台_v1"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$outLog = Join-Path $logDir "lite-server.log"
$errLog = Join-Path $logDir "lite-server.error.log"

Write-Host "[启动] 正在预热 Sat2Cap 视觉编码器与 192 图测试题库，请稍候..." -ForegroundColor Cyan

$proc = Start-Process -FilePath $PythonExe `
    -ArgumentList @("-B", $scriptPath, "--port", "$Port") `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog `
    -WindowStyle Hidden `
    -PassThru

$ready = $false
for ($i = 0; $i -lt 120; $i++) {
    if ($proc.HasExited) {
        Write-Host "[错误] 演示后端异常退出，错误日志如下：" -ForegroundColor Red
        Get-Content $errLog -Tail 15 -ErrorAction SilentlyContinue
        if (-not $Detach) { pause }
        exit 1
    }
    if ((Test-GeoNavHttpReady "$BaseUrl/api/meta") -and (Test-GeoNavHttpReady "$BaseUrl/")) {
        $ready = $true
        break
    }
    Start-Sleep -Milliseconds 400
}

if (-not $ready) {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    Write-Host "[超时] 服务在 48 秒内未能完成就绪响应，请检查日志: $errLog" -ForegroundColor Red
    if (-not $Detach) { pause }
    exit 1
}

Write-Host "[就绪] 服务已完成预热并响应 HTTP 200 (PID: $($proc.Id))，正在打开浏览器..." -ForegroundColor Green
if (-not $NoBrowser) {
    Start-Process "$BaseUrl/"
}

if ($Detach) {
    Write-Host "GeoNav Lite 已在后台运行: $BaseUrl/ · PID $($proc.Id)" -ForegroundColor Green
    exit 0
}

Write-Host "服务正在运行中... 按 Ctrl+C 或直接关闭本窗口即可干净退出服务。" -ForegroundColor Gray
try {
    Wait-Process -Id $proc.Id
} finally {
    if (-not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
}
