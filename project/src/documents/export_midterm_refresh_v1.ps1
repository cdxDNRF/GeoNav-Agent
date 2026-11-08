param([ValidateSet('report','slides')][string]$Kind='report',[ValidatePattern('^r[1-9][0-9]*$')][string]$Revision='r2')
$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$taskBase=(Resolve-Path -LiteralPath (Join-Path $taskRoot '中期报告相关/中期成果更新_20261025')).Path
$env:TEMP=Join-Path $taskBase '临时';$env:TMP=$env:TEMP
$taskIsReport=$Kind -eq 'report'
if ($taskIsReport) {
    $taskSource=Join-Path $taskBase "临时/中期报告候选_$Revision.docx"
    $taskPdf=Join-Path $taskBase "临时/中期报告候选_$Revision.pdf"
    $taskReceipt=Join-Path $taskBase "核验/报告原生导出_$Revision.json"
} else {
    $taskSource=Join-Path $taskBase '交付/中期成果汇报.pptx'
    $taskPdf=Join-Path $taskBase '交付/中期成果汇报.pdf'
    $taskReceipt=Join-Path $taskBase '核验/PPT原生PDF导出.json'
}
if (-not (Test-Path -LiteralPath $taskSource -PathType Leaf)) {throw '缺少本批候选'}
if ((Test-Path -LiteralPath $taskPdf) -or (Test-Path -LiteralPath $taskReceipt)) {throw '已有导出/回执，先核验，禁止盲重跑'}
$taskBeforeHash=(Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash
$taskApp=$null;$taskDoc=$null;$taskBefore=0;$taskExported=$false;$taskCloseWarning=$false
try {
    if ($taskIsReport) {
        $taskApp=New-Object -ComObject Word.Application;$taskBefore=$taskApp.Documents.Count
        if ($taskBefore -eq 0) {$taskApp.Visible=$false}
        $taskDoc=$taskApp.Documents.Open($taskSource,$false,$true)
        $null=$taskDoc.Fields.Update();$taskDoc.Repaginate();$taskCount=$taskDoc.ComputeStatistics(2)
        $taskDoc.ExportAsFixedFormat($taskPdf,17)
        $taskEngine='WPS Writer via Word.Application COM'
    } else {
        $taskApp=New-Object -ComObject PowerPoint.Application;$taskBefore=$taskApp.Presentations.Count
        $taskDoc=$taskApp.Presentations.Open($taskSource,$true,$false,$false)
        $taskCount=$taskDoc.Slides.Count;$taskDoc.SaveAs($taskPdf,32)
        $taskEngine='WPS Presentation via PowerPoint.Application COM'
    }
    if (-not (Test-Path -LiteralPath $taskPdf -PathType Leaf)) {throw 'PDF未生成'}
    if ((Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash -ne $taskBeforeHash) {throw '只读导出改变了候选原件'}
    [PSCustomObject]@{passed=$true;engine=$taskEngine;version=$taskApp.Version;pages_or_slides=$taskCount;source_readonly=$true;source_sha256=$taskBeforeHash;preexisting_documents=$taskBefore;source=$taskSource;pdf=$taskPdf} | ConvertTo-Json | Set-Content -LiteralPath $taskReceipt -Encoding utf8
    $taskExported=$true
} finally {
    if ($taskDoc) {
        try {if ($taskIsReport) {$taskDoc.Close(0)} else {$taskDoc.Saved=$true;$taskDoc.Close()}} catch {$taskCloseWarning=$true}
    }
    if ($taskApp -and $taskBefore -eq 0) {try {if ($taskIsReport) {$taskApp.Quit(0)} else {$taskApp.Quit()}} catch {$taskCloseWarning=$true}}
}
if ($taskExported) {Write-Output "PDF exported and source unchanged;close_warning=$taskCloseWarning";exit 0}
