param(
    [string]$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$Python = "python",
    [string]$SuiteConfig = "configs\multimodal_full_suite.json",
    [ValidateSet("all", "windowing", "validation", "training", "collection")]
    [string]$Only = "all",
    [string]$IncludeExperiments = "",
    [string]$ExcludeExperiments = "",
    [switch]$RequireWindows,
    [switch]$DryRun,
    [switch]$NoPreflight,
    [switch]$NoAudit,
    [switch]$NoReuseExistingWindows,
    [switch]$NoDedupeWindowing,
    [switch]$NoSkipCompletedTraining
)

$ErrorActionPreference = "Stop"

$LogDir = Join-Path $Repo "outputs\logs"
$LogPath = Join-Path $LogDir "multimodal_full_suite.log"
$PreflightPath = Join-Path $Repo "outputs\multimodal_full_suite_preflight.json"
$AuditPath = Join-Path $Repo "outputs\multimodal_full_suite_audit.json"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $Repo
$env:PYTHONUNBUFFERED = "1"

function Invoke-LoggedPython {
    param([string[]]$ArgsList)
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] CMD $Python $($ArgsList -join ' ')" |
        Tee-Object -FilePath $LogPath -Append
    & $Python @ArgsList 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] start multimodal full suite" |
    Tee-Object -FilePath $LogPath -Append

if (-not $NoPreflight) {
    $PreflightArgs = @(
        "scripts\preflight_fog_suite.py",
        "--config", $SuiteConfig,
        "--output-json", $PreflightPath
    )
    if ($RequireWindows) {
        $PreflightArgs += "--require-windows"
    }
    Invoke-LoggedPython -ArgsList $PreflightArgs
}

$SuiteArgs = @(
    "scripts\run_fog_suite.py",
    "--config", $SuiteConfig,
    "--only", $Only
)
if ($DryRun) {
    $SuiteArgs += "--dry-run"
}
if ($NoReuseExistingWindows) {
    $SuiteArgs += "--no-reuse-existing-windows"
}
if ($NoDedupeWindowing) {
    $SuiteArgs += "--no-dedupe-windowing"
}
if ($NoSkipCompletedTraining) {
    $SuiteArgs += "--no-skip-completed-training"
}
if ($IncludeExperiments) {
    $SuiteArgs += @("--include-experiments", $IncludeExperiments)
}
if ($ExcludeExperiments) {
    $SuiteArgs += @("--exclude-experiments", $ExcludeExperiments)
}

Invoke-LoggedPython -ArgsList $SuiteArgs

if ((-not $NoAudit) -and (-not $DryRun) -and ($Only -in @("all", "training", "collection"))) {
    $AuditArgs = @(
        "scripts\audit_fog_suite_results.py",
        "--config", $SuiteConfig,
        "--output-json", $AuditPath
    )
    Invoke-LoggedPython -ArgsList $AuditArgs
}

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] finished multimodal full suite" |
    Tee-Object -FilePath $LogPath -Append
