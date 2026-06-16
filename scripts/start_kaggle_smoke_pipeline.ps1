param(
    [string]$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$Python = "python",
    [string]$DatasetRoot = "",
    [int]$SmokeLimit = 5,
    [ValidateSet("compressed", "none")]
    [string]$RecordCompression = "compressed",
    [ValidateSet("all", "windowing", "validation", "training", "collection")]
    [string]$Only = "all",
    [switch]$Execute,
    [switch]$Resume,
    [switch]$Overwrite,
    [switch]$NoPreflight,
    [switch]$AllowExecuteWithoutPreflight,
    [switch]$AllowExecuteWithoutStatusGate,
    [switch]$NoValidation,
    [switch]$PostCheckWindowDryRun,
    [switch]$NoSuite,
    [switch]$NoReuseExistingWindows,
    [switch]$NoDedupeWindowing,
    [switch]$NoSkipCompletedTraining,
    [switch]$ProfileData,
    [string]$LogPath = "",
    [string]$PreflightPath = "",
    [string]$DryRunReportPath = "",
    [string]$StatusPath = ""
)

$ErrorActionPreference = "Stop"

if ($Resume -and $Overwrite) {
    throw "-Resume and -Overwrite are mutually exclusive."
}
if ($Execute -and $NoPreflight -and (-not $AllowExecuteWithoutPreflight)) {
    throw "-Execute with -NoPreflight requires -AllowExecuteWithoutPreflight."
}

$Repo = (Resolve-Path -LiteralPath $Repo).Path
if ($DatasetRoot -eq "") {
    $DatasetRoot = Join-Path $Repo "dataset"
}
$DatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path

function Resolve-RunPath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Repo $PathValue))
}

function Get-KaggleDir {
    param([string]$Root)
    $matches = @(Get-ChildItem -LiteralPath $Root -Force | Where-Object { $_.PSIsContainer -and $_.Name -like "2.Kaggle*" })
    if ($matches.Count -ne 1) {
        throw "Expected one 2.Kaggle* directory under $Root, found $($matches.Count)."
    }
    return $matches[0].FullName
}

$KaggleDir = Get-KaggleDir -Root $DatasetRoot
$ProcessedSmoke = Join-Path $KaggleDir "processed_smoke"
if ($LogPath -eq "") {
    $LogPath = Join-Path $Repo "outputs\logs\kaggle_smoke_pipeline.log"
} else {
    $LogPath = Resolve-RunPath -PathValue $LogPath
}
if ($PreflightPath -eq "") {
    $PreflightPath = Join-Path $Repo "outputs\kaggle_preflight_report.json"
} else {
    $PreflightPath = Resolve-RunPath -PathValue $PreflightPath
}
if ($DryRunReportPath -eq "") {
    $DryRunReportPath = Join-Path $Repo "outputs\kaggle_smoke_streaming_dry_run.json"
} else {
    $DryRunReportPath = Resolve-RunPath -PathValue $DryRunReportPath
}
if ($StatusPath -eq "") {
    $StatusPath = Join-Path $Repo "outputs\kaggle_status.json"
} else {
    $StatusPath = Resolve-RunPath -PathValue $StatusPath
}
$SuiteConfig = "configs\kaggle_smoke_suite.json"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
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

function Write-LogLine {
    param([string]$Message)
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" |
        Tee-Object -FilePath $LogPath -Append
}

Write-LogLine "start Kaggle smoke pipeline execute=$Execute smoke_limit=$SmokeLimit"

if (-not $NoPreflight) {
    Invoke-LoggedPython -ArgsList @(
        "scripts\check_kaggle_fog_preflight.py",
        "--repo-root", $Repo,
        "--dataset-root", $DatasetRoot,
        "--suite-config", $SuiteConfig,
        "--smoke-limit", [string]$SmokeLimit,
        "--skip-pytest",
        "--output-json", $PreflightPath
    )
}

$PreprocessArgs = @(
    "scripts\preprocess_kaggle_fog_streaming.py",
    "--dataset-root", $DatasetRoot,
    "--source", "both",
    "--valid-only",
    "--task-only",
    "--strict-metadata",
    "--smoke-limit", [string]$SmokeLimit,
    "--record-compression", $RecordCompression
)
$DryRunArgs = @(
    "--check-headers",
    "--dry-run",
    "--dry-run-output-json", $DryRunReportPath
)
if ($ProfileData) {
    $DryRunArgs += "--profile-data"
}

if ($Execute) {
    Invoke-LoggedPython -ArgsList ($PreprocessArgs + $DryRunArgs)
    if (-not $AllowExecuteWithoutStatusGate) {
        $StatusArgs = @(
            "scripts\kaggle_fog_status.py",
            "--repo-root", $Repo,
            "--dataset-root", $DatasetRoot,
            "--preflight-json", $PreflightPath,
            "--dry-run-json", $DryRunReportPath,
            "--output-json", $StatusPath,
            "--require-ready", "smoke"
        )
        if ($Resume -or $Overwrite) {
            $StatusArgs += "--allow-existing-output"
        }
        Invoke-LoggedPython -ArgsList $StatusArgs
    }
    if ($Resume) {
        $PreprocessArgs += "--resume"
    }
    if ($Overwrite) {
        $PreprocessArgs += "--overwrite"
    }
} else {
    $PreprocessArgs += $DryRunArgs
}

Invoke-LoggedPython -ArgsList $PreprocessArgs

if ($Execute -and (-not $NoValidation)) {
    if ($PostCheckWindowDryRun) {
        Invoke-LoggedPython -ArgsList @(
            "scripts\check_processed_pipeline.py",
            "--processed-dir", $ProcessedSmoke,
            "--expected-channels", "3",
            "--require-success",
            "--window-seconds", "1",
            "--stride-seconds", "1",
            "--label-mode", "binary",
            "--nan-policy", "error",
            "--target-hz", "100"
        )
    } else {
        Invoke-LoggedPython -ArgsList @(
            "scripts\validate_processed_records.py",
            $ProcessedSmoke,
            "--expected-channels", "3",
            "--require-success"
        )
    }
} elseif (-not $Execute) {
    Write-LogLine "skip processed_smoke validation because -Execute was not provided"
}

if (-not $NoSuite) {
    $SuiteArgs = @(
        "scripts\run_fog_suite.py",
        "--config", $SuiteConfig,
        "--only", $Only,
        "--validate-experiment-configs"
    )
    if (-not $Execute) {
        $SuiteArgs += @("--dry-run", "--skip-collection")
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
    Invoke-LoggedPython -ArgsList $SuiteArgs
}

Write-LogLine "finished Kaggle smoke pipeline execute=$Execute"
