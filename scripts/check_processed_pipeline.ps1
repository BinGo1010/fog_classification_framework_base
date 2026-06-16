param(
  [Parameter(Mandatory = $true)]
  [string]$ProcessedDir,
  [int]$ExpectedChannels = 0,
  [string]$RepoRoot = "E:\fog",
  [double]$WindowSeconds = 1.0,
  [double]$StrideSeconds = 1.0,
  [ValidateSet("binary", "three-class")]
  [string]$LabelMode = "binary",
  [ValidateSet("error", "zero")]
  [string]$NanPolicy = "error",
  [double]$TargetHz = 0.0,
  [switch]$AllowNan,
  [switch]$RequireSuccess,
  [switch]$KeepOutput
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
  param(
    [string]$Name,
    [scriptblock]$Body
  )
  Write-Host ""
  Write-Host "== $Name =="
  & $Body
}

function New-SafeTempOutput {
  param(
    [string]$RepoRoot,
    [string]$ProcessedDir
  )
  $outputs = Join-Path $RepoRoot "outputs"
  New-Item -ItemType Directory -Path $outputs -Force | Out-Null
  $resolvedOutputs = (Resolve-Path -LiteralPath $outputs).Path
  $leaf = Split-Path -Leaf $ProcessedDir
  $safeLeaf = $leaf -replace '[^A-Za-z0-9_.-]', '_'
  $name = "_tmp_window_dry_run_${safeLeaf}_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')"
  $path = Join-Path $resolvedOutputs $name
  return $path
}

function Remove-SafeTempOutput {
  param(
    [string]$Path,
    [string]$RepoRoot
  )
  if (-not (Test-Path -LiteralPath $Path)) {
    return
  }
  $outputs = (Resolve-Path -LiteralPath (Join-Path $RepoRoot "outputs")).Path
  $resolved = (Resolve-Path -LiteralPath $Path).Path
  if (-not $resolved.StartsWith($outputs, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to remove path outside outputs/: $resolved"
  }
  Remove-Item -LiteralPath $resolved -Recurse -Force
}

$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$ProcessedDir = (Resolve-Path -LiteralPath $ProcessedDir).Path
$OutputDir = New-SafeTempOutput -RepoRoot $RepoRoot -ProcessedDir $ProcessedDir

Write-Host "RepoRoot: $RepoRoot"
Write-Host "ProcessedDir: $ProcessedDir"
Write-Host "WindowDryRunDir: $OutputDir"

try {
  Invoke-Step "Validate sample-level processed records" {
    $args = @(
      (Join-Path $RepoRoot "scripts\validate_processed_records.py"),
      $ProcessedDir
    )
    if ($ExpectedChannels -gt 0) {
      $args += @("--expected-channels", "$ExpectedChannels")
    }
    if ($AllowNan) {
      $args += "--allow-nan"
    }
    if ($RequireSuccess) {
      $args += "--require-success"
    }
    python @args
  }

  Invoke-Step "Window dry-run" {
    $args = @(
      (Join-Path $RepoRoot "scripts\prepare_processed_record_windows.py"),
      "--processed-dir", $ProcessedDir,
      "--output-dir", $OutputDir,
      "--window-seconds", "$WindowSeconds",
      "--stride-seconds", "$StrideSeconds",
      "--label-mode", $LabelMode,
      "--nan-policy", $NanPolicy,
      "--dry-run"
    )
    if ($TargetHz -gt 0) {
      $args += @("--target-hz", "$TargetHz")
    }
    if ($RequireSuccess) {
      $args += "--require-success"
    }
    python @args
  }

  if (Test-Path -LiteralPath (Join-Path $OutputDir "windows.npz")) {
    throw "Window dry-run unexpectedly created windows.npz"
  }
  if (-not (Test-Path -LiteralPath (Join-Path $OutputDir "file_summary.csv"))) {
    throw "Window dry-run did not create file_summary.csv"
  }
  if (-not (Test-Path -LiteralPath (Join-Path $OutputDir "config.json"))) {
    throw "Window dry-run did not create config.json"
  }

  Write-Host ""
  Write-Host "Processed pipeline check passed."
}
finally {
  if (-not $KeepOutput) {
    Remove-SafeTempOutput -Path $OutputDir -RepoRoot $RepoRoot
    Write-Host "Removed temporary window dry-run output."
  }
}
