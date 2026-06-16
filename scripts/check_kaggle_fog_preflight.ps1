param(
  [string]$RepoRoot = "E:\fog",
  [string]$DatasetRoot = "E:\fog\dataset",
  [string]$PytestBaseTemp = "",
  [string]$SuiteConfig = "",
  [double]$ReserveGib = 5.0,
  [int]$SmokeLimit = 0,
  [switch]$AllowInsufficientStorage,
  [switch]$SkipPytest,
  [string]$OutputJson = ""
)

$ErrorActionPreference = "Stop"

if ($ReserveGib -lt 0) {
  throw "-ReserveGib must be >= 0."
}
if ($SmokeLimit -lt 0) {
  throw "-SmokeLimit must be >= 0."
}

function Invoke-Step {
  param(
    [string]$Name,
    [scriptblock]$Body
  )
  Write-Host ""
  Write-Host "== $Name =="
  try {
    $global:LASTEXITCODE = 0
    & $Body
    if ($LASTEXITCODE -ne 0) {
      throw "Step '$Name' failed with exit code $LASTEXITCODE."
    }
    $Script:Steps += [PSCustomObject]@{
      name = $Name
      status = "passed"
      returncode = 0
    }
  }
  catch {
    $Script:Steps += [PSCustomObject]@{
      name = $Name
      status = "failed"
      returncode = $LASTEXITCODE
    }
    throw
  }
}

function Get-KaggleDir {
  param([string]$Root)
  $matches = @(Get-ChildItem -LiteralPath $Root -Force | Where-Object { $_.PSIsContainer -and $_.Name -like "2.Kaggle*" })
  if ($matches.Count -ne 1) {
    throw "Expected one 2.Kaggle* directory under $Root, found $($matches.Count)."
  }
  return $matches[0].FullName
}

function Format-GiB {
  param([Int64]$Bytes)
  return ("{0:N3} GiB" -f ($Bytes / 1GB))
}

function Get-DirectoryFileStats {
  param([string]$Path)
  $files = @(Get-ChildItem -LiteralPath $Path -Recurse -File -Force)
  $sum = ($files | Measure-Object -Property Length -Sum).Sum
  if ($null -eq $sum) {
    $sum = 0
  }
  return [PSCustomObject]@{
    Count = $files.Count
    SizeBytes = [Int64]$sum
  }
}

function Get-ExtractedCompetitionDataReport {
  param([string]$KaggleDir)
  $extractedPath = Join-Path $KaggleDir "competition data"
  $exists = Test-Path -LiteralPath $extractedPath
  if (-not $exists) {
    return [PSCustomObject]@{
      exists = $false
      path = $extractedPath
      file_count = 0
      size_bytes = 0
      size_gib = 0.0
      status = "ignored_by_zip_streaming_pipeline"
    }
  }
  $stats = Get-DirectoryFileStats -Path $extractedPath
  return [PSCustomObject]@{
    exists = $true
    path = $extractedPath
    file_count = $stats.Count
    size_bytes = $stats.SizeBytes
    size_gib = [Math]::Round(($stats.SizeBytes / 1GB), 6)
    status = "ignored_by_zip_streaming_pipeline"
  }
}

function Write-ExtractedCompetitionDataReport {
  param([object]$Stats)
  Write-Host "extracted_competition_data exists: $($Stats.exists)"
  if (-not $Stats.exists) {
    return
  }
  Write-Host "extracted_competition_data files: $($Stats.file_count)"
  Write-Host "extracted_competition_data size: $(Format-GiB -Bytes $Stats.size_bytes)"
  Write-Host "extracted_competition_data status: ignored by zip-streaming Kaggle pipeline"
}

function Read-JsonFile {
  param([string]$Path)
  if (($Path -eq "") -or (-not (Test-Path -LiteralPath $Path))) {
    return $null
  }
  return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Get-ObjectPropertyValue {
  param(
    [object]$Object,
    [string]$Name
  )
  if ($null -eq $Object) {
    return $null
  }
  $property = $Object.PSObject.Properties[$Name]
  if ($null -eq $property) {
    return $null
  }
  return $property.Value
}

function Get-ObjectCount {
  param(
    [object]$Object,
    [string]$Name
  )
  $value = Get-ObjectPropertyValue -Object $Object -Name $Name
  if ($null -eq $value) {
    return 0
  }
  return [int64]$value
}

function Get-ZipStructureReport {
  param([string]$KaggleDir)
  $inventoryPath = Join-Path $KaggleDir "inventory\kaggle_zip_inventory_summary.json"
  $summary = Read-JsonFile -Path $inventoryPath
  if ($null -eq $summary) {
    return [PSCustomObject]@{
      ok = $false
      inventory_path = $inventoryPath
      required_path_buckets = [PSCustomObject]@{}
      required_metadata_files = [PSCustomObject]@{}
      selected_supervised_train_csv_files = 0
      skipped_path_buckets = [PSCustomObject]@{}
      errors = @("Missing zip inventory summary: $inventoryPath")
      warnings = @()
    }
  }

  $errors = New-Object System.Collections.Generic.List[string]
  $requiredPathBuckets = [ordered]@{}
  foreach ($bucketName in @("train/tdcsfog", "train/defog")) {
    $bucket = Get-ObjectPropertyValue -Object $summary.path_buckets -Name $bucketName
    $fileCount = Get-ObjectCount -Object $bucket -Name "file_count"
    $csvCount = Get-ObjectCount -Object $bucket -Name "csv_count"
    $compressedSize = Get-ObjectCount -Object $bucket -Name "compressed_size"
    $uncompressedSize = Get-ObjectCount -Object $bucket -Name "uncompressed_size"
    $exists = (($fileCount -gt 0) -and ($csvCount -gt 0))
    $requiredPathBuckets[$bucketName] = [PSCustomObject]@{
      exists = $exists
      file_count = $fileCount
      csv_count = $csvCount
      compressed_size = $compressedSize
      uncompressed_size = $uncompressedSize
    }
    if (-not $exists) {
      $errors.Add("Missing required supervised train CSV bucket: $bucketName")
    }
  }

  $metadataGroup = Get-ObjectPropertyValue -Object $summary.groups -Name "metadata"
  $metadataPaths = @()
  if (($null -ne $metadataGroup) -and ($null -ne $metadataGroup.sample_paths)) {
    $metadataPaths = @($metadataGroup.sample_paths | ForEach-Object { [string]$_ })
  }
  $requiredMetadataFiles = [ordered]@{}
  foreach ($metadataName in @(
    "tdcsfog_metadata.csv",
    "defog_metadata.csv",
    "subjects.csv",
    "events.csv",
    "tasks.csv",
    "daily_metadata.csv"
  )) {
    $exists = $metadataPaths -contains $metadataName
    $requiredMetadataFiles[$metadataName] = [PSCustomObject]@{ exists = $exists }
    if (-not $exists) {
      $errors.Add("Missing required metadata file: $metadataName")
    }
  }

  $skippedPathBuckets = [ordered]@{}
  foreach ($bucketName in @("train/notype", "unlabeled")) {
    $bucket = Get-ObjectPropertyValue -Object $summary.path_buckets -Name $bucketName
    $skippedPathBuckets[$bucketName] = [PSCustomObject]@{
      file_count = Get-ObjectCount -Object $bucket -Name "file_count"
      csv_count = Get-ObjectCount -Object $bucket -Name "csv_count"
      compressed_size = Get-ObjectCount -Object $bucket -Name "compressed_size"
      uncompressed_size = Get-ObjectCount -Object $bucket -Name "uncompressed_size"
    }
  }

  $selectedCount = 0
  foreach ($bucket in $requiredPathBuckets.Values) {
    $selectedCount += [int]$bucket.csv_count
  }

  return [PSCustomObject]@{
    ok = ($errors.Count -eq 0)
    inventory_path = $inventoryPath
    required_path_buckets = [PSCustomObject]$requiredPathBuckets
    required_metadata_files = [PSCustomObject]$requiredMetadataFiles
    selected_supervised_train_csv_files = $selectedCount
    skipped_path_buckets = [PSCustomObject]$skippedPathBuckets
    errors = @($errors)
    warnings = @()
  }
}

function Write-ZipStructureReport {
  param([object]$Report)
  foreach ($bucketName in @("train/tdcsfog", "train/defog")) {
    $bucket = Get-ObjectPropertyValue -Object $Report.required_path_buckets -Name $bucketName
    Write-Host "$bucketName`: files=$($bucket.file_count) csv=$($bucket.csv_count) uncompressed=$(Format-GiB -Bytes $bucket.uncompressed_size)"
  }
  foreach ($metadataName in @(
    "tdcsfog_metadata.csv",
    "defog_metadata.csv",
    "subjects.csv",
    "events.csv",
    "tasks.csv",
    "daily_metadata.csv"
  )) {
    $status = Get-ObjectPropertyValue -Object $Report.required_metadata_files -Name $metadataName
    Write-Host "$metadataName`: exists=$($status.exists)"
  }
  foreach ($message in @($Report.errors)) {
    Write-Error "zip_structure_error: $message"
  }
}

function Read-And-RemoveJson {
  param([string]$Path)
  $value = Read-JsonFile -Path $Path
  if (($Path -ne "") -and (Test-Path -LiteralPath $Path)) {
    Remove-Item -LiteralPath $Path -Force
  }
  return $value
}

function Write-JsonAtomic {
  param(
    [string]$Path,
    [object]$Value
  )
  if ($Path -eq "") {
    return
  }
  $parent = Split-Path -Parent $Path
  if ($parent -ne "") {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
  }
  $leaf = Split-Path -Leaf $Path
  $tmp = Join-Path $parent ".$leaf.tmp"
  $Value | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $tmp -Encoding UTF8
  Move-Item -LiteralPath $tmp -Destination $Path -Force
}

function Convert-ErrorToReport {
  param([object]$ErrorRecord)
  if ($null -eq $ErrorRecord) {
    return $null
  }
  return [PSCustomObject]@{
    type = $ErrorRecord.Exception.GetType().Name
    message = $ErrorRecord.Exception.Message
  }
}

function Write-PreflightReport {
  param(
    [string]$Status,
    [object]$ErrorRecord = $null
  )
  if ($OutputJson -eq "") {
    return
  }
  $hasProcessedAfter = Test-Path -LiteralPath $ProcessedPath
  $hasSmokeAfter = Test-Path -LiteralPath $SmokePath
  $report = [PSCustomObject]@{
    status = $Status
    repo_root = $RepoRoot
    dataset_root = $DatasetRoot
    kaggle_dir = $KaggleDir
    processed_output_guard = [PSCustomObject]@{
      processed_path = $ProcessedPath
      processed_smoke_path = $SmokePath
      processed_exists_before = $HadProcessed
      processed_smoke_exists_before = $HadSmoke
      processed_exists_after = $hasProcessedAfter
      processed_smoke_exists_after = $hasSmokeAfter
      no_processed_output_created = (($HadProcessed -or (-not $hasProcessedAfter)) -and ($HadSmoke -or (-not $hasSmokeAfter)))
    }
    extracted_competition_data = $ExtractedStats
    zip_inventory = Read-JsonFile -Path (Join-Path $KaggleDir "inventory\kaggle_zip_inventory_summary.json")
    zip_structure = $Script:ZipStructureReport
    preflight_options = [PSCustomObject]@{
      smoke_limit = $SmokeLimit
      suite_config = $SuiteConfig
    }
    storage_estimate = Read-And-RemoveJson -Path $StorageReportPath
    streaming_dry_run = Read-And-RemoveJson -Path $StreamingReportPath
    suite_preflight = Read-And-RemoveJson -Path $SuitePreflightReportPath
    suite_dry_run = [PSCustomObject]@{
      config = $SuiteConfig
      validated_experiment_configs = (($Status -eq "passed") -and [bool]($Script:Steps | Where-Object { $_.name -eq "Kaggle suite config dry-run" -and $_.status -eq "passed" }))
    }
    pytest = [PSCustomObject]@{
      ran = [bool]($Script:Steps | Where-Object { $_.name -eq "Synthetic Kaggle tests" })
      basetemp = $PytestBaseTemp
    }
    steps = $Script:Steps
  }
  $errorInfo = Convert-ErrorToReport -ErrorRecord $ErrorRecord
  if ($null -ne $errorInfo) {
    $report | Add-Member -NotePropertyName error -NotePropertyValue $errorInfo
  }
  Write-JsonAtomic -Path $OutputJson -Value $report
  Write-Host "preflight_report_json: $OutputJson"
}

$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$DatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path
if ($PytestBaseTemp -eq "") {
  $PytestBaseTemp = Join-Path $RepoRoot "outputs\pytest-kaggle-preflight"
}
if ($SuiteConfig -eq "") {
  $SuiteConfig = Join-Path $RepoRoot "configs\kaggle_smoke_suite.json"
} elseif (-not [System.IO.Path]::IsPathRooted($SuiteConfig)) {
  $SuiteConfig = Join-Path $RepoRoot $SuiteConfig
}
$SuiteConfig = (Resolve-Path -LiteralPath $SuiteConfig).Path
if ($OutputJson -ne "") {
  if (-not [System.IO.Path]::IsPathRooted($OutputJson)) {
    $OutputJson = Join-Path $RepoRoot $OutputJson
  }
  $OutputJson = [System.IO.Path]::GetFullPath($OutputJson)
}
$OutputJsonParent = if ($OutputJson -ne "") { Split-Path -Parent $OutputJson } else { "" }
$OutputJsonLeaf = if ($OutputJson -ne "") { Split-Path -Leaf $OutputJson } else { "" }
$StorageReportPath = if ($OutputJson -ne "") { Join-Path $OutputJsonParent ".$OutputJsonLeaf.storage.tmp.json" } else { "" }
$StreamingReportPath = if ($OutputJson -ne "") { Join-Path $OutputJsonParent ".$OutputJsonLeaf.streaming.tmp.json" } else { "" }
$SuitePreflightReportPath = if ($OutputJson -ne "") { Join-Path $OutputJsonParent ".$OutputJsonLeaf.suite_preflight.tmp.json" } else { "" }
$Script:Steps = @()
$Script:ZipStructureReport = $null
$KaggleDir = Get-KaggleDir -Root $DatasetRoot
$ProcessedPath = Join-Path $KaggleDir "processed"
$SmokePath = Join-Path $KaggleDir "processed_smoke"
$HadProcessed = Test-Path -LiteralPath $ProcessedPath
$HadSmoke = Test-Path -LiteralPath $SmokePath
$ExtractedStats = Get-ExtractedCompetitionDataReport -KaggleDir $KaggleDir

Write-Host "RepoRoot: $RepoRoot"
Write-Host "DatasetRoot: $DatasetRoot"
Write-Host "KaggleDir: $KaggleDir"
Write-Host "processed exists before: $HadProcessed"
Write-Host "processed_smoke exists before: $HadSmoke"
Write-ExtractedCompetitionDataReport -Stats $ExtractedStats

try {
Invoke-Step "Compile scripts" {
  python -m py_compile `
    (Join-Path $RepoRoot "scripts\inspect_kaggle_fog_zip.py") `
    (Join-Path $RepoRoot "scripts\estimate_kaggle_fog_storage.py") `
    (Join-Path $RepoRoot "scripts\preprocess_kaggle_fog_streaming.py") `
    (Join-Path $RepoRoot "scripts\kaggle_fog_status.py") `
    (Join-Path $RepoRoot "scripts\run_fog_experiment.py") `
    (Join-Path $RepoRoot "scripts\run_fog_suite.py") `
    (Join-Path $RepoRoot "scripts\preflight_fog_suite.py") `
    (Join-Path $RepoRoot "scripts\start_kaggle_full_pipeline.py") `
    (Join-Path $RepoRoot "scripts\start_kaggle_smoke_pipeline.py") `
    (Join-Path $RepoRoot "scripts\check_processed_pipeline.py") `
    (Join-Path $RepoRoot "scripts\prepare_processed_record_windows.py") `
    (Join-Path $RepoRoot "scripts\validate_processed_records.py")
}

Invoke-Step "Inspect zip central directory only" {
  python (Join-Path $RepoRoot "scripts\inspect_kaggle_fog_zip.py") --dataset-root $DatasetRoot
}

Invoke-Step "Validate zip supervised structure" {
  $Script:ZipStructureReport = Get-ZipStructureReport -KaggleDir $KaggleDir
  Write-ZipStructureReport -Report $Script:ZipStructureReport
  if (-not $Script:ZipStructureReport.ok) {
    throw "Kaggle zip supervised structure check failed."
  }
}

Invoke-Step "Estimate supervised storage budget" {
  $StorageArgs = @(
    (Join-Path $RepoRoot "scripts\estimate_kaggle_fog_storage.py"),
    "--dataset-root", $DatasetRoot,
    "--source", "both",
    "--suite-config", $SuiteConfig,
    "--reserve-gib", ([string]$ReserveGib),
    "--smoke-limit", ([string]$SmokeLimit)
  )
  if (-not $AllowInsufficientStorage) {
    $StorageArgs += "--fail-if-insufficient"
  }
  if ($OutputJson -ne "") {
    $StorageArgs += @("--output-json", $StorageReportPath)
  }
  python @StorageArgs
}

Invoke-Step "Streaming dry-run only" {
  $StreamingArgs = @(
    (Join-Path $RepoRoot "scripts\preprocess_kaggle_fog_streaming.py"),
    "--dataset-root", $DatasetRoot,
    "--source", "both",
    "--valid-only",
    "--task-only",
    "--check-headers",
    "--strict-metadata",
    "--smoke-limit", ([string]$SmokeLimit),
    "--dry-run"
  )
  if ($OutputJson -ne "") {
    $StreamingArgs += @("--dry-run-output-json", $StreamingReportPath)
  }
  python @StreamingArgs
}

Invoke-Step "Kaggle suite config dry-run" {
  python (Join-Path $RepoRoot "scripts\run_fog_suite.py") `
    --config $SuiteConfig `
    --dry-run `
    --skip-collection `
    --validate-experiment-configs
}

Invoke-Step "FOG suite preflight before processed" {
  $SuitePreflightArgs = @(
    (Join-Path $RepoRoot "scripts\preflight_fog_suite.py"),
    "--config", $SuiteConfig,
    "--dataset-root", $DatasetRoot,
    "--allow-missing-processed"
  )
  if ($OutputJson -ne "") {
    $SuitePreflightArgs += @("--output-json", $SuitePreflightReportPath)
  }
  python @SuitePreflightArgs
}

if (-not $SkipPytest) {
  Invoke-Step "Synthetic Kaggle tests" {
    python -m pytest (Join-Path $RepoRoot "tests\test_kaggle_streaming_preprocess.py") -q --basetemp $PytestBaseTemp
  }
}

$HasProcessedAfter = Test-Path -LiteralPath $ProcessedPath
$HasSmokeAfter = Test-Path -LiteralPath $SmokePath
Write-Host ""
Write-Host "processed exists after: $HasProcessedAfter"
Write-Host "processed_smoke exists after: $HasSmokeAfter"

if ((-not $HadProcessed) -and $HasProcessedAfter) {
  throw "Preflight created processed unexpectedly: $ProcessedPath"
}
if ((-not $HadSmoke) -and $HasSmokeAfter) {
  throw "Preflight created processed_smoke unexpectedly: $SmokePath"
}

Write-PreflightReport -Status "passed"
Write-Host ""
Write-Host "Kaggle FOG preflight passed without creating processed records."
}
catch {
  Write-PreflightReport -Status "failed" -ErrorRecord $_
  throw
}
