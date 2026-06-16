param(
    [Parameter(Mandatory = $true)]
    [int]$TrainPid
)

$ErrorActionPreference = "Continue"
$Repo = "E:\fog"
$DataDir = "dataset\processed\fog_patch_blocks_seq128_prefog2"
$OutDir = "outputs\patch_transformer_prefog2_mid_loss_1_4_2"
$Folds = "1,2,3,4,5,6,7,8,10,15,16,17,19"
$LogDir = Join-Path $Repo "outputs\logs"
$OutLog = Join-Path $LogDir "patch_transformer_prefog2_mid_loss_1_4_2.threshold.out.log"
$ErrLog = Join-Path $LogDir "patch_transformer_prefog2_mid_loss_1_4_2.threshold.err.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $Repo
$env:PYTHONUNBUFFERED = "1"

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] waiting for training PID $TrainPid" | Add-Content -Path $OutLog
try {
    Wait-Process -Id $TrainPid -ErrorAction Stop
} catch {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] wait failed: $($_.Exception.Message)" | Add-Content -Path $ErrLog
    exit 1
}

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] training finished; start threshold evaluation" | Add-Content -Path $OutLog
& python -u scripts\evaluate_patch_transformer_thresholds.py `
    --data-dir $DataDir `
    --experiment-dir $OutDir `
    --folds $Folds `
    --batch-size 256 `
    --device cuda `
    --amp >> $OutLog 2>> $ErrLog

$ExitCode = $LASTEXITCODE
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] threshold evaluation done; exit=$ExitCode" | Add-Content -Path $OutLog
exit $ExitCode
