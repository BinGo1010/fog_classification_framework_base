$ErrorActionPreference = "Stop"

$Repo = "E:\fog"
$DataDir = "dataset\processed\fogstar_loso_activity1_notask2_7_3class_pre_fog5p0s_win90"
$OutDir = "outputs\sleepyco_fogstar_win90_two_stage_10fold_prebs1024"
$LogDir = Join-Path $Repo "outputs\logs"
$LogPath = Join-Path $LogDir "sleepyco_fogstar_win90_two_stage_10fold_prebs1024.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $Repo
$env:PYTHONUNBUFFERED = "1"

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] start sleepyco fogstar win90 two-stage training prebs1024" | Tee-Object -FilePath $LogPath -Append

$ArgsList = @(
    "scripts\run_sleepyco_fog_two_stage.py",
    "--data-dir", $DataDir,
    "--output-dir", $OutDir,
    "--stage", "both",
    "--folds", "all",
    "--baselines", "seq2one_gru,seq2seq_gru",
    "--seq-len", "5",
    "--seq-stride", "1",
    "--target-position", "center",
    "--pretrain-epochs", "30",
    "--finetune-epochs", "40",
    "--pretrain-patience", "8",
    "--finetune-patience", "10",
    "--pretrain-batch-size", "1024",
    "--finetune-batch-size", "128",
    "--feature-dim", "128",
    "--projection-dim", "128",
    "--num-scales", "3",
    "--hidden-dim", "128",
    "--num-layers", "1",
    "--dropout", "0.2",
    "--gru-pool", "attn",
    "--loss", "focal",
    "--focal-gamma", "2.0",
    "--class-weight", "balanced",
    "--sampler", "manual",
    "--sampler-class-weights", "1,4,2",
    "--samples-per-epoch", "auto",
    "--num-workers", "0",
    "--device", "auto",
    "--amp",
    "--resume"
)

& python -u @ArgsList 2>&1 | Tee-Object -FilePath $LogPath -Append

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] finished sleepyco fogstar win90 two-stage training prebs1024" | Tee-Object -FilePath $LogPath -Append
