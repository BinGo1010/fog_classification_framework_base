$ErrorActionPreference = "Stop"

$Repo = "E:\fog"
$WaitPid = 10868
$DataDir = "dataset\processed\fog_patch_blocks_seq128_prefog2"
$OutDir = "outputs\patch_transformer_prefog2_mid_loss_1_4_2"
$LogDir = Join-Path $Repo "outputs\logs"
$LogPath = Join-Path $LogDir "patch_transformer_prefog2_mid_loss_1_4_2.log"
$Folds = "1,2,3,4,5,6,7,8,10,15,16,17,19"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $Repo
$env:PYTHONUNBUFFERED = "1"

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] queued patch_transformer PRE_FOG=2s mid_loss_1_4_2" | Tee-Object -FilePath $LogPath -Append

if ($WaitPid -gt 0) {
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] waiting for PID $WaitPid to finish before using GPU" | Tee-Object -FilePath $LogPath -Append
        Start-Sleep -Seconds 120
    }
}

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] start training" | Tee-Object -FilePath $LogPath -Append

& python -u scripts\run_patch_transformer_loso.py `
    --data-dir $DataDir `
    --output-dir $OutDir `
    --folds $Folds `
    --val-strategy eventful `
    --epochs 15 `
    --patience 5 `
    --batch-size 128 `
    --lr 0.0008 `
    --weight-decay 0.0001 `
    --d-model 128 `
    --num-heads 4 `
    --encoder-layers 2 `
    --lstm-layers 2 `
    --dropout 0.15 `
    --roll-pos-encoding `
    --loss-class-weights "1,4,2" `
    --sampler manual `
    --sampler-class-weights "1,8,2" `
    --sampler-power 0.75 `
    --samples-per-epoch auto `
    --num-workers 0 `
    --seed 42 `
    --amp `
    --resume `
    --device cuda 2>&1 | Tee-Object -FilePath $LogPath -Append

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] start threshold evaluation" | Tee-Object -FilePath $LogPath -Append

& python -u scripts\evaluate_patch_transformer_thresholds.py `
    --data-dir $DataDir `
    --experiment-dir $OutDir `
    --folds $Folds `
    --batch-size 256 `
    --device cuda `
    --amp 2>&1 | Tee-Object -FilePath $LogPath -Append

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] finished patch_transformer PRE_FOG=2s mid_loss_1_4_2" | Tee-Object -FilePath $LogPath -Append
