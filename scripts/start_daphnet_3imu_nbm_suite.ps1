$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$PythonExe = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$Launcher = Join-Path $PSScriptRoot "start_daphnet_3imu_nbm_suite.py"
$env:PYTHONUNBUFFERED = "1"

& $PythonExe -u $Launcher @args
exit $LASTEXITCODE
