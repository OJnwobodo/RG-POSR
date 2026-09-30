@echo off
setlocal

set "WORK=G:\ONYEKA_NEW_EXPERIMENT\Onyeka_Review_Experiment\02_reliability_models"
set "PROCESSED=G:\ONYEKA\who-is-alyx-processed"
set "FROZEN=G:\ONYEKA_NEW_EXPERIMENT\RG_POSR_FROZEN_v1"
set "BLIND=%WORK%\review_blind_cache"

cd /d "%WORK%" || exit /b 1

call :run_variant uniform uniform
if errorlevel 1 goto :failed

call :run_variant direct direct_active_fraction
if errorlevel 1 goto :failed

call :run_variant linear linear
if errorlevel 1 goto :failed

call :run_variant nonlinear nonlinear
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo ALL FOUR PROTOCOL-COMPLETE SEED-72 RUNS FINISHED
echo ============================================================
exit /b 0

:run_variant
set "LABEL=%~1"
set "VARIANT=%~2"
set "OUT=%WORK%\protocol_seed72_%LABEL%"

if exist "%OUT%\experiment_report.json" (
    echo ============================================================
    echo SEED-72 %LABEL% ALREADY COMPLETE - SKIPPING
    echo ============================================================
    exit /b 0
)

echo ============================================================
echo Protocol-complete seed-72 validation: %LABEL%
echo ============================================================

python train_review_rg_posr.py ^
  --processed-root "%PROCESSED%" ^
  --frozen-root "%FROZEN%" ^
  --blind-cache-root "%BLIND%" ^
  --output "%OUT%" ^
  --reliability-variant %VARIANT% ^
  --seeds 72 ^
  --folds 0 1 2 3 4 ^
  --aggregation-seconds 30 60 120 ^
  --gamma-candidates 0 0.5 1 2 ^
  --epochs 30 ^
  --batch-size 64 ^
  --learning-rate 0.001 ^
  --weight-decay 0.0001 ^
  --patience 8 ^
  --corruption-probability 0.6 ^
  --device auto

if errorlevel 1 exit /b 1
exit /b 0

:failed
echo.
echo ============================================================
echo SEED-72 RUN FAILED OR WAS INTERRUPTED.
echo Re-run this same BAT file to resume/continue.
echo Completed variants will be skipped automatically.
echo ============================================================
exit /b 1
