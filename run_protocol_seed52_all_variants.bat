@echo off
setlocal

set "WORK=G:\ONYEKA_NEW_EXPERIMENT\Onyeka_Review_Experiment\02_reliability_models"
set "PROCESSED=G:\ONYEKA\who-is-alyx-processed"
set "FROZEN=G:\ONYEKA_NEW_EXPERIMENT\RG_POSR_FROZEN_v1"
set "BLIND=%WORK%\review_blind_cache"

cd /d "%WORK%" || exit /b 1

echo ============================================================
echo Protocol-complete seed-52 validation: UNIFORM
echo ============================================================
python train_review_rg_posr.py ^
  --processed-root "%PROCESSED%" ^
  --frozen-root "%FROZEN%" ^
  --blind-cache-root "%BLIND%" ^
  --output "%WORK%\protocol_seed52_uniform" ^
  --reliability-variant uniform ^
  --seeds 52 ^
  --folds 0 1 2 3 4 ^
  --aggregation-seconds 30 60 120 ^
  --gamma-candidates 0 0.5 1 2 ^
  --epochs 30 ^
  --batch-size 64 ^
  --learning-rate 0.001 ^
  --weight-decay 0.0001 ^
  --patience 8 ^
  --corruption-probability 0.6 ^
  --device auto ^
  --overwrite
if errorlevel 1 goto :failed

echo ============================================================
echo Protocol-complete seed-52 validation: DIRECT
echo ============================================================
python train_review_rg_posr.py ^
  --processed-root "%PROCESSED%" ^
  --frozen-root "%FROZEN%" ^
  --blind-cache-root "%BLIND%" ^
  --output "%WORK%\protocol_seed52_direct" ^
  --reliability-variant direct_active_fraction ^
  --seeds 52 ^
  --folds 0 1 2 3 4 ^
  --aggregation-seconds 30 60 120 ^
  --gamma-candidates 0 0.5 1 2 ^
  --epochs 30 ^
  --batch-size 64 ^
  --learning-rate 0.001 ^
  --weight-decay 0.0001 ^
  --patience 8 ^
  --corruption-probability 0.6 ^
  --device auto ^
  --overwrite
if errorlevel 1 goto :failed

echo ============================================================
echo Protocol-complete seed-52 validation: LINEAR
echo ============================================================
python train_review_rg_posr.py ^
  --processed-root "%PROCESSED%" ^
  --frozen-root "%FROZEN%" ^
  --blind-cache-root "%BLIND%" ^
  --output "%WORK%\protocol_seed52_linear" ^
  --reliability-variant linear ^
  --seeds 52 ^
  --folds 0 1 2 3 4 ^
  --aggregation-seconds 30 60 120 ^
  --gamma-candidates 0 0.5 1 2 ^
  --epochs 30 ^
  --batch-size 64 ^
  --learning-rate 0.001 ^
  --weight-decay 0.0001 ^
  --patience 8 ^
  --corruption-probability 0.6 ^
  --device auto ^
  --overwrite
if errorlevel 1 goto :failed

echo ============================================================
echo Protocol-complete seed-52 validation: NONLINEAR
echo ============================================================
python train_review_rg_posr.py ^
  --processed-root "%PROCESSED%" ^
  --frozen-root "%FROZEN%" ^
  --blind-cache-root "%BLIND%" ^
  --output "%WORK%\protocol_seed52_nonlinear" ^
  --reliability-variant nonlinear ^
  --seeds 52 ^
  --folds 0 1 2 3 4 ^
  --aggregation-seconds 30 60 120 ^
  --gamma-candidates 0 0.5 1 2 ^
  --epochs 30 ^
  --batch-size 64 ^
  --learning-rate 0.001 ^
  --weight-decay 0.0001 ^
  --patience 8 ^
  --corruption-probability 0.6 ^
  --device auto ^
  --overwrite
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo ALL FOUR PROTOCOL-COMPLETE SEED-52 RUNS FINISHED
echo ============================================================
exit /b 0

:failed
echo.
echo ============================================================
echo SEED-52 RUN FAILED. Check the error above.
echo Later variants were not run.
echo ============================================================
exit /b 1
