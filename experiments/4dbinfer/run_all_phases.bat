@echo off
echo ============================================================
echo 4DBInfer Stage 11 - Complete Reproduction Script (Windows)
echo ============================================================

echo.
echo Phase A: Baseline Verification
echo --------------------------------
python run_baseline_smoke_v2.py
if %ERRORLEVEL% neq 0 (
    echo FAILED: Phase A baseline verification
    exit /b 1
)

echo.
echo Phase B: Integration Mapping
echo -----------------------------
echo Phase B completed (documentation in hhgt_integration_plan.md)

echo.
echo Phase C: Adapter Implementation  
echo --------------------------------
python -m pytest test_4dbinfer_adapter.py -v
if %ERRORLEVEL% neq 0 (
    echo FAILED: Phase C adapter tests
    exit /b 1
)

echo.
echo Phase D: Integration Evaluation
echo --------------------------------
python run_integration_eval.py
if %ERRORLEVEL% neq 0 (
    echo FAILED: Phase D integration evaluation
    exit /b 1
)

echo.
echo Phase E: Ablation Studies
echo -------------------------
python run_ablation_studies.py
if %ERRORLEVEL% neq 0 (
    echo FAILED: Phase E ablation studies
    exit /b 1
)

echo.
echo ============================================================
echo ✅ All phases completed successfully!
echo 📁 Check output directories for results
echo ============================================================
