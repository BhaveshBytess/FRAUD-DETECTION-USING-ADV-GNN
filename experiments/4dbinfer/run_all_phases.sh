#!/bin/bash
echo "============================================================"
echo "4DBInfer Stage 11 - Complete Reproduction Script (Unix)"  
echo "============================================================"

echo ""
echo "Phase A: Baseline Verification"
echo "--------------------------------"
python run_baseline_smoke_v2.py
if [ $? -ne 0 ]; then
    echo "FAILED: Phase A baseline verification"
    exit 1
fi

echo ""
echo "Phase B: Integration Mapping"
echo "-----------------------------"
echo "Phase B completed (documentation in hhgt_integration_plan.md)"

echo ""
echo "Phase C: Adapter Implementation"
echo "--------------------------------"
python -m pytest test_4dbinfer_adapter.py -v
if [ $? -ne 0 ]; then
    echo "FAILED: Phase C adapter tests"
    exit 1
fi

echo ""
echo "Phase D: Integration Evaluation"
echo "--------------------------------"
python run_integration_eval.py
if [ $? -ne 0 ]; then
    echo "FAILED: Phase D integration evaluation"
    exit 1
fi

echo ""
echo "Phase E: Ablation Studies"
echo "-------------------------"
python run_ablation_studies.py
if [ $? -ne 0 ]; then
    echo "FAILED: Phase E ablation studies"
    exit 1
fi

echo ""
echo "============================================================"
echo "All phases completed successfully!"
echo "Check output directories for results"
echo "============================================================"
