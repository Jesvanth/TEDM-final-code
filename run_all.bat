@echo off
if "%~2"=="" (echo Usage: run_all.bat FEATURE_MANIFEST SPLIT_PREFIX & exit /b 1)
python -m experiments.run_five_seeds --manifest "%~1" --split-prefix "%~2" --out outputs/main || exit /b 1
python -m experiments.run_ablations --manifest "%~1" --split-prefix "%~2" --out outputs/ablations || exit /b 1
if not exist results mkdir results
python -m results.generate_tables --root outputs/main --out results/five_seed_summary.csv
python -m experiments.statistics --root outputs/main --out results/statistics.csv
python -m analysis.plot_roc --root outputs/main --out results/roc.png
python -m analysis.plot_ablation --root outputs/ablations --out results/ablation.png
python -m experiments.edge_benchmark > results/edge_benchmark.json
