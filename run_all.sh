#!/usr/bin/env bash
set -euo pipefail
MANIFEST=${1:?Usage: bash run_all.sh FEATURE_MANIFEST SPLIT_PREFIX}
SPLIT=${2:?Usage: bash run_all.sh FEATURE_MANIFEST SPLIT_PREFIX}
python -m experiments.run_five_seeds --manifest "$MANIFEST" --split-prefix "$SPLIT" --out outputs/main
python -m experiments.run_ablations --manifest "$MANIFEST" --split-prefix "$SPLIT" --out outputs/ablations
mkdir -p results
python -m results.generate_tables --root outputs/main --out results/five_seed_summary.csv
python -m experiments.statistics --root outputs/main --out results/statistics.csv
python -m analysis.plot_roc --root outputs/main --out results/roc.png
python -m analysis.plot_ablation --root outputs/ablations --out results/ablation.png
python -m experiments.edge_benchmark > results/edge_benchmark.json
