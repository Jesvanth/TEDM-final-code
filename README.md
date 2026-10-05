# TEDM — Temporal Emotion Drift Model

Reproducibility package for **Detecting Multimodal Sarcasm through Temporal Emotion Drift Modelling**.

## Scientific configuration
This repository implements the configuration used for the manuscript revision: `bert-base-uncased` (768-D text), ResNet-18 (512-D visual), 40 MFCC coefficients at 16 kHz with FFT=512, 25 ms window and 10 ms hop, 256-D shared latent space, one 256-unit GRU layer, two Transformer encoder layers, four attention heads, FFN=512, dropout=0.1, Adam LR=1e-4, batch size 8, maximum 50 epochs, cosine annealing, patience 5, and seeds 11/22/33/44/55.

### Acoustic dimensionality
The extractor computes **40 MFCC coefficients** and summarizes each coefficient over time with its **mean and standard deviation**, yielding an **80-D acoustic vector**. Thus `n_mfcc=40` and `audio_dim=80` are consistent, not contradictory.

### Classifier/loss
The network returns one logit. Training uses `BCEWithLogitsLoss`; sigmoid is applied to logits only to obtain probabilities. This is numerically equivalent to sigmoid + binary cross-entropy while being more stable.

## Important parameter-count clarification
BERT-base-uncased alone has far more parameters than the manuscript's stated 50.0M complete-system value, while the trainable TEDM fusion network operating on pre-extracted embeddings is much smaller. Therefore this repository **does not hard-code or fabricate a 50.0M count**. `experiments/edge_benchmark.py` computes the parameter count, FP32 size, latency and throughput of the actual deposited fusion model. If the manuscript currently states 50.0M/200 MB as the parameter count of this exact implementation, that statement must be reconciled with the measured deposited code before publication.

## Baseline provenance
Text-BERT, MISA, MMBT and Early Fusion Transformer in this repository are **author implementations for controlled comparison** based on the corresponding architectural ideas. MISA/MMBT are not claimed to be byte-for-byte copies of the original authors' source repositories.

## Data
MUStARD and MUStARD++ are not redistributed. Obtain them under their original terms. Prepare a CSV with:
`dialogue_id,turn_index,text,audio_path,video_path,label`.

## Reproduction
```bash
pip install -r requirements.txt
python scripts/verify_environment.py
python -m features.extract_features --manifest data/mustard_manifest.csv --out data/features/mustard
python data/make_fixed_split.py --manifest data/features/mustard/feature_manifest.csv --out-prefix data/splits/mustard
python -m experiments.run_five_seeds --manifest data/features/mustard/feature_manifest.csv --split-prefix data/splits/mustard --out outputs/mustard
python -m experiments.run_ablations --manifest data/features/mustard/feature_manifest.csv --split-prefix data/splits/mustard --out outputs/mustard_ablations
python -m results.generate_tables --root outputs/mustard --out results/mustard_table5.csv
python -m experiments.statistics --root outputs/mustard --out results/mustard_statistics.csv
python -m analysis.plot_roc --root outputs/mustard --out results/mustard_roc.png
python -m analysis.plot_ablation --root outputs/mustard_ablations --out results/mustard_ablation.png
python -m experiments.edge_benchmark
```
Repeat with the MUStARD++ manifest/split prefix.

## Fixed split + five seeds
Create the split once and preserve the generated ID files. All five training seeds reuse the same train/validation/test IDs; the seeds change training randomness, not the partition.

## Code availability
Archive the final repository in a DOI-assigning repository after verification. Insert the assigned DOI into the manuscript only after it exists.
