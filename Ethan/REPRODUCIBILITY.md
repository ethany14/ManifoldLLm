# EmoBank / ManifoldLLm Reproducibility Guide

This note documents the exact data-cleaning, embedding, and evaluation setup used for
the current full-data results. Share it with collaborators before comparing scores.

## 1. Data source

Use the official EmoBank CSV from the JULIE Lab repository:

`https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv`

Save the unchanged file as:

`data/source/emobank_official.csv`

Expected SHA-256:

`1ADE4A4A453E880C0F39D0536D2B355E8A716E0CF88236C64CDB4D438CF9605B`

EmoBank is licensed under CC BY-SA 4.0. Cite Buechel and Hahn (2017).

## 2. Environment

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The completed run used Python 3.13.7, PyTorch 2.14.0+cu130,
Transformers 5.17.0, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1,
and SciPy 1.18.1. Exact versions are useful for replication, although the repository
requirements specify compatible minimum versions.

## 3. Cleaning and normalization

Run:

```powershell
.\.venv\Scripts\python.exe .\prepare_emobank.py `
  --input .\data\source\emobank_official.csv `
  --output-dir .\data\emobank
```

`prepare_emobank.py` performs only the following operations:

1. Requires the official `id`, `split`, `V`, `A`, `D`, and `text` columns.
2. Rejects missing annotation fields and unexpected split names.
3. Checks that official V/A/D scores are within 1-5.
4. Removes one row whose text is missing.
5. Preserves the official `train`, `dev`, and `test` assignments.
6. Normalizes every VAD score using `normalized = (score - 3) / 2`, producing
   values in [-1, 1].
7. Preserves row order and writes a fixed schema:
   `id,text,valence,arousal,dominance,split,source`.

Expected full output:

- 10,061 usable rows
- 8,062 train
- 999 dev
- 1,000 test

Expected SHA-256 for `data/emobank/emobank_manifold.csv`:

`AD0A57A2C4E0C01281B5A8D29557B54E515B2470D2D2E2623B428F96B6E6D653`

The pilot file is sampled separately within each official split using
`random_state=42`: 800 train, 100 dev, and 100 test rows. It must not replace the
full dataset for the final result.

## 4. Frozen Qwen embeddings

Run the same model, layer, pooling, and maximum token length:

```powershell
.\.venv\Scripts\python.exe .\extract_embeddings.py `
  --csv .\data\emobank\emobank_manifold.csv `
  --output .\artifacts\emobank_full\embeddings.npy `
  --model Qwen/Qwen3-4B `
  --layer -1 `
  --batch-size 1 `
  --max-length 256
```

The LLM is frozen: evaluation mode, inference mode, and gradients disabled. The script
uses masked mean pooling over the final hidden layer. Padding tokens are excluded.
The resulting matrix has shape `(10061, 2560)` and must remain row-aligned with
`emobank_manifold.csv`.

The cached model snapshot used here is revision:

`1cfa9a7208912126459214e8b04321603b3df60c`

Expected embedding SHA-256:

`BA5C8DD951EA54EA6F2D9D6947F8B36D5C1A4369684DF6DD7F89DF5DB31C848C`

If this hash differs, record the model revision, package versions, GPU precision, and
pooling settings before comparing downstream numbers.

## 5. Full ablation experiment

Run:

```powershell
.\.venv\Scripts\python.exe .\run_ablation_study.py `
  --csv .\data\emobank\emobank_manifold.csv `
  --embeddings .\artifacts\emobank_full\embeddings.npy `
  --output-dir .\artifacts\emobank_full\ablation_seed_42 `
  --dimensions 16 `
  --epochs 80 `
  --hidden-dim 256 `
  --batch-size 128 `
  --geometry-weight 0.1 `
  --reconstruction-weight 0.1 `
  --seed 42
```

The script compares:

- supervised linear bottleneck;
- supervised nonlinear bottleneck;
- supervised nonlinear bottleneck plus geometry loss;
- nonlinear autoencoder plus VAD supervision;
- nonlinear autoencoder plus VAD supervision and geometry loss.

Every representation has 16 dimensions and is evaluated using the same
`Ridge(alpha=10)` probe. Standardization is fitted on training rows only. Development
data controls early stopping, and the official test split is used only for final
metrics.

## 6. Expected full-data result (seed 42)

| Method | Valence R2 | Arousal R2 | Dominance R2 | Mean VAD R2 |
|---|---:|---:|---:|---:|
| AE + VAD + Geometry | 0.526 | 0.198 | 0.184 | 0.302 |
| AE + VAD | 0.519 | 0.199 | 0.185 | 0.301 |
| Supervised Nonlinear | 0.507 | 0.173 | 0.159 | 0.280 |
| Supervised Nonlinear + Geometry | 0.506 | 0.177 | 0.147 | 0.277 |
| Supervised Linear | 0.481 | 0.154 | 0.089 | 0.241 |

Minor floating-point differences may occur across GPU and library versions. Large
differences usually indicate a changed model revision, pooling method, row order,
normalization, split, or random seed.

## 7. Pre-comparison checklist

Before combining results, verify that every collaborator used:

- the same official source-file hash;
- the same 10,061-row processed CSV hash;
- the unchanged official split column;
- VAD normalized to [-1, 1] with the documented formula;
- Qwen/Qwen3-4B, final hidden layer, masked mean pooling, and max length 256;
- embeddings in exactly the same row order as the CSV;
- train-only fitted scalers and representation models;
- seed 42 for the directly comparable run;
- the independent 1,000-row official test set only for final evaluation.

Do not compare a pilot result against a full-data result, and do not select model
hyperparameters using test performance.

## 8. Team result folders

Each contributor should copy `team_results/template` to
`team_results/<github_username>` and commit only compact result files, configuration,
and notes. Do not commit model weights, cached LLM files, virtual environments, or
full embedding arrays.
