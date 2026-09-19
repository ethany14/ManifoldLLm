# Frozen-LLM Nonlinear Affective Manifold Adapter

This prototype separates two jobs:

1. A **frozen language model** measures each text as a contextual hidden-state vector.
2. **Isomap** discovers nonlinear affective geometry in those vectors.
3. A small **nonlinear adapter** learns the Isomap coordinates so that unseen text can be
   placed on the same manifold without rerunning Isomap.

Only the adapter is trained. The language model is always in evaluation mode, runs under
`torch.inference_mode()`, and has gradients disabled.

## Data format

Use a CSV containing `text`. For the scientifically useful configuration, also include:

- `valence` and `arousal`, scaled consistently (for example, -1 to 1)
- `emotion`, such as joy, fear, sadness, anger, surprise, or neutral
- optional `sequence_id` and `time` for later trajectory evaluation

`data/example_emotions.csv` illustrates the schema but is too small for real training.
Use at least several thousand independently sourced examples and keep speakers or
conversations separated between training and evaluation.

## Included research dataset: EmoBank

`data/source/emobank_official.csv` is the unchanged official EmoBank file. The
reproducible `prepare_emobank.py` conversion creates:

- `data/emobank/emobank_manifold.csv`: 10,061 usable sentences
- `data/emobank/emobank_pilot_1000.csv`: fixed 800/100/100 pilot split
- separate train, dev, and test files
- `dataset_metadata.json`: source, license, counts, and score transformation

EmoBank's 1-5 valence, arousal, and dominance ratings are mapped to -1 to 1 with
`normalized = (official_score - 3) / 2`. The official split is retained. One official
record with missing text is excluded and documented in the metadata. Cite Buechel and
Hahn (2017); the dataset is CC BY-SA 4.0.

## Installation

Create an environment with Python 3.10-3.12, then install:

```powershell
pip install -r requirements.txt
```

## Run the pipeline

### 1. Extract frozen Qwen states

```powershell
python extract_embeddings.py `
  --csv your_emotions.csv `
  --output artifacts/embeddings.npy `
  --model Qwen/Qwen3-4B `
  --layer -1
```

This downloads the model on first use. This installation defaults to Qwen3-4B because
the local GPU has 12 GB of memory. Qwen3-8B remains appropriate on a larger GPU. Do not
quantize the primary scientific run; quantization may change geometry.

## This computer

The project-local environment is located at `.venv`. From this directory, run
`run_adapter.ps1`; the launcher selects the correct Python environment automatically.
Downloaded Hugging Face models are kept under `artifacts/hf_cache`, not in the user's
global profile.

### 2. Discover the nonlinear manifold

```powershell
python discover_manifold.py `
  --embeddings artifacts/embeddings.npy `
  --csv data/emobank/emobank_manifold.csv `
  --output-dir artifacts/manifold `
  --dimensions 2 `
  --neighbors 12
```

Repeat with several neighbour counts and report trustworthiness. PCA should be fitted as
a linear baseline in the final experiment.

To compare PCA and Isomap configurations without retraining Qwen:

```powershell
.\.venv\Scripts\python.exe benchmark_manifolds.py `
  --csv data/emobank/emobank_pilot_1000.csv `
  --embeddings artifacts/emobank_pilot/embeddings.npy `
  --output-dir artifacts/emobank_pilot/benchmark
```

### 3. Train only the adapter

```powershell
python train_adapter.py `
  --csv your_emotions.csv `
  --embeddings artifacts/embeddings.npy `
  --coordinates artifacts/manifold/coordinates.npy `
  --output-dir artifacts/model
```

The total objective is:

```text
coordinate reconstruction
+ local distance preservation
+ valence/arousal supervision
+ emotion-class supervision
```

The label heads are auxiliary: the central output is the continuous manifold coordinate.

### 4. Map new examples

Extract the new texts with exactly the same LLM and layer, then run:

```powershell
python predict.py `
  --checkpoint artifacts/model/adapter.pt `
  --embeddings artifacts/new_embeddings.npy `
  --output artifacts/predictions.json
```

## What must be validated before making a research claim

- Compare Isomap against PCA and direct valence/arousal regression.
- Choose layer and neighbour count using validation data, never the test set.
- Split by speaker/conversation to prevent linguistic leakage.
- Report neighbourhood trustworthiness and continuity, affect prediction error, emotion
  accuracy/F1, and trajectory smoothness for longitudinal dialogue.
- Repeat with a second frozen model to test whether the finding is model-specific.
- Treat a 2D plot as a visualization, not proof that emotion is intrinsically 2D.

The adapter is an out-of-sample extension of the discovered geometry. It does not prove
that the Isomap coordinates are the unique or biologically correct structure of emotion.
