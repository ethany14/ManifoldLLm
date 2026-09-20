# Digital Personality Manifold - Ethan

This folder contains the current person-level method and its exploratory evaluation. The research question is **how to use manifold geometry to form a revisable digital-personality state**, not simply whether a nonlinear embedding improves one score.

## Current answer and status

For each participant, a causal encoder reads ordered observations and estimates a slowly changing person posterior `q(phi | history)` and a fast state `q(z | current observation, phi)`. A nonlinear decoder `g(phi, z)` reconstructs observable cues. On a prespecified fast-state slice, the decoder Jacobian defines the local person-space metric

```text
G(phi) = J_phi g(phi, 0)^T J_phi g(phi, 0) + 1e-4 I
```

Geometry becomes operational when decoded path length is used for person similarity or inside the same-person contrastive training loss. The current implementation uses a sampled straight coordinate path, **not an optimized geodesic**. It does not use Isomap coordinates as targets. The model can re-estimate `phi` and `z` after a new observation.

The method is implemented, but the current development results do **not** establish that nonlinear geometry beats a matched Euclidean identity objective, or that `phi` carries personality information beyond simple valence mean and variability. No LLM persona is trained here. See the [current method report](output/pdf/PSYCHARCHIVES_CORE_METHOD_REPORT_EN.pdf) for the literature lineage, exact interpretation, and next-stage roadmap.

## Current dataset and boundaries

The licensed PsychArchives smartphone-sensing panel links 9,790 repeated questionnaire snapshots from 455 people to person-level variables. The scripts read `Codebook.csv`, `person_variables.csv`, and `es_sensing_variables.csv` from the user's `Downloads` folder by default; use `--source-dir` if your licensed files live elsewhere. Do **not** copy these files into Git or share participant-level rows or coordinates without checking the Scientific Use License.

The fixed split is by person: 318 train, 68 development, and 69 reserved test people. The current scripts use 21 sensing/time cues plus valence, ordered by questionnaire ID. The target is valence at the next questionnaire, **not** a known number of hours later. The reserved test people have not been scored in the reported PsychArchives comparisons. Preprocessing statistics are fitted on training people.

Source: [PsychArchives dataset record](https://psycharchives.org/en/item/975a7624-cd16-46bb-9d80-d988bed5e780).

## Setup

Run from the repository root in PowerShell. Use the existing `.venv` if available, or create one:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Ethan\requirements.txt
```

The training scripts use PyTorch, pandas, NumPy, and scikit-learn. The optional PDF builder additionally needs ReportLab:

```powershell
.\.venv\Scripts\python.exe -m pip install reportlab
```

## Reproduce the current method checks

Run these from the repository root with your licensed files available. They are separate development analyses, not a single pipeline that should be tuned repeatedly against the same people.

```powershell
# Ordered, missingness-aware slow/fast model and online inference interface
.\.venv\Scripts\python.exe Ethan\run_psycharchives_sequence_manifold.py

# Fixed personality readouts and component removals
.\.venv\Scripts\python.exe Ethan\run_psycharchives_evaluation_12.py

# Matched Euclidean-identity versus corrected decoder-path-identity training
.\.venv\Scripts\python.exe Ethan\run_psycharchives_geometry_training.py

# Cue-group sensitivity and post-hoc non-neural valence summaries
.\.venv\Scripts\python.exe Ethan\run_psycharchives_feature_groups.py
.\.venv\Scripts\python.exe Ethan\run_psycharchives_valence_summary_baselines.py
```

Basic implementation checks:

```powershell
.\.venv\Scripts\python.exe Ethan\test_psycharchives_sequence_manifold.py -v
.\.venv\Scripts\python.exe Ethan\test_psycharchives_geometry_training.py -v
```

The scripts record aggregate results under ignored `Ethan/artifacts/psycharchives/`. They do not publish licensed source rows. Do not open the reserved test split for model selection.

## Read the evidence in order

1. [Model and data preparation](PSYCHARCHIVES_MANIFOLD_IMPLEMENTATION.md)
2. [Trait, window-matching, and component checks](PSYCHARCHIVES_EVALUATION_12_RESULTS.md)
3. [Corrected geometry-training comparison](PSYCHARCHIVES_GEOMETRY_TRAINING_RESULTS.md)
4. [Feature-group sensitivity and simple baseline](PSYCHARCHIVES_FEATURE_GROUP_RESULTS.md)
5. [Supervisor report focused on the HOW](output/pdf/PSYCHARCHIVES_CORE_METHOD_REPORT_EN.pdf)

The corresponding `*_PROTOCOL.md` files record fixed design choices and corrections. In particular, `PSYCHARCHIVES_GEOMETRY_TRAINING_PROTOCOL.md` is a superseded numerical pilot; the corrected comparison uses `PSYCHARCHIVES_GEOMETRY_TRAINING_PROTOCOL_V2.md`.

Regenerate the PDF without re-running or opening licensed data:

```powershell
.\.venv\Scripts\python.exe Ethan\build_psycharchives_core_report.py
```

## What is still needed

- Define an independent personality-relevant later behavior target, not only person matching or next-valence prediction.
- Audit whether sensing/missingness features identify devices or contexts.
- Compare a fixed geometry use with the same Euclidean architecture and simple affective-history summaries on untouched people or an independent cohort.
- Obtain consented participant-linked longitudinal language before testing LLM conditioning; no such test is part of the current result.

## Repository organization

- `run_psycharchives_*.py` and their tests: active person-level experiments.
- `PSYCHARCHIVES_*.md`: protocols and aggregate interpretation, not raw data.
- `build_psycharchives_core_report.py` and `output/pdf/`: current report source and PDF.
- `results/`: team comparison submissions; do not place private data there.
- EmoBank, DAIR, Beck, and PersDyn scripts and notes: **historical exploratory work** retained for reproducibility. They are not the primary evidence for the current manifold-personality conclusion. The older EmoBank-focused [reproducibility guide](REPRODUCIBILITY.md) applies only to that legacy pipeline.
