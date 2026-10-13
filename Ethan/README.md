# Digital Personality Manifold - Ethan

This folder contains the current person-level method and its exploratory evaluation. The research question is **how to use manifold geometry to form a revisable digital-personality state**, not simply whether a nonlinear embedding improves one score.

## Current answer and status

For each participant, a causal encoder reads ordered observations and estimates a slowly changing person posterior `q(phi | history)` and a fast state `q(z | current observation, phi)`. A nonlinear decoder `g(phi, z)` reconstructs observable cues. On a prespecified fast-state slice, the decoder Jacobian defines the local person-space metric

```text
G(phi) = J_phi g(phi, 0)^T J_phi g(phi, 0) + 1e-4 I
```

Geometry becomes operational when decoded path length is used for person similarity or inside the same-person contrastive training loss. The current implementation uses a sampled straight coordinate path, **not an optimized geodesic**. It does not use Isomap coordinates as targets. The model can re-estimate `phi` and `z` after a new observation.

The method is implemented, but the current development results do **not** establish that nonlinear geometry beats a matched Euclidean identity objective, or that `phi` carries personality information beyond simple valence mean and variability. No LLM persona is trained here. See the [current method report](output/pdf/PSYCHARCHIVES_CORE_METHOD_REPORT_EN.pdf) for the literature lineage, exact interpretation, and next-stage roadmap.

## Latest method report

Read [Learned Manifolds for Digital Personality: Method and Rationale (English PDF)](output/pdf/LEARNED_MANIFOLD_METHOD_AND_RATIONALE_EN.pdf) for the selected encoder-decoder method, decoder-induced Riemannian geometry, literature foundations, IPIP comparisons, limitations, and implementation roadmap. Core equations are typeset with LaTeX-style notation. The 16D setting is provisional; the report does not claim predictive dominance or a completed personality-conditioned language model.

## Current dataset and boundaries

The completed [high-dimensional sensing audit and pilot](PSYCHARCHIVES_HIGHDIM_RESULTS.md) examines the full local schema. Training-only screening keeps **227 sensing cues + current valence = 228 inputs**, compared with a newly trained 22-input control using the same 16D slow / 4D fast architecture. The expanded model has worse development next-valence RMSE (1.0736 vs 0.8630) and trait-probe macro R² (0.0201 vs 0.1289). This single-seed pilot does not justify replacing the current IPIP core; the sensing panel remains a possible behavioral/temporal extension. No raw text or LIWC lexical proportions survive this conservative screen. Local products are ignored under `artifacts/psycharchives/highdim_m1_v2/`; test people remain unscored.

```powershell
.\.venv\Scripts\python.exe Ethan\run_psycharchives_highdim.py
```

The licensed PsychArchives smartphone-sensing panel links 9,790 repeated questionnaire snapshots from 455 people to person-level variables. The scripts read `Codebook.csv`, `person_variables.csv`, and `es_sensing_variables.csv` from the user's `Downloads` folder by default; use `--source-dir` if your licensed files live elsewhere. Do **not** copy these files into Git or share participant-level rows or coordinates without checking the Scientific Use License.

The fixed split is by person: 318 train, 68 development, and 69 reserved test people. The current scripts use 21 sensing/time cues plus valence, ordered by questionnaire ID. The target is valence at the next questionnaire, **not** a known number of hours later. The reserved test people have not been scored in the reported PsychArchives comparisons. Preprocessing statistics are fitted on training people.

Source: [PsychArchives dataset record](https://psycharchives.org/en/item/975a7624-cd16-46bb-9d80-d988bed5e780).

## Setup

### New high-dimensional questionnaire pilot: IPIP-NEO-300

The new M1 experiment reads `data/Kajonius_Johnson_2019_SUPPL.zip` directly, including its nested Johnson (2005) 20,993-case data and official item key. It uses **300 personality questionnaire items**, not frozen-LLM embeddings or Isomap coordinates. A masked nonlinear variational encoder learns an 8-dimensional Gaussian personality coordinate; a nonlinear decoder supplies `G(phi)=J_g^T J_g+1e-4 I`. Split-form contrastive training uses sampled decoder-path lengths. This is a **cross-sectional questionnaire model**, not the earlier slow/fast sequence model or a trained language persona.

```powershell
.\.venv\Scripts\python.exe -m pip install xlrd
.\.venv\Scripts\python.exe Ethan\test_ipip_manifold.py -v
.\.venv\Scripts\python.exe Ethan\run_ipip_manifold.py
```

The first run saves the immutable split, preprocessing, checkpoint, epoch history and development metrics in ignored `Ethan/artifacts/ipip/m1_seed42/`. Re-running into an existing output directory is refused; choose `--output` for a separate experiment. Both raw ZIP and participant-level products are excluded from Git by `Ethan/.gitignore`. The source archive is not modified or copied into public results. Read [the IPIP protocol](IPIP_M1_PROTOCOL.md) for cleaning rules and [completed pilot results](IPIP_M1_RESULTS.md) for scores and limitations. Posterior variances are near the lower bound, so uncertainty is not calibrated. Reserved test cases remain unscored.

The completed [8/16/32D comparison](IPIP_DIMENSION_COMPARISON.md) uses three training seeds with identical data splits and item forms. Mean hidden-item RMSE is **0.9374 / 0.9103 / 0.8977**. The 32D model predicts best, but its metric condition number is much larger. **16D is the provisional primary geometry-focused implementation**, with 32D retained as a capacity comparison; this is an engineering tradeoff, not an intrinsic-dimension finding. The raw 150D ridge reference still predicts better (RMSE 0.8815).

```powershell
.\.venv\Scripts\python.exe Ethan\run_ipip_dimension_sweep.py
```

The sweep reuses completed runs and saves aggregate results to ignored `artifacts/ipip/dimension_sweep_v1/`. For individual new runs, keep `--data-seed 42` fixed and change `--seed` only for model-training randomness. The reusable primary checkpoint is `artifacts/ipip/dimension_sweep_v1/dim16_seed42/model.pt`.

The completed [16D geometry-objective comparison](IPIP_GEOMETRY_ABLATION.md) holds architecture and data fixed across decoder-path / Euclidean / no-contrastive training, each with three seeds. Mean hidden-item RMSE is **0.91035 / 0.91065 / 0.91181**. Mean path-readout matching is **94.14% / 94.14% / 94.40%**. These small differences do **not** establish decoder-path training superiority. The decoder defines the manifold metric in every condition; a distance-based contrastive training term is optional. Posterior uncertainty remains uncalibrated and test cases remain unscored.

```powershell
.\.venv\Scripts\python.exe Ethan\run_ipip_geometry_ablation.py
```

New comparison runs and aggregate results are local under ignored `artifacts/ipip/geometry_ablation_v1/`. The existing 16D path checkpoint remains a reference prototype, not a proven winner.

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
5. [Final report](output/pdf/PSYCHARCHIVES_CORE_METHOD_REPORT_EN.pdf)

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
