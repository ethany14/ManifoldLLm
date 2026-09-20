# PersDyn public longitudinal pilot

This is the first no-new-data experiment for the digital-personality project. It tests whether a model can forecast repeated **self-rated personality states** for participants not used to fit its global parameters. It does not test language generation, independent behavior, or a learned manifold.

## Source and preparation

Source: Sosnowska, Kuppens, De Fruyt, and Hofmans (2020), *New Directions in the Conceptualization and Assessment of Personality - A Dynamic Systems Approach*, [paper](https://doi.org/10.1002/per.2233), [open materials](https://osf.io/t5v34/). The pilot reads the original `dataES copy.xlsx`, sheet `only TIPI`, from the public OSF download. The script verifies SHA-256 `3b6c8ba9d02a57ac23e590e5db9b5a20c97c78531ce68e130a203315ac8cdbd1`. The raw workbook is kept under `data/persdyn/source/`, which is gitignored; do not redistribute it without checking the source's terms.

The sheet has 2,100 planned rows for 60 participants. We keep 1,844 rows with participant ID, timestamp, and all five O/C/E/A/N state scores; we retain the source's 0-100 scale and chronological order. For each participant, the first 10 complete observations establish a history. Every later observation is a next-state target. Inputs are only the previous state, the running mean of earlier states, elapsed time, and prediction-time hour. The current target rating is never an input. The hour is valid only for prediction at the moment of the scheduled survey, not forecasting an unknown future observation time.

We split **participants**, not rows: 36 train, 12 development, and 12 test participants at seed 42. The resulting forecast examples are 742/255/247. For a new test participant, the model may use that participant's first 10 observed ratings as calibration; the global ridge/MLP parameters are learned on train participants only. This is **few-shot new-person forecasting**, not zero-shot prediction. All reported results below are exploratory from one split and one model seed.

## Results on 12 held-out participants

| Predictor | Mean MAE, 0-100 scale | Mean RMSE | Mean R2 |
|---|---:|---:|---:|
| Previous rating (persistence) | 14.723 | 20.245 | -0.257 |
| Participant's running history mean | 12.549 | 16.407 | 0.174 |
| Ridge regression on history features | **12.243** | **16.108** | **0.197** |
| Small nonlinear MLP on same features | 12.489 | 16.412 | 0.162 |

In this first pilot, the nonlinear MLP did **not** beat the linear ridge baseline. Stable individual history was more useful than naive last-observation persistence. The data do not yet support an advantage for nonlinear models, autoencoders, or manifold geometry. They only establish a working person-held-out longitudinal benchmark. Do not compare these numbers directly with the earlier EmoBank/dair-ai emotion scores: tasks, targets, and splits differ.

## Reproduce

From `Ethan/`, install `requirements.txt` in an environment where you have write access, then run:

```powershell
python .\run_persdyn_public_pilot.py
```

The script downloads the unchanged source workbook if absent, validates its checksum, and writes the full split IDs and metrics to `artifacts/persdyn_public_pilot/metrics.json`. The raw workbook is not committed. The current run used an isolated `openpyxl` installation because the sandbox could not modify the existing `.venv`; a normal team installation from the updated requirements file should supply it.

## Next scientifically meaningful comparison

Keep this split and primary target fixed. First repeat the baseline across several prespecified person splits and report participant-level uncertainty. Then add a matched-capacity hierarchical dynamic model with a slow person variable and fast state, followed by the *same* model plus a geometric term. Select architecture and dimension on development participants only. Geometry matters only if it improves held-out future-state prediction beyond the no-geometry hierarchy. Because this source has no independent behavioral outcome or dialogue text, even a positive result would support personality-state modeling, **not** a validated digital personality or LLM persona.
