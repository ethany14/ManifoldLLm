# Beck person–situation benchmark: next-prompt studying

## What this experiment answers

This is a first behavioral benchmark for the digital-personality-manifold project. It asks whether a representation of an individual's past states helps predict whether they **self-report studying at the next recorded prompt**. It does **not** yet test a manifold, establish stable personality, or validate an LLM persona. The objective is to lock a stronger behavioral baseline before comparing nonlinear geometry with an information-matched Euclidean model.

Data: the [openESM harmonization](https://openesmdata.org/datasets/0060_beck/) of [Beck and Jackson (2022)](https://doi.org/10.1177/09567976221093307), also archived at [Zenodo](https://doi.org/10.5281/zenodo.17361673). The harmonized table contains 8,672 prompts from 199 IDs. This analysis deliberately defines a **new cohort**; it does not reproduce the paper's 104-person, 5,971-assessment analysis. The harmonized source is licensed CC BY-NC 4.0; keep the downloaded data out of Git.

## Fixed protocol

- Keep participants with at least 15 recorded prompts. Sort by recorded timestamp, reject duplicate `(id, timestamp)` keys, and use the first 10 prompts as online personal-history calibration. Prediction pairs begin with prompt 10 predicting prompt 11; the individual can have more previous responses by then.
- Predict only when both current and next `studying` responses exist and the next recorded prompt is within 8 hours. This excludes overnight and long gaps; it does not imply a fixed prediction horizon.
- Split by participant, seed 42, into 105 train, 35 development, and 36 held-out test people. No participant crosses splits. The script saves exact IDs in `artifacts/beck_behavior_pilot/metrics.json`.
- Primary metric: Brier score (lower is better). Also report log loss, average precision, AUROC, sample count, and target prevalence. The test set is inspected once; do not retune on these test results.
- All input columns come from the current or earlier prompt. `studying` at the next prompt is the only target. `history_rate` uses outcomes through the current prompt, never the next one. Imputation and scaling are fit on training participants only.
- Comparators: train-set prevalence; current-response persistence (0.01/0.99 probability); online per-person history; fixed-parameter logistic regression with history and time; and the same regression adding current situation, affect, and selected activity reports. Personality items are not added to this first comparison because, by design, they are sparsely observed (about 20% nonmissing per item); they require a separately defined missingness/aggregation strategy.

## Results

After the cohort, calibration, target, and gap rules, there are 4,355 pairs from 176 people. The median retained gap is 3.98 hours. The train/development/test pair counts are 2,680/871/804; studying prevalence is 30.3%/26.2%/25.1% respectively.

| Model | Development Brier | Test Brier | Test average precision | Test AUROC |
|---|---:|---:|---:|---:|
| Train prevalence | 0.1949 | 0.1908 | 0.2512 | 0.5000 |
| Current-response persistence | 0.3095 | 0.2890 | 0.3096 | 0.5959 |
| Online person history | 0.1855 | 0.1872 | 0.3470 | 0.6146 |
| Logistic history + time | **0.1845** | **0.1804** | **0.3899** | 0.6335 |
| Logistic + situation/affect/activity | 0.1880 | 0.1843 | 0.3768 | **0.6418** |

The added context has a slightly higher test AUROC but a worse primary Brier score than history + time. These differences are exploratory, with no confidence intervals yet. A strong manifold claim is therefore premature. Context may contain useful ranking information but needs better calibration, dimensionality control, or temporal modeling; its current inclusion does not improve probability accuracy. The same test split was subsequently used for an exploratory matched graph-geodesic comparison; see [its result](BECK_GRAPH_GEOMETRY_RESULT.md). It is no longer an untouched confirmatory holdout.

## Next comparison for the research conclusion

1. Freeze this cohort and participant split. On the **development** people only, compare an information-matched Euclidean slow–fast encoder with a nonlinear manifold variant, using the same input windows, dimension, optimizer budget, and output head. Add an ablation with the geometry term disabled.
2. Use a fixed selection rule based on development Brier. Evaluate the chosen variants on the held-out test people without changing the protocol. Report participant-bootstrap confidence intervals for paired Brier differences and calibration plots.
3. Repeat for `procrastinating` and, if the conclusion aims to generalize beyond students, on a second population. For a genuine LLM digital personality, separately collect consented longitudinal language and behavior data; this table has no dialogue text or independent behavioral verification.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe Ethan\run_beck_behavior_pilot.py
```

Machine-readable metrics and exact split IDs: `Ethan/artifacts/beck_behavior_pilot/metrics.json`. Source files are expected in `Ethan/data/person_situation/source/` and are gitignored.
