# Beck behavior pilot: matched graph-geometry comparison

## Question and design

Does an explicit nonlinear graph-distance constraint improve next-prompt self-reported `studying` prediction over the **same** slow–fast neural architecture without that constraint? This experiment uses the exact 105/35/36 train/development/test participant split and 4,355 eligible prediction pairs established in [the baseline protocol](BECK_BEHAVIOR_PILOT.md). The task is exploratory because the test split was already inspected in the baseline experiment.

Each model receives identical information: a person's mean situation, affect, and selected activity reports from the first 10 prompts; the current prompt's deviation from that anchor; current/past studying history; and current time variables. The slow encoder maps the person anchor to four dimensions, the fast encoder maps the deviation to four dimensions, and a shared head predicts the next-prompt probability. Person anchors use only the first 10 prompts. Features are imputed/scaled using training participants only. Both variants have the same parameter count, optimizer, learning rate, training ceiling, early-stopping rule, and three random seeds.

The geometric variant adds **unsupervised graph-geodesic stress**: an 8-nearest-neighbor graph is built from the first-10-prompt anchors of training people only, shortest-path distances on that graph are computed, and pairwise distances of their four-dimensional slow embeddings are encouraged to match normalized graph distances. The comparator sets that loss weight to zero. Geometry weight `0.01` and graph `k=8` were fixed for this pilot. This is a nonlinear graph regularizer, **not** a learned Riemannian metric, a direct observation of personality geometry, or an LLM training method.

## Results

Three-seed probability ensembles; Brier is the primary score (lower is better).

| Model | Development Brier | Test Brier | Test AUPRC | Test AUROC |
|---|---:|---:|---:|---:|
| Logistic history + time (earlier baseline) | **0.1845** | **0.1804** | **0.3899** | 0.6335 |
| Slow–fast, Euclidean/no geometry | 0.1847 | 0.1859 | 0.3798 | 0.6436 |
| Same slow–fast + graph-geodesic stress | 0.1841 | 0.1855 | 0.3790 | 0.6434 |

For the paired neural comparison, test Brier difference (graph minus no-geometry) is **−0.00041**. A 2,000-resample participant-cluster percentile interval is **[−0.00086, 0.00000]** (unrounded upper endpoint 0.0000031), so this pilot does not give robust evidence of improvement. Both neural variants remain worse than the simpler logistic history-and-time baseline on the main score. The graph variant also does not improve AUPRC or AUROC over its matched neural comparator.

## Current conclusion

The data support using longitudinal personal history and time to predict a narrowly defined self-reported behavior. They **do not yet justify a claim that nonlinear manifold geometry improves behavioral prediction or forms a digital personality**. A small benefit within one neural architecture is insufficient when the confidence interval touches zero and a simpler comparator performs better.

The test set has now been viewed in multiple model-development steps. All subsequent tuning on this split must be labeled exploratory. For a confirmatory claim, pre-register the final pipeline and use a genuinely untouched participant sample or independent dataset. Also test more than one behavioral outcome and later add consented longitudinal language data before making any LLM-personality claim.

## Reproduce and inspect

From repository root:

```powershell
.\.venv\Scripts\python.exe Ethan\run_beck_behavior_pilot.py
.\.venv\Scripts\python.exe Ethan\run_beck_graph_geometry.py
```

Detailed seed results, exact participant IDs, feature list, and bootstrap interval are in `Ethan/artifacts/beck_behavior_pilot/graph_geometry_results.json`. The downloaded harmonized data are kept out of Git. Data source: [Beck & Jackson (2022)](https://doi.org/10.1177/09567976221093307); [openESM dataset](https://openesmdata.org/datasets/0060_beck/); [Zenodo harmonized record](https://doi.org/10.5281/zenodo.17361673).
