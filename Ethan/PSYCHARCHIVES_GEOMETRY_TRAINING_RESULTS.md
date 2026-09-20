# Geometry-in-training development results

The corrected comparison follows
[protocol v2](PSYCHARCHIVES_GEOMETRY_TRAINING_PROTOCOL_V2.md).
Run from the repository root:

```powershell
.\.venv\Scripts\python.exe Ethan\run_psycharchives_geometry_training.py
.\.venv\Scripts\python.exe Ethan\test_psycharchives_geometry_training.py -v
```

It trains the same slow/fast sequence model on 318 training people under
three fixed conditions and evaluates 68 development people. There are
6,534 training and 1,377 development next-questionnaire pairs. The 69
reserved test people were not fitted or scored. Results below are means
across seeds 1, 2, and 3. The 57 development people with complete trait
parameters enter trait-probe scoring.

## Matched condition results

| Development readout | Base | Euclidean identity loss | Decoder-path identity loss |
| --- | ---: | ---: | ---: |
| Next-valence RMSE, original 1–6 scale | 0.873 | 0.890 | 0.879 |
| Five-trait linear-probe macro R² | 0.071 | 0.135 | 0.138 |
| Five-trait pooled RMSE, train-standardized units | 1.093 | 1.055 | 1.053 |
| Early/late same-person rank-1, Euclidean readout | 11.3% | 16.7% | 15.2% |
| Early/late same-person rank-1, decoder-path readout | 9.8% | 15.7% | 16.7% |
| Euclidean within/between median-distance ratio | 0.608 | 0.534 | 0.523 |
| Decoder-path within/between median-distance ratio | 0.521 | 0.529 | 0.575 |

Chance rank-1 matching is 1/68 = **1.47%**. Lower RMSE and distance ratio,
and higher R² and rank-1 matching, are preferable. The training-mean trait
baseline has pooled development RMSE **1.140** in the same standardized
units. The identity loss uses nonoverlapping first-five and last-five
windows from training people, with no personality trait pair labels.

| Mean trait R² | Base | Euclidean identity | Decoder-path identity |
| --- | ---: | ---: | ---: |
| NPAR, emotional stability | 0.196 | 0.234 | 0.238 |
| EPAR, extraversion | 0.127 | 0.199 | 0.200 |
| OPAR, openness | 0.037 | 0.084 | 0.085 |
| CPAR, conscientiousness | -0.093 | 0.067 | 0.065 |
| APAR, agreeableness | 0.088 | 0.092 | 0.101 |

The geometry-trained condition had a sampled decoder-metric eigenvalue
range of approximately **0.002–0.228** across seeds. It varied across
sampled locations, with no repeat of the scale inflation seen in the first
pilot. Positivity is partly guaranteed by the `1e-4 I` ridge term and is
only a numerical check, not a claim of psychologically meaningful
curvature.

## Interpretation and method decision

The added **same-person contrastive objective** improves independent-window
matching and modestly improves trait decodability, at the expense of a
small next-valence RMSE increase. The effect is present with either a
plain Euclidean coordinate distance or decoder-path length. Decoder-path
training is now a real part of optimization: gradients pass through the
nonlinear decoder and slow encoder. Its own path-distance matching is
16.7% versus 15.7% for the Euclidean-trained control, and trait macro R²
is 0.138 versus 0.135. Those small differences do **not** establish a
distinct benefit from manifold geometry: Euclidean readout matching favors
the Euclidean-trained model, and the decoder-path within/between ratio is
worse for geometry training. Per-seed differences also vary in direction.

The first numerical pilot, documented by
[protocol v1](PSYCHARCHIVES_GEOMETRY_TRAINING_PROTOCOL.md), detached the
distance normalizer's gradient. Its decoder-path objective inflated local
metric scale (sampled maximum eigenvalue about 8.3, compared with about
0.24 in the base model). We kept that pilot's ignored local artifacts for
diagnosis but do not use its comparisons as evidence. Version 2 allows
gradients through the normalizer; a synthetic scaling check confirms that
rescaling the decoder leaves the normalized contrastive loss unchanged.

**Current implementation answer:** use an ordered, missingness-aware
slow/fast encoder with a nonlinear decoder. The decoder induces a local
manifold metric, and an optional decoder-path contrastive objective can
train person-space distances. On these data, the robustly observed gain is
from *same-person contrastive learning*, not yet from using the nonlinear
metric instead of Euclidean distance. Trait R² remains low, and device or
context patterns may explain part of the person signal. Therefore `phi`
is still a candidate digital-personality coordinate, not a validated
personality construct. The reserved test cohort should remain sealed until
the representation and its interpretation are fixed.

Aggregate per-seed metrics and checkpoints are local and Git-ignored under
`Ethan/artifacts/psycharchives/geometry_training_v2/`. The licensed CSVs
remain in Downloads; no individual row, participant ID, or coordinate is
exported in the aggregate report.
