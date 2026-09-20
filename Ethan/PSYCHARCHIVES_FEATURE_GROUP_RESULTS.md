# Feature-group sensitivity: identity signal versus trait information

This development-only analysis follows
[the fixed feature-group protocol](PSYCHARCHIVES_FEATURE_GROUP_PROTOCOL.md).
It uses the same PsychArchives source files, 318 training people, 68
development people, three seeds, and 30 epochs per cue group. The auxiliary
trait-loss weight is **zero** for all models. Traits are used only for a
fixed `Ridge(alpha=1)` probe fitted on the 266 training people with complete
traits and evaluated on the 57 corresponding development people.
The 69-person test split remains unscored. Raw licensed data stay in
Downloads; aggregate outputs and checkpoints are Git-ignored.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe Ethan\run_psycharchives_feature_groups.py
.\.venv\Scripts\python.exe Ethan\run_psycharchives_valence_summary_baselines.py
```

## Development results

Model entries are means over seeds 1, 2, and 3. The five-trait training-
mean predictor has pooled development RMSE **1.140** in training-
standardized units. Rank-1 same-person matching chance is **1/68 = 1.47%**.

| Inputs | Next-valence RMSE (1–6) | Five-trait macro R² | Five-trait pooled RMSE | Disjoint-window same-person rank-1 |
| --- | ---: | ---: | ---: | ---: |
| All 22 cues | 0.892 | 0.126 | 1.060 | 16.7% |
| Activity + time + valence (9) | 0.890 | 0.147 | 1.048 | 6.9% |
| Time + valence (6) | 0.886 | 0.151 | 1.046 | 5.9% |
| Valence alone (1) | 0.885 | 0.145 | 1.049 | 6.4% |

The excluded cues in the 9-input group are App, Phone, Screen, GPS,
Notification, Power, and Wifi features and their missingness indicators.
All groups use the same GRU hidden width, 4-D slow/fast states, nonlinear
decoder, Euclidean same-person contrastive objective, split, and training
budget. Input/decoder parameter counts necessarily change with cue count.
Matching here uses coordinate-Euclidean distance. In the one-cue condition,
a 4-D decoder pullback metric has rank at most one before ridge
regularization; that condition is a feature-information control, not a
standalone claim of four-dimensional Riemannian geometry.

| Trait R² (three-seed mean) | All 22 | Activity/time/valence | Time/valence | Valence only |
| --- | ---: | ---: | ---: | ---: |
| NPAR, emotional stability | 0.216 | 0.242 | 0.227 | 0.218 |
| EPAR, extraversion | 0.186 | 0.219 | 0.242 | 0.217 |
| OPAR, openness | 0.072 | 0.085 | 0.101 | 0.107 |
| CPAR, conscientiousness | 0.066 | 0.087 | 0.072 | 0.065 |
| APAR, agreeableness | 0.089 | 0.101 | 0.112 | 0.120 |

## Post-hoc non-neural reference

After inspecting the feature-group results, we added a **post-hoc** simple
baseline. It is not part of the fixed protocol or model selection. A linear
probe using only each person's mean valence reaches macro trait R²
**0.147**. Using mean and standard deviation of valence reaches **0.159**
and pooled RMSE **1.040**, better than every model above on these 57
development people. In the independent first-five/last-five matching task,
the mean-plus-SD summary reaches **7.4%** rank-1, similar to the reduced-
cue models but below the full 22-cue model. The mean-only rank-1 is 1.5%;
ties in a five-response average make that one-dimensional matching readout
coarse. This baseline is a direct check against attributing ordinary
affective averages and variability to a learned manifold.

## Interpretation

Removing mobile-usage cues sharply lowers disjoint-window person matching
(16.7% to roughly 6–7%) while leaving measured-trait prediction similar
or slightly higher. This is **consistent with** those cues contributing
person-specific usage patterns, but the ablation cannot identify whether
they are device fingerprints, genuine habits, or contextual behavior.
Valence-only histories also match above chance, so not all individual
signal comes from phone usage.

More decisively, the simple mean-plus-SD valence baseline predicts the
five measured trait parameters at least as well as the learned slow
coordinate in this repeatedly inspected development cohort. Thus the
current PsychArchives experiments do **not** establish that a nonlinear
manifold adds personality information beyond basic affective summaries.
The engineering answer to *how to construct* a manifold state remains
valid—ordered history to slow `phi`, current deviation to fast `z`, and a
decoder-induced metric—but the empirical claim that `phi` is a superior
digital-personality representation is unsupported here. More model tuning
on the same development people would further weaken an eventual claim.
Before opening the reserved test cohort, a new measurement target or
independent longitudinal cohort should be identified, especially if the
intended output is an LLM persona rather than affect prediction.

Per-seed aggregate metrics are local under
`Ethan/artifacts/psycharchives/feature_groups/`.
