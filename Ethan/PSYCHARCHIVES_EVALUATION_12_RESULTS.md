# PsychArchives personality readouts and matched ablations

This development-only analysis follows
[PSYCHARCHIVES_EVALUATION_PROTOCOL.md](PSYCHARCHIVES_EVALUATION_PROTOCOL.md).
The script recorded that protocol's SHA-256 hash before evaluation. The
licensed source files were read in place. No individual record, coordinate,
trait value, or participant ID was exported. The 69-person test set was not
scored. Run from the repository root:

```powershell
.\.venv\Scripts\python.exe Ethan\run_psycharchives_evaluation_12.py
```

The run uses 318 training and 68 development people, with 6,534/1,377
next-questionnaire pairs respectively. All conditions use the same 22 cues,
person split, preprocessing, three random seeds, and fixed 30-epoch budget.
Trait probes use 57 development people with complete five-trait data.

## Results

Values below are means across seeds 1, 2, and 3. Trait RMSE is in units
standardized using training people's trait distributions; lower is better.
The five-trait training-mean baseline has pooled development RMSE **1.140**.
For early-to-late matching, chance rank-1 accuracy is **1/68 = 1.47%**.

| Readout | Full | No trait loss | No transition |
| --- | ---: | ---: | ---: |
| Next-valence RMSE, 1–6 scale | 0.873 | 0.872 | 0.880 |
| Five-trait linear-probe macro R² | 0.071 | 0.115 | 0.058 |
| Five-trait pooled RMSE | 1.093 | 1.067 | 1.100 |
| Early/late rank-1 same-person match, Euclidean | 11.3% | 9.3% | 8.8% |
| Median within-person / between-person Euclidean distance | 0.608 | 0.595 | 0.623 |

The trait probe fits a separate fixed `Ridge(alpha=1)` readout to the
training people's final `phi` and measures development people. It is the
same probe procedure for all conditions, including the no-trait-loss model;
the no-trait model's original auxiliary head is not used as an assessment.

| Trait R² (mean across seeds) | Full | No trait loss | No transition |
| --- | ---: | ---: | ---: |
| NPAR, emotional stability | 0.196 | 0.203 | 0.210 |
| EPAR, extraversion | 0.127 | 0.191 | 0.103 |
| OPAR, openness | 0.037 | 0.053 | 0.015 |
| CPAR, conscientiousness | -0.093 | 0.046 | -0.112 |
| APAR, agreeableness | 0.088 | 0.081 | 0.073 |

The independent-window task encodes a person's **first five and last five
questionnaires separately** with the recurrent state reset, so the two
windows share no records. In the full model, the Euclidean early-to-late
rank-1 rate is 11.3%, versus 9.8% when distances are measured by the
eight-segment length of a straight path through the nonlinear decoder
(`z=0`). The decoder-path distance gives a lower median within/between
ratio (0.521 vs 0.608), however. These two readouts therefore give mixed
evidence. The decoder path is not a geodesic optimum; it is a fixed-path
approximation to the decoder-induced geometry.

## Interpretation

1. There is some repeatable individual signal in `phi`: disjoint early/late
   windows match the same person more often than the 1.47% chance rate.
   That does not establish a psychologically valid personality manifold;
   sensing patterns, device use, and context may also identify people.
2. Trait association is weak and uneven. Macro R² is only 0.071 in the full
   condition, and conscientiousness is negative. Removing the auxiliary
   trait loss did **not** reduce trait decodability in this run. We should
   not assert that the current trait loss is an effective personality anchor.
3. Removing the explicit fast transition modestly worsened next-valence
   RMSE (0.880 vs 0.873), but this is a small development-set difference,
   not a proof of improved dynamics.
4. The geometric readout did not improve rank-1 person matching. It has a
   favorable median distance ratio but worse rank-1 accuracy. Crucially,
   the pullback metric is calculated **after training**, so these runs do
   not test whether training *with* Riemannian geometry improves the model.

The 30-epoch budget was chosen after earlier development inspection, and
these 68 development people have been used repeatedly. The analyses are
method diagnostics, not final generalization estimates. The implementation
answer remains: infer a slowly updated person coordinate and fast state
from ordered sensing/affect history, then define local geometry through the
nonlinear decoder. To call the result a *digital personality* empirically,
we still need stronger, prespecified evidence that `phi` is trait-relevant,
stable across contexts, and not merely a device/situation fingerprint.
Keep the reserved test people unopened until the model and interpretation
criteria are finalized.

Aggregate per-seed results and local checkpoints are in ignored
`Ethan/artifacts/psycharchives/evaluation_12/`.
