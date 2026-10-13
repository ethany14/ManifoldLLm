# PsychArchives high-dimensional input audit and M1 pilot

## Scope

Audit the already licensed smartphone panel before treating its raw column count as rich individual personality evidence. Then compare the earlier 22 selected inputs with a broader, training-only screened set using the same **16D slow person / 4D fast state** nonlinear sequence encoder/decoder. This is a separate longitudinal implementation pilot, not an extension of the IPIP questionnaire scores or a trained language persona.

The original IPIP models, their reports and all reserved test cases remain unchanged. No licensed records, individual IDs, or participant coordinates were exported. The local source CSVs were read in place.

## Feature audit

The sensing/experience-sampling CSV has **1,367 columns**: two identifiers, eight DIAMONDS self-report outcomes, current valence, and **1,356 sensing candidates**. This schema count distinguishes sensing from all non-ID columns; the dataset's broader published feature description should not be interpreted as 1,365 independent personality traits.

The dictionary includes **963 KB keyboard columns**. Their median training snapshot coverage is approximately **7.36%**, so most do not give densely observed language evidence per person. Many are LIWC category aggregates, not raw chat messages.

Rules fixed before the primary comparison:

- Exclude IDs, DIAMONDS outcomes and person-level trait labels from model inputs. Current valence remains a common input in both conditions.
- Include sensing from all dictionary cue groups (phone, situation and environment), rather than screening only `Smartphone_Cue`.
- Apply fixed dictionary range checks where specified; nonnumeric and infinite entries become missing. Nonnegative count/duration variables use log1p.
- Require observations in at least 70% of training snapshots and at least 70% of training people.
- Remove near constants and columns where one value exceeds 98% of observed training values.
- Greedily remove absolute correlations >=0.98, computed after training-median filling, in fixed source-column order.
- Fit medians and scaling on training people only; carry missingness masks into the model.

Of 1,356 candidates, **258** pass support/variation rules; redundancy filtering removes **31**, leaving **227 sensing inputs**. Add common current valence to obtain **228 inputs**.

| Retained feature family | Columns |
|---|---:|
| Activity | 20 |
| App | 105 |
| GPS | 39 |
| Headphone | 2 |
| KB | 3 |
| Music | 1 |
| Notification | 5 |
| Phone | 23 |
| Power | 3 |
| Screen | 14 |
| Timestamp | 5 |
| Wifi | 7 |

The three retained KB variables are usage counts in communication, social-media and internet apps. **No LIWC lexical-category proportions survive this screen.** This run therefore does not learn a language-content personality manifold.

The conservative coverage thresholds prioritize usable individual observations. They are not an exhaustive missing-data strategy or evidence that sparse columns are psychologically irrelevant. A future masked model with deliberate modality handling would require a separately specified protocol, not relaxing thresholds after seeing development scores.

## Matched pilot setup

Person split: **318 train / 68 development / 69 reserved test**, identical to the prior PsychArchives protocol. Records are ordered by questionnaire ID, which does not establish fixed time intervals. Prediction targets valence at the next questionnaire, not the next hour/day. Codebook-fixed range/log conversion is applied in memory; no transformation parameters or feature-selection decisions are estimated from test people, and no model inference/scoring uses them.

Both conditions use seed42, 30 fixed epochs, a masked 64-unit GRU history encoder, 16D Gaussian slow state, 4D Gaussian fast state, a nonlinear 64-unit decoder, fast-state transition and next-valence head. There is **no auxiliary trait training loss and no contrastive training loss**. Reconstruction, KL, transition and slow-stability terms match between conditions. The trait readout is a post-hoc train-person-fitted ridge probe, not an untrained neural trait head. The checkpoint is after epoch30, not selected by development performance.

This is not an exactly parameter-matched comparison: 22-input model **33,496 parameters**, 228-input model **152,358 parameters**. The architecture's hidden widths and latent dimensions are matched, but larger inputs expand connected layers. Preprocessing policy is shared between these two freshly trained models; numbers are not directly interchangeable with older 4D/32-hidden-unit runs.

## Completed development results

| Measure | 22-input control | 228-input screened model |
|---|---:|---:|
| Next-questionnaire valence RMSE (1–6), lower better | **0.8630** | 1.0736 |
| Next-questionnaire valence MAE (1–6), lower better | **0.6505** | 0.8137 |
| Trait ridge-probe macro R², higher better | **0.1289** | 0.0201 |
| Disjoint early/late 5-record windows: coordinate top-1 | 8.82% | 10.29% |
| Same windows: decoder-path top-1 | 4.41% | 8.82% |
| Sampled median metric condition number | 1401.20 | 48.10 |

Matching uses the same 68 development people (chance approximately 1.47%), with independently encoded first/last windows. A modest matching gain does not imply a more psychologically valid personality state: device, app habits, missingness and context can identify people. Next-valence prediction and independent trait association deteriorate. The metric is calculated on ten early-window states under the fixed `z=0` slice, using `J_phi g^T J_phi g+1e-4 I`; eight-segment path lengths include the same ridge contribution. They are straight-coordinate paths, not optimized geodesics.

The 228-input metric is less poorly conditioned at sampled points, but this is a numerical property in different ambient feature spaces, not proof of better geometry. Absolute metric lengths are not comparable across different decoded feature schemas. A 16D choice made on IPIP also does not establish that 16D is suitable for this 455-person sensing panel.

## Conclusion and next decision

**Do not replace the current IPIP personality-questionnaire core with this high-dimensional sensing pilot on the basis of column count.** The expanded sensing model can construct an operational nonlinear manifold, but this configuration performs worse on trait association and next-valence prediction. More features did not automatically supply better personality evidence.

The practical research distinction is:

- IPIP: direct, relatively dense multidimensional personality measurements; use as the current personality-manifold foundation.
- PsychArchives: repeated behavioral/context evidence; retain as a later updating/behavioral extension, not a validated substitute for dense trait evidence.

If revisiting the sensing extension, first separate behavioral, context, device and linguistic modalities; audit missingness/device confounds; avoid letting a large app-feature group dominate reconstruction; and test the model across additional seeds. The current run has one seed and should not be treated as proof that high-dimensional sensing is intrinsically unsuitable. New questionnaire candidates such as SAPA require their own sparse-observation audit before claiming richer per-person evidence.

## Reproduction and provenance

```powershell
.\.venv\Scripts\python.exe Ethan\run_psycharchives_highdim.py
```

Primary completed output: ignored `Ethan/artifacts/psycharchives/highdim_m1_v2/`, with `feature_audit.json`, `protocol.json`, exact training-source snapshot, aggregate `development_results.json` and both model checkpoints. Running into a populated output directory is refused. Source file and script SHA-256 hashes are recorded, and split IDs are represented only by aggregate set hashes, not exported participant lists.

Version1 used the too-restrictive `Smartphone_Cue` dictionary category and produced 118 inputs; it is preserved locally as a restricted-scope exploratory run. Version2 corrected that schema restriction to include legitimate sensing in all groups. The coverage, variation and redundancy thresholds were not tuned to development outcomes; the v2 correction follows dictionary semantics. Nevertheless, both runs examined the same development cohort, so v2 results remain exploratory.

Method lineage: Kingma & Welling's variational encoder/decoder and Arvanitidis et al.'s decoder-induced geometry are adapted to a causal slow/fast sequence model; the architecture and mixed training objective are project-specific, not a reproduction of a validated personality model.
