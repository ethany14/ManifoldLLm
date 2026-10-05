# PsychArchives longitudinal manifold implementation

## What the supplied files contribute

The three supplied files join by `user_id`. `es_sensing_variables.csv` has
9,790 questionnaire records from 455 people; `person_variables.csv` has one
row per person and five trait parameters; `Codebook.csv` defines their meaning.
All 455 sensing IDs join to the person table. Only 380 people have all five
trait parameters. `NPAR` is *emotional stability* in this codebook, not an
unreversed neuroticism score. `valence` is a 1–6 current-feeling response.
`es_questionnaire_id` orders observations within a person, but this release
does not supply an actual per-prompt timestamp in the selected data.

Source: [Schoedel et al., PsychArchives dataset](https://psycharchives.org/en/item/975a7624-cd16-46bb-9d80-d988bed5e780).
The two data CSVs are Scientific Use License v1; the codebook is CC-BY 4.0.
Keep the licensed data in `Downloads`, do not commit it, and confirm any
further distribution or derivative-sharing permissions against your license.
The repository ignores `Ethan/data/psycharchives/` and
`Ethan/artifacts/psycharchives/`.

## Concrete training design

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe Ethan\run_psycharchives_manifold.py
```

`--source-dir` and `--output-dir` can override the default `Downloads` and
ignored local artifact directory. The script verifies the codebook and joins,
checks duplicate person/questionnaire keys, reads only 21 sensing/time cues
plus valence, and logs count/duration cues with `log1p`. It excludes keyboard
features and raw participant text in this first pass. It orders each person's
questionnaires by their running ID and constructs a causal, per-cue mean of
observations through questionnaire *t*. The target is valence at the **next
questionnaire**, not valence at a known number of hours later.

A stratified, seeded split is made by **person**: 318 training, 68 development,
69 untouched test people. Missing-value medians, cue standardization, and
trait standardization are fitted on training people only. The model takes
two inputs at each observed questionnaire:

- Slow input: the expanding prefix mean, representing evidence about the person.
- Fast input: the current cue vector minus that prefix mean, representing the
  present deviation/situation.

Two nonlinear encoders estimate 4-D slow `phi` and 4-D fast `z` Gaussian
posteriors. A nonlinear decoder `g(phi,z)` reconstructs the 22 selected
sensing/valence variables; another head estimates next-questionnaire valence.
A five-output trait head on `phi` supplies an auxiliary anchor only where the
person has all five measured traits. Adjacent-prefix stability and a small KL
term regularize the slow state. **Traits are targets, not input features.**

The person manifold is operationally defined by the decoder-induced metric
`G(phi) = J_phi g(phi, 0)^T J_phi g(phi, 0) + 1e-4 I`.
The metric is computed from the trained decoder, not from Isomap coordinates.
`phi` can be re-estimated when another questionnaire arrives, so this is a
variable-history implementation of a digital-personality *state estimate*.
It is not yet a calibrated continuous-time filter or an LLM agent.
The `infer_history(checkpoint, chronological_cues)` function in the training
module is the in-memory update interface. Given a DataFrame of one person's
chronologically ordered cue records, it returns slow and fast coordinates,
estimated trait parameters, next-questionnaire valence, and local metric
eigenvalues. Appending a new observation and calling it again updates the
person estimate; it does not store the observations or person ID.

## First development run

Three seeds completed. Mean development next-valence RMSE was **1.015** on
the 1–6 response scale (individual seeds: 1.058, 0.921, 1.067).
For descriptive context, a training-mean predictor scored 1.189 and simply
carrying forward current valence scored 1.068 on the same development rows.
The metric eigenvalues at 10 sampled development coordinates were positive
for every seed; this checks numerical definiteness only. The large seed
variation means these are exploratory engineering outputs, not a robust
performance comparison, evidence of intrinsic curvature, or evidence that
the inferred axes are psychologically identifiable. The test participants
were not used for preprocessing, selection, or evaluation.

Aggregate results and checkpoints are in ignored local
`Ethan/artifacts/psycharchives/`. No raw record or participant ID is written
there by the script. Its JSON records source hashes, counts, feature names,
and aggregate development results so the run can be checked against the
locally licensed files.

## Ordered-history and state-transition upgrade

The follow-on script leaves the prefix-mean model unchanged:

```powershell
.\.venv\Scripts\python.exe Ethan\run_psycharchives_sequence_manifold.py
.\.venv\Scripts\python.exe Ethan\test_psycharchives_sequence_manifold.py -v
```

It uses the same person split, 22 input variables, next-questionnaire target,
and training-only feature scaling. An ordered, one-directional GRU reads
standardized cues **with a separate observed/missing indicator for each
cue**. The slow 4-D estimate at prompt `t` depends only on prompts through
`t`. The fast 4-D estimate reads that prompt's cues and slow estimate. A
transition network maps `(phi_t, z_t)` to a predicted `z_(t+1)`; the next-
valence head reads `(phi_t, predicted_z_(t+1))`. This is an explicit
one-questionnaire-step transition, not a clock-time dynamics model. The
decoder again reconstructs observed cues and defines a Jacobian metric on
slow coordinates. Training also uses an auxiliary trait head, adjacent slow-
state stability, KL regularization, and consistency between predicted and
next observed fast states. Reconstruction loss excludes missing cues.

`infer_history(checkpoint, chronological_cues)` in the new module is the
in-memory update API. Supplying one additional observation re-runs the
ordered history and returns updated slow/fast coordinates, trait estimates,
next-questionnaire valence, and local metric eigenvalues. A synthetic check
confirmed that changing a future record does not change earlier outputs,
while changing record order does. The local checkpoints and aggregate JSON
are in ignored `Ethan/artifacts/psycharchives/sequence/`.

In the first development run, three seeds had next-valence RMSE values near
**0.87** (mean **0.871**) on the response's 1-to-6 scale. The earlier
prefix-mean script averaged **1.015** on the same split. This is a useful
implementation check, but it changes the encoder, transition, optimization,
and objective together, so the gap does **not** isolate manifold geometry or
establish superiority. The trait head's development RMSE is about
**1.07-1.12 in training-standardized units**, versus **1.14** for predicting
the training trait mean. That modest difference is not enough to call the
slow coordinates validated personality traits. Sampled local metrics are
positive-definite and vary by location, but those numerical checks do not
establish psychological meaning or intrinsic curvature. Test participants
remain reserved.
Across seeds, median adjacent coordinate changes were approximately
0.26-0.32 for slow `phi` and 1.70-1.96 for fast `z`. This is consistent with
the intended update behavior, but coordinate scales are learned and the
slow-stability penalty explicitly favors that pattern; it is not an
independent validation of trait stability.

## Implementation conclusion and next step

This dataset enables the concrete method we needed: estimate a slowly updated
person coordinate from longitudinal cues, a fast state from current deviation,
and obtain a nonlinear Riemannian metric from how a decoder turns person-space
motion into observable multi-cue changes. It joins person-level traits to
repeated affect and sensing from the **same people**, unlike isolated emotion
sentence datasets. The ordered, missingness-aware history encoder and
event-step transition are now implemented. Before interpreting `phi` as
personality, we still need stronger trait and temporal-stability validation,
uncertainty calibration, and a prespecified evaluation on untouched people.
These data do **not** provide participant-linked
language, so connecting this manifold to an LLM will require a separate,
consented text-linked dataset or a carefully specified interface experiment.

The development-only trait, independent-window, and component-ablation
checks are documented in
[PSYCHARCHIVES_EVALUATION_12_RESULTS.md](PSYCHARCHIVES_EVALUATION_12_RESULTS.md).
The subsequent comparison that places Euclidean or decoder-path distances
inside a matched contrastive training loss is in
[PSYCHARCHIVES_GEOMETRY_TRAINING_RESULTS.md](PSYCHARCHIVES_GEOMETRY_TRAINING_RESULTS.md).
The latest [cue-group sensitivity study](PSYCHARCHIVES_FEATURE_GROUP_RESULTS.md)
finds that mobile-usage cues improve person matching but not measured-trait
prediction; a simple valence mean-plus-variability baseline matches or
exceeds the current slow coordinate on trait prediction in development.
