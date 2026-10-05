# How the current prototype uses a manifold to form a digital-personality state

This is an **implementation demonstration**, not a claim of psychological
validity or geometry superiority. It runs on the existing Beck longitudinal
self-report data. There is no Isomap-coordinate target and no LLM text
generation in this demonstration. The current version trains the slow
posterior on **expanding observed-history prefixes**; the older
`run_beck_riemannian_vae.py` checkpoint used a fixed first-ten-prompt anchor.

## Concrete computational object

For a person at an observation time, the runtime returns:

1. A four-dimensional *slow person posterior* `q(phi | history_<=t)` (mean
   and standard deviation), inferred from per-feature means of every
   observation available through the current prompt.
2. A four-dimensional *fast state posterior* `q(z)` from the current
   standardized deviation from the current history prefix.
3. A local metric `G(phi) = J_phi g(phi, z=0, time=0)^T
   J_phi g(phi, z=0, time=0) + 1e-4 I`, where the nonlinear decoder `g`
   reconstructs multidimensional standardized situation/affect/activity
   features. The runtime reports its eigenvalues as a basic numerical check.
4. A next-prompt self-reported studying probability conditioned on mean
   `phi`, mean `z`, and current standardized time features.

The manifold is the person-coordinate space **equipped with this
decoder-induced metric**, not a plotted two-dimensional Isomap image. The
runtime approximates a person-to-person geodesic by shortest path over the
105 training-person coordinates plus two query people; edges are weighted by
decoded straight-path lengths. It reports the reference-graph size and also
reports coordinate-Euclidean and straight-decoded-path lengths so the
approximation is inspectable.

## Working interface

`personality_manifold_runtime.py` implements:

- `posterior(anchor)` and `state_posterior(deviation)`
- `local_metric(phi)`
- `infer(anchor, deviation, time)`
- `update_fast_state(previous_snapshot, new_deviation, new_time)`
- `infer_from_raw_history(history, timestamp)` and
  `update_history(snapshot, new_raw_observation, timestamp)` for checkpoints
  trained with variable-length prefixes. The latter re-estimates **both**
  `q(phi)` and `q(z)` as evidence accumulates.
- `approximate_geodesic(phi_a, phi_b, reference_phi)`
- `llm_condition(snapshot)`, which returns **structured side information**,
  not generated language or a claim about identity.

The earlier `infer` interface expects standardized Beck feature vectors.
The new raw-history interface accepts chronological rows with the exact
`STATE` feature schema, plus a timestamp. The trained checkpoint carries
training-only imputation and scaling statistics. The demonstration script
`run_online_history_manifold_demo.py` uses development examples and writes
`artifacts/beck_behavior_pilot/online_history_runtime_demo.json`. It uses
pseudonymous labels rather than storing raw text.

From the repository root:

```powershell
.\.venv\Scripts\python.exe Ethan\run_beck_online_history_vae.py
.\.venv\Scripts\python.exe Ethan\run_online_history_manifold_demo.py
```

## Interpretation boundaries

The new model is trained across variable-length prefixes, so re-estimating
`phi` after a new observation is within its training contract. This is **not
yet a calibrated Bayesian filtering process**: history is reduced to
per-feature means rather than encoded as a full ordered sequence, and the
fast state has no trained transition model. In the one-prompt demonstration,
the 10-to-11-observation update changed the slow-posterior mean by 0.143 in
coordinate-Euclidean norm and the fast-state mean by 0.862. Those are
engineering outputs, not psychological magnitudes. Raw-history inference
reproduced the corresponding preprocessed-row probability exactly.

The metric uses a fixed zero reference for fast state and time, and the
graph geodesic depends on finite training-person coverage. Positive metric
eigenvalues at sampled points do not prove nonzero intrinsic curvature or
psychological meaning. Beck has no longitudinal text paired with these
people, so an LLM can currently receive typed estimates but has not been
trained to use them. The previous development cohort was inspected in
earlier work; these results are not confirmatory.

## Next implementation milestone

Upgrade the prefix-mean encoder to an ordered, masked history encoder and
train a transition model for `z`. Define the metric under several
prespecified reference contexts rather than one zero state, and check
posterior uncertainty calibration. Only after this should a consented
longitudinal text encoder be joined to the **same people** and a language
model conditioned on their revisable posterior.

## PsychArchives follow-on

The [PsychArchives Smartphone Sensing Panel dataset](https://psycharchives.org/en/item/975a7624-cd16-46bb-9d80-d988bed5e780)
reports 9,790 repeated situational snapshots from 455 people over 14 days,
plus person-level Big Five variables and mobile-sensing features. This is a
promising **person-level, multimodal extension** to Beck, but the release does
not advertise longitudinal participant text. Its files are marked
**Scientific Use License v1**, so license/ethics review and codebook inspection
should precede ingestion or redistribution. The user has now supplied the
licensed files locally, and a separate implementation has been trained;
see [PSYCHARCHIVES_MANIFOLD_IMPLEMENTATION.md](PSYCHARCHIVES_MANIFOLD_IMPLEMENTATION.md).
The Beck checkpoints discussed above remain Beck-only.
The follow-on [development evaluation](PSYCHARCHIVES_EVALUATION_12_RESULTS.md)
tests trait association, nonoverlapping-window person matching, and matched
component removals without opening the reserved PsychArchives test split.
The [geometry-training comparison](PSYCHARCHIVES_GEOMETRY_TRAINING_RESULTS.md)
then tests a decoder-path contrastive loss against the same Euclidean
identity objective, while retaining the test holdout.
The latest [feature-group and simple-baseline check](PSYCHARCHIVES_FEATURE_GROUP_RESULTS.md)
shows that current `phi` has not demonstrated personality information beyond
basic valence mean and variability on the reused development cohort.
