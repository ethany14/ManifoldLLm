# Development-only feature-group sensitivity protocol

This experiment asks whether the slow coordinate `phi` depends on mobile
usage and sensing features that could act as a person/device fingerprint.
The protocol is fixed before this run, but the development cohort has been
inspected repeatedly, so results are exploratory. The 69-person test split
remains reserved.

Use the same licensed PsychArchives files, person split (seed 42), ordered
questionnaires, training-only median imputation and standardization,
`SequenceManifold` architecture (GRU hidden size 32, slow and fast dimensions
4 each), and next-questionnaire valence target. Fit three seeds (1, 2, 3)
for exactly 30 epochs with AdamW (learning rate 0.002, weight decay 0.001)
and 32-person minibatches. Keep reconstruction, transition, slow-stability,
KL, and next-valence losses. Set the auxiliary trait-loss weight to **zero**;
traits must not shape the trained representation. Add the same Euclidean
same-person contrastive loss to every group: weight 0.02, temperature 0.2,
differentiable mean-off-diagonal distance normalization, and independent
first-five/last-five training windows. Only the input cue group changes.

Four nested input groups, all drawn from the previous 22-cue selection:

1. `full_22`: all 21 sensing/time cues plus current valence.
2. `activity_time_valence_9`: three activity-probability cues, five
   timestamp status flags, and valence. This removes App, Phone, Screen,
   GPS, Notification, Power, and Wifi cues and their missingness patterns.
3. `time_valence_6`: five timestamp status flags and valence; no mobile
   sensing values or sensing missingness indicators.
4. `valence_only_1`: valence alone. This assesses whether a person's
   affective-response sequence carries a repeatable signature by itself.

Each group retains the same people and outcome pairs. Fit preprocessing
separately on training rows for that group's columns. Network input and
decoder dimensions necessarily change with the number of cues; hidden
widths and training budget stay fixed. This is a feature-sensitivity
study, not a parameter-count-matched causal isolation.

For each group and seed, report: next-valence RMSE; five per-trait and macro
development R² plus pooled RMSE from a fixed `Ridge(alpha=1)` probe fitted
on complete-trait **training** people; and rank-1 same-person matching plus
within/between Euclidean distance ratio using independent first-five and
last-five development windows. The development trait subset and its
training-mean baseline are identical across groups. The rank-1 chance rate
is 1/68. Keep per-seed values; do not tune features or select a group by
one metric. No participant IDs, source rows, or individual coordinates are
written to aggregate results.

If identity matching falls sharply when usage/sensing cues are removed,
that is consistent with dependence on those cues, **not proof** that the
original model encoded a device fingerprint. If trait prediction remains
or improves with valence/time alone, that supports examining affective
trajectories further, but still does not validate a personality manifold.
