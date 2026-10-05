# Development-only personality evaluation and ablation protocol

This protocol was fixed before running the evaluation below. The development
set has already been inspected in earlier experiments, so these results are
exploratory, **not** preregistered confirmation. The 69-person test split
remains sealed. Source CSVs stay local under their scientific-use license.

## Fixed cohort and training

- Reuse `load_data`, `split_people`, and `prepare` from the PsychArchives
  sequence implementation. The split is by person (318 train, 68 development,
  69 reserved test), seed 42; all 22 cues, missingness indicators, and
  train-only imputation/standardization are unchanged.
- Fit three seeds (1, 2, 3) per training condition for exactly **30 epochs**.
  This budget was chosen after previous development inspection, so it cannot
  confer confirmatory status. Use the same minibatch size (32 people),
  optimizer (AdamW, learning rate 0.002, weight decay 0.001), and random seed
  across matched conditions. No development-based checkpoint selection.
- Conditions: full sequence model; `no_trait_loss` (set auxiliary trait-loss
  weight to zero); `no_transition` (pass inferred current `z` directly to the
  valence head and omit transition-consistency loss). The unused transition
  parameters in `no_transition` are initialized but receive no gradient.
  Thus these are component-removal diagnostics, not exact parameter-matched
  causal estimates.

## Fixed development readouts

1. Next-questionnaire valence RMSE and MAE, both on the original 1-to-6
   response scale. This checks the fast prediction task but is **not** the
   criterion for calling `phi` personality.
2. Five trait outcomes, separately for NPAR (emotional stability), EPAR,
   OPAR, CPAR, APAR. For each condition, take the final slow coordinate of
   each train/dev person. Fit one `Ridge(alpha=1)` linear probe using the
   training people with complete traits only. Evaluate each trait's R² and
   RMSE on the 57 development people with complete traits. Report the
   training-mean trait predictor on those same people. This probe is an
   *assessment* of retained trait information, not part of model training.
3. Independent-window stability/individuality: encode the **first five**
   and **last five** prompts of every development person separately (no
   overlapping records, recurrent state reset). Calculate the within-person
   versus cross-person median-distance ratio and rank-1 matching accuracy
   of early windows to late windows. The chance matching rate is 1/68.
   Stable individual structure would require a lower ratio and matching
   above chance, but context effects can make this task difficult.
4. Geometry-readout control on the *same trained full-model coordinates*:
   repeat early-to-late matching using Euclidean distance versus eight-step
   decoder straight-path length, with fast state fixed to zero. The latter
   approximates length under the decoder pullback metric along a straight
   coordinate path; it is **not** an optimized geodesic. The current metric
   is post-training and absent from the loss, so switching readouts does
   not constitute a training ablation or establish manifold benefit.

For each metric retain per-seed values and aggregate mean or median. Do not
pick a winner from a single seed or a single trait. R² may be negative.
Individual-level coordinates, traits, and participant IDs must not be
written to the repository or to the aggregate JSON. The loader checks all
source rows, and complete-trait status is used to stratify the person split.
After splitting, reserved test rows and trait values are not placed in model
blocks, used for fitting/selection, or scored.
