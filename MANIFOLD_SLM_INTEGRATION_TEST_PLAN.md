# Manifold-to-SLM Integration and Test Plan

Date: 2026-10-12. Status: **Draft for review before implementation.**

This document proposes the connection and the tests to write. It does not
implement an adapter, train a model, run the proposed experiments, or authorize
access to licensed participant data. Existing model behavior remains unchanged.

**Review order:** Sections 1-3 define the connection, Section 6 defines its
contract, Section 7 specifies the test cases, and Sections 8-10 define the
execution sequence and decisions required before coding.

## 1. Proposed Decision

Connect the Manifold ordered-history person model to the shared small language model
through a **separate, explicitly trained conditioning layer**. Freeze the Manifold
checkpoint in the initial experiment. Preserve the shared SLM's existing
token-only API and keep imports from personal research folders out of its core.

Use the current SLM as the first integration control. A Transformer-based SLM
remains the modernization target, but test that backbone replacement separately
so an architecture change cannot be mistaken for a benefit of person conditioning.
No pretrained models or external LLM calls are introduced by this plan.

The first question is narrow:

> Can a frozen history-derived state influence next-token probabilities through
> a learned bridge, without breaking causality, checkpoint compatibility, or
> the unconditioned SLM?

Passing that test is not evidence of a complete personality model. Separate
gates are needed for useful personalization and for any benefit from geometry.
The broader alternatives and paper rationale are in
[PERSONALITY_MANIFOLD_LLM_RESEARCH.md](PERSONALITY_MANIFOLD_LLM_RESEARCH.md).

## 2. Verified Starting Interfaces

These statements are based on source inspection, not a new runtime validation.

| Component | Current interface | Consequence for integration |
| --- | --- | --- |
| [Ethan/run_psycharchives_sequence_manifold.py](Ethan/run_psycharchives_sequence_manifold.py) | `infer_history(checkpoint, chronological_cues)` reads a checkpoint and one chronological history | It is stateless and reloads the model on each call; it does not group, sort, or persist a person's history |
| `SequenceManifold` in the same module | Missingness-aware causal history encoder, 4-D slow state, 4-D fast state, decoder, transition, and next-valence head | Pin the model family and preprocessing; arbitrary Manifold checkpoints are not interchangeable |
| [small_language_model/core.py](small_language_model/core.py) | `CausalLanguageModel.forward(tokens)` returns token logits; inherited `encode_tokens(tokens)` returns causal hidden states | There is no existing person-state argument, bridge trainer, or persistent person memory |
| [Alexander/model.py](Alexander/model.py) | `EmotionLanguageModel.forward(tokens, desired_vad)` conditions on three VAD values | Do not put the Manifold model's four-dimensional `phi` or `z` into `desired_vad` or call the private code personality |
| [small_language_model/README.md](small_language_model/README.md) | Shared scratch-trained language components are independent of student research | Integration belongs outside the shared core; the plain model is not automatically a trained dialogue model |

### Manifold Input Contract

The current public inference function expects a nonempty pandas DataFrame with
all columns in `CUES`, currently 22, in **raw native units**. It applies the
configured log transforms, saved imputation medians, and saved scaling before
inference. Input valence is on the 1-6 questionnaire scale, not VAD's scale.
Do not log-transform or standardize a second time before calling it.

The integration layer must check participant isolation and event ordering before
calling the Manifold model. An ordered DataFrame is an input precondition, not something the
current inference function verifies. The dataset's questionnaire order does not
supply elapsed time in hours.

### Manifold Output Contract

| Current key | Meaning | Proposed initial use |
| --- | --- | --- |
| `observations_seen` | Number of supplied history rows | Alignment and stale-state checks |
| `slow_phi` | Four slow posterior-mean coordinates | Conditioning features |
| `fast_z` | Four current fast posterior-mean coordinates | Conditioning features; separate slow-only ablation |
| `estimated_traits` | Five predicted trait readouts | Optional diagnostics, not independent labels or initial SLM inputs |
| `next_questionnaire_valence_estimate` | Scalar next-questionnaire prediction converted back to the questionnaire scale | Diagnostic only; not full VAD and not guaranteed to be bounded by the output head |
| `local_metric_eigenvalues` | Four eigenvalues of a ridged local decoder metric | Numerical diagnostics only; eigenvalues do not reconstruct the full metric |

The current function does **not** return posterior covariance/log variance,
the full Jacobian or metric, or a persistent identity handle. The initial bridge
uses point estimates and must not claim calibrated posterior uncertainty.

The function constructs a full-cue, transition-enabled `SequenceManifold`.
Start with a pinned ordered-sequence checkpoint and its preprocessing. Reject
unrecognized checkpoint families, reduced-cue variants, and incompatible
transition settings rather than attempting to load everything with a `.pt`
extension. Geometry-trained v2 checkpoints require a separately verified
compatibility rule and provenance; they are not the first integration target.

## 3. Proposed Connection

```mermaid
flowchart LR
    history["One person's allowed history through event t"] --> guard["Schema and ordering validation"]
   guard --> manifold["Frozen Manifold inference"]
   manifold --> state["slow_phi plus fast_z: 8 values"]
    state --> bridge["Trainable conditioning projection"]
    tokens["Prompt and causal response prefix"] --> slm["Shared SLM token states"]
    bridge --> combine["Conditioned hidden states"]
    slm --> combine
    combine --> head["Existing token head"]
    head --> output["Next-token probabilities"]
```

Use a separate wrapper around the plain `CausalLanguageModel`, not a silent
change to Alexander's affective model. With person state
$s_{i,t}=[\phi_{i,t};z_{i,t}]\in\mathbb R^8$, propose the smallest initial bridge:

$$
r_{i,t}=P_\omega s_{i,t},\qquad
\widetilde h_{i,k}=h_{i,k}+r_{i,t},\qquad
\ell_{i,k}=W_o\widetilde h_{i,k}+b_o.
$$

Here $k$ is token position, $t$ is history-event position, and $P_\omega$ maps
eight state values to the SLM hidden width. Derive that width from the selected
SLM configuration; do not hardcode the current default of 64.

The projection has no bias and starts at zero. An explicit disabled-conditioning
path must return the original SLM result. Zero is a numerical bypass, not an
average or psychologically neutral person. A fixed person snapshot is used
throughout one response, and only previously available evidence may define it.

Freeze **both** upstream models for the first bridge-only learning experiment:
only $P_\omega$ is optimized. Compute the Manifold state once per history snapshot,
not once per generated token. A later experiment may unfreeze the SLM, with an
equally trained no-state control, but end-to-end tuning of Manifold is out of scope
for the first integration.

**Known limitation:** with the current linear token head, this bridge adds a
person-dependent vocabulary bias that is constant across token positions. It is
a useful plumbing baseline, not the final rich personality mechanism. A learned
prefix or context-dependent adapter is a later, separately compared extension
if this simple bridge passes its tests but lacks expressiveness.

**Language checkpoint prerequisite:** the plain SLM needs an explicitly trained
language-only checkpoint before a language-quality experiment. Random weights
are acceptable only for wiring tests and controlled toy fixtures.
[Alexander/train.py](Alexander/train.py) exports an `EmotionLanguageModel`
checkpoint with additional research heads and a VAD-conditioned training
objective. Do not load that checkpoint into the plain class with missing or
unexpected parameters silently ignored. If that affective model is selected
instead, revise this plan explicitly and keep VAD separate and equally supplied
across comparison conditions.

## 4. Three Distinct Test Stages

| Stage | Data | What a pass means | What a pass does not mean |
| --- | --- | --- | --- |
| Wiring and regression | Synthetic histories, tiny vocabularies, synthetic checkpoints | Schemas, causality, isolation, serialization, and gradient routing work | Meaningful language or personality |
| Learnable synthetic control | Fixed artificial relationship between state and target tokens | The bridge can learn and use a known conditioning signal | Generalization to a real participant |
| Scientific personalization | Consented, participant-linked histories and held-out language or choices | Personal state improves a prespecified real outcome over controls | A whole person is captured, or nonlinear geometry is superior |

The active PsychArchives study has no participant-linked language. EmoBank
sentences and unrelated sensing histories must not be paired as if they came
from the same person. Synthetic associations are engineering fixtures only.
The scientific stage is blocked until the appropriate linked data and approvals
exist. See [Ethan/README.md](Ethan/README.md) for the current data restrictions.

## 5. Initial Acceptance Gates

1. **Contract:** the exact Manifold checkpoint, cue schema, preprocessing, SLM
   configuration, and vocabulary are pinned; invalid histories fail clearly.
2. **Regression:** disabled or zero-initialized conditioning preserves baseline
   logits; both event-history and token-prefix causality are preserved.
3. **Learning:** a deterministic synthetic task shows successful conditioning,
   while the same SLM without state cannot solve that state-dependent task.
4. **Reproducibility:** save/load reproduces logits and rejects mismatched
   upstream checkpoints or vocabulary ordering.
5. **Scientific validity:** report a real-person benefit only after linked-data
   evaluation with no-state, shuffled-state, and simple-history controls.

All gates above are **planned, not run**. Review the target SLM, upstream
checkpoint, and linked-data availability before approving implementation.

## 6. Integration Contract to Implement

The names in this section are proposed interfaces, not APIs that already exist.

### 6.1 Responsibilities and Ownership

| Proposed component | Responsibility | Boundary |
| --- | --- | --- |
| History adapter | Validate one history; call Manifold; normalize the output schema | Owns Manifold-specific imports and metadata; does not refit preprocessing |
| Person-state snapshot | Carry `phi`, `z`, history cutoff, and provenance | No raw observations or participant identifiers in public reports |
| Conditioned-SLM wrapper | Combine a validated state with causal token states | Does not import Manifold inside the shared language core |
| Paired-example loader | Align allowed history, prompt, response, and split | Separate from the existing EmoBank CSV contract |
| Bridge trainer/evaluator | Train the projection; run fixed controls and save aggregate results | Does not train Manifold or select on the reserved test cohort |

Place these components in a new integration package outside the shared core and
student-specific implementations. Reuse `unittest` and temporary synthetic
fixtures, consistent with the existing tests; do not introduce a new test
framework for this slice.

The Manifold sequence module currently uses script-style sibling imports such as
`from run_psycharchives_manifold import ...`. A root-level package import needs
an explicit compatibility decision and a smoke test; do not assume it works
because launching the Manifold script works. If package-relative imports are needed,
limit that follow-up to import compatibility and preserve existing script
commands. Do not introduce global path mutation throughout the shared core.

### 6.2 Snapshot and Tensor Schema

| Field | Proposed contract |
| --- | --- |
| `schema_version` | Explicit bridge schema version, initially 1 |
| `slow_phi`, `fast_z` | Each contains exactly four finite numeric values from the pinned checkpoint |
| `observations_seen`, `history_cutoff` | Count and last allowed event-order key; supplied history must match both |
| `manifold_checkpoint_sha256` | Fingerprint of the exact source checkpoint |
| `preprocessing_sha256` | Fingerprint covering cue order, log cues, medians, means, and scales |
| Local participant handle | Used for protected routing only; never a numeric language-conditioning feature |
| Batched `person_state` | Floating tensor shaped `[batch, 8]`, ordered `phi` then `z` |
| `tokens` | Integer tensor shaped `[batch, token_count]`, using the pinned SLM vocabulary |
| Projected offset | `[batch, 1, hidden_dim]`, broadcast across token positions, not across participants |
| Returned logits | `[batch, token_count, vocabulary_size]`, matching the plain SLM convention |

Construct state tensors on the SLM's device and with an explicit floating dtype.
Use CPU float32 as the first reference platform. Accelerator tests are optional
until that device is declared supported, and require their own numerical
tolerances. Do not infer hardware support from a paper or from installed PyTorch.

The bridge artifact must also bind the SLM checkpoint fingerprint, model class
and dimensions, exact vocabulary order and special-token IDs, tokenizer version,
state order, training settings, seeds, and any state normalization statistics.
If state normalization is introduced, fit it on training histories only and
save it; do not silently normalize each evaluation batch.

**Coordinate compatibility:** different Manifold seeds or retrained checkpoints
can rotate or rescale latent coordinates. A bridge fitted to one checkpoint
must reject another unless a new bridge is fitted or an explicit alignment is
validated. Matching the four-plus-four dimensions is not sufficient.

### 6.3 Input Validation Policy

These are **proposed wrapper guarantees**, not all existing checks in Manifold.

- Require one participant per call, a strictly increasing event-order key, and
   no duplicate event keys. Reject ambiguous order instead of silently sorting.
- Require the full expected cue schema; select values in the model's `CUES`
   order. Verify saved cue/log-transform metadata matches the supported producer.
- Accept missing cue values through the existing observed-mask mechanism.
   Reject an entirely unobserved history; permit a partly missing history.
   Reject positive/negative infinity rather than converting it into evidence.
- Reject finite valence outside 1-6, negative count/duration cues, malformed
   numeric fields, empty history, nonfinite learned state, and nonpositive or
   nonfinite saved feature scales. Validate preprocessing-array lengths.
- Do not pass batch padding rows into Manifold as real observations. For batching,
   compute each person's final valid state, not a GRU state after padded events.
- A deliberately absent state may use an explicitly requested unconditioned
   baseline. A corrupt checkpoint or malformed history must raise an error,
   not silently fall back to apparently personalized output.
- Load only trusted local artifacts; restricted weight loading is not a general
   security boundary for arbitrary external files.

### 6.4 Target Alignment and Training Loss

One future paired example contains a protected participant handle, allowed
history through event $t$, a prompt available after that cutoff, and that
participant's subsequent response. The response and any questionnaire values
derived from it are targets, never inputs to the same prediction. Store an
explicit alignment rule; do not invent timestamps for the current release.

[small_language_model/data.py](small_language_model/data.py) currently loads
sentence IDs, text, split, and VAD, not participant histories. Keep that loader
unchanged. The new paired loader needs participant-level splits, causal history
cutoffs, and a response-token loss mask. Do not invent VAD labels merely to fit
the old loader or Alexander's research loss.

Use next-token cross-entropy over the response targets, including the intended
end-of-sequence target, while excluding prompt-only positions and padding.
Check the one-position input/target shift explicitly. Reject examples with no
scorable response targets. Build vocabulary only from allowed training text;
keep `<pad>=0`, `<unk>=1`, `<bos>=2`, and `<eos>=3` consistent with the shared
loader. Log unknown-token rates so they cannot hide poor language coverage.

## 7. Detailed Test Matrix

Every test below has status **Not run: proposed for implementation**. Existing
tests listed in Section 8 are useful regressions, not evidence that the new
integration already passes. For deterministic CPU float32 comparisons, propose
`atol=1e-6` and `rtol=1e-5`; freeze the tolerance policy before recording results.

### 7.1 Contract and Preprocessing

| ID | Setup or action | Expected result / pass criterion |
| --- | --- | --- |
| C01 | Import the integration entry point from the repository root; run the Manifold model's existing test-script entry points | Both import paths work; no training, source-data loading, or artifact writes occur merely from imports |
| C02 | Save a tiny synthetic full-cue sequence checkpoint with known preprocessing, then infer one history | Wrapper `phi`, `z`, and count match direct `infer_history` within tolerance; state shape is `[1, 8]` |
| C03 | Use nontrivial medians, scales, a count requiring `log1p`, and a missing cue; compare with an independently calculated preprocessed reference | Log transform occurs once, scaling uses saved values, and the observed mask stays zero for the missing cue |
| C04 | Supply empty history, missing columns, invalid valence/counts, infinity, bad saved scales, or an incompatible/reduced-cue checkpoint | Explicit contract error before generation; no silent fallback or partial weight loading |
| C05 | Supply duplicate/out-of-order events, mixed participant handles, or a future row beyond the declared cutoff | Reject the example; do not infer from ambiguous or disallowed history |
| C06 | Test a one-row valid history, partly missing history, entirely missing history, and reordered input columns | One-row/partly missing inputs work; entirely missing history is rejected; column order does not change the selected cue order or result |

For C03, obtain the reference from the raw synthetic values and declared
transformations, not by calling the wrapper's preprocessing twice. Otherwise
the test could reproduce the same error on both sides.

### 7.2 SLM Behavior, Causality, and Isolation

| ID | Setup or action | Expected result / pass criterion |
| --- | --- | --- |
| B01 | Same tokens and base weights with conditioning disabled, and with the zero-initialized projection | Logits equal the original plain SLM within tolerance |
| B02 | Use a hand-constructed nonzero projection and token-head contrast; change only `phi`, then only `z` | Selected logits change in the analytically expected direction; do not rely on arbitrary random weights or different sampled sentences |
| B03 | Change a later token while keeping state and earlier prefix fixed | Earlier logits do not change; preserve this invariant when a Transformer is substituted |
| B04 | Add/change a later event, but evaluate the same earlier history cutoff | Earlier state and logits do not change; evaluation never recomputes that cutoff from the full future history |
| B05 | Compare three participants separately versus a batch of size 3 and token length 5; permute the batch | Corresponding logits match; no accidental broadcasting, order dependence, or participant-state mixing |
| B06 | Append right padding; compare different history lengths batched and unbatched | Valid-token logits and loss agree; final person states come from real events, not history padding |
| B07 | Predict for A, then B, then A; append a real event to A | A's unchanged snapshot reproduces its output; only A's new version is recomputed; no hidden shared identity state |
| B08 | Evaluate invalid state width, nonfinite state, out-of-range token IDs, and mismatched token/state batch sizes | Clear validation error; valid conversions to the declared device/dtype are tested explicitly |

Updating a history need not change every output for every possible input.
For the update-sensitivity part of B07, use a deliberately informative synthetic
event and known responsive fixture. Otherwise assert freshness and causality,
not that every real observation must move personality coordinates.

### 7.3 Training and the Positive Control

| ID | Setup or action | Expected result / pass criterion |
| --- | --- | --- |
| L01 | Backpropagate and take one bridge optimizer step on an informative toy batch | Projection has finite nonzero gradient and changes; Manifold and SLM parameters and buffers remain unchanged and are absent from the optimizer |
| L02 | Check response-token masks, target shifts, padding, and an all-masked example against a hand-computed loss | Valid loss matches reference; all-masked examples fail instead of producing NaN or an empty mean |
| L03 | Run the fixed synthetic conditioning task below | Correct-state model reaches the prespecified toy accuracy and reduces held-out loss; state-free control remains unable to identify the state-dependent label |
| L04 | Reevaluate L03 with shuffled state, and with conditioning disabled | Correct-state performance exceeds shuffled/disabled controls; report the actual gap and all fixed seeds |

**Synthetic task specification:** generate artificial eight-dimensional states
and a balanced binary target from a fixed, known linear rule. All examples use
the same prompt; labels are two ordinary, explicitly included vocabulary tokens,
such as `left` and `right`, with no psychological interpretation. Separate
training and evaluation states, exclude points arbitrarily close to the rule's
boundary, and give the frozen toy SLM head a known nonzero contrast between the
two tokens. This makes the fixture intentionally solvable by the proposed
linear bridge. Propose at least 95% held-out accuracy for each of three fixed
seeds as a software acceptance target, not a projected real-person result.

For the balanced identical-prompt task, a deterministic state-free classifier
has only 50% accuracy; use the same base SLM and fixed evaluation examples.
Random shuffling need not yield exactly 50% in a finite sample. Predeclare a
fixed shuffle or repeat fixed permutations and report variability. Do not tune
the shuffle or select only successful seeds.

This task tests bridge learning in isolation. C02 and the end-to-end test below
separately exercise the real Manifold producer using synthetic raw histories.
It is not necessary or legitimate to assign psychological meaning to synthetic
`phi` values to verify a tensor connection.

Freezing a model's weights must not detach the bridge's path through the token
head. Conversely, the Manifold diagnostic Jacobian still requires differentiation
with respect to its coordinates; do not surround the entire `infer_history`
call with a global inference-mode block that prevents that calculation.

### 7.4 End-to-End, Artifacts, and Privacy

| ID | Setup or action | Expected result / pass criterion |
| --- | --- | --- |
| E01 | Synthetic raw history -> saved Manifold checkpoint -> snapshot -> wrapper -> logits -> short generation | Finite outputs, correct shapes, valid token IDs, valid EOS/max-length termination; generation uses the same conditioning as training and one fixed history snapshot |
| E02 | Save the bridge artifact and reload it with the same upstream models and vocabulary | Teacher-forced logits and deterministic generation reproduce; one modified upstream weight or reordered vocabulary is rejected |
| E03 | Present a different Manifold seed/checkpoint with identical tensor dimensions, or stale preprocessing/history metadata | Reject incompatible state/bridge pairs and stale snapshots; dimensions alone do not authorize reuse |
| E04 | Repeat a seeded CPU run and inspect optimizer membership and output manifest | Settings and results reproduce within the declared policy; frozen-model fingerprints do not change |
| E05 | Capture logs and committed outputs from a synthetic run; test an unauthorized participant route | No individual histories, coordinates, identity handles, or generated personal text enter public logs; unauthorized routing fails before inference |
| E06 | Remove a person's stored evidence/state and request the old snapshot | That state cannot be retrieved or reused; no cache retains it under an obsolete version |
| E07 | Profile history inference and response generation separately over increasing history/token lengths | Report latency and peak memory; verify Manifold is not called once per generated token; choose a resource budget before a performance pass/fail claim |

Use temporary files for test checkpoints. The initial implementation can omit
caching entirely. If caching is later added, its key must include the local
participant handle, history version/cutoff, and checkpoint/preprocessing
fingerprints; extend E03, E05, and E06 before enabling it.

Keep participant-level states, generated responses, and checkpoints trained on
licensed observations in approved local storage only. Git-ignore protection
does not itself grant permission to store or share them. Deleting a state does
not unlearn examples already used in shared weights; training-data withdrawal
needs a separately defined retention/retraining policy.

## 8. Execution Sequence After Approval

| Phase | Work to perform | Exit gate |
| --- | --- | --- |
| 0: Lock the contract | Choose the plain SLM route, a supported Manifold checkpoint, import strategy, interpreter, and data permissions | Contract reviewed; checkpoint and vocabulary expectations written down |
| 1: Capture regressions | Run the existing synthetic unit tests; create synthetic history/checkpoint fixtures | Record existing failures separately; do not attribute them to the new bridge |
| 2: Build the bridge | Implement history validation, snapshot, wrapper, and save/load contract | C01-C06, B01-B08, L01-L02, and E01-E06 pass |
| 3: Prove learning | Train only the projection on the fixed artificial task; profile the complete path | L03-L04 pass; E07 recorded; describe the result only as an engineering success |
| 4: Prepare real evaluation | Establish a trained plain-SLM baseline and consented, linked language/behavior data | Split, target, practical effect threshold, and analysis locked before test use |
| 5: Evaluate personalization | Run the controlled comparisons in Section 9 | Report held-out results and uncertainty, including negative results |
| 6: Investigate richer mechanisms | Compare prefix/context-dependent conditioning, then a Transformer backbone, then any geometry-specific use | One experimental factor changes at a time; rerun the applicable regression matrix |

Phases 1-3 do not require licensed source data. A permitted local-checkpoint
smoke test can follow them, but is not a substitute for linked-data validation.
Do not start training or data collection merely because this draft exists.

### Existing Regression Commands

Run from the repository root **after implementation is approved**, with `python`
bound to the selected project environment and dependencies available. These
commands refer to existing tests; none was executed while drafting this plan.

```sh
python -m unittest discover -s small_language_model -t . -v
python Ethan/test_psycharchives_sequence_manifold.py -v
python Ethan/test_psycharchives_geometry_training.py -v
python -m unittest discover -s Alexander -t . -v
```

The first three cover the primary touched contracts. Run Alexander's suite as a
compatibility gate if shared language code or imports change. Its current
affective outputs must not be silently redefined by this integration.

Existing tests to extend or reuse as patterns:

- [small_language_model/test_core.py](small_language_model/test_core.py): token
   causality and the plain model's language-only module boundary.
- [small_language_model/test_data.py](small_language_model/test_data.py):
   train-only vocabulary, token alignment, and invalid-data fixtures.
- [Ethan/test_psycharchives_sequence_manifold.py](Ethan/test_psycharchives_sequence_manifold.py):
   history causality, missing-cue masking, order sensitivity, and padded-pair masks.
- [Ethan/test_psycharchives_geometry_training.py](Ethan/test_psycharchives_geometry_training.py):
   independent windows, normalized path-loss scaling, and geometry-gradient flow.

Add a separate integration test entry point during implementation and document
its real command then. The test IDs above are specifications, not already
available test names or CLI options.

## 9. Scientific Evaluation, Conditional on Linked Data

### Data Access and Splits

Split participants before fitting any bridge or normalizer. For held-out
participants, use only an allowed adaptation-history prefix and score later,
disjoint responses. Report known-person future prediction separately from
new-person adaptation. Include held-out situation categories when the sample
supports it. Audit topics and missingness/device patterns as possible shortcuts.

The existing PsychArchives 69-person test cohort remains sealed for this
planning and software-test work. Do not repurpose it for an unplanned language
study; the selected release has no linked participant language. A new cohort or
properly licensed, consented linked dataset needs its own frozen protocol.

### Minimum Comparisons

| Condition | Question isolated |
| --- | --- |
| Same frozen SLM, no state | Does conditioning add to the language-only reference? |
| SLM plus simple history summaries | Do mean/variability or other prespecified summaries explain the improvement? |
| SLM plus slow `phi` only | Is the slow representation useful without current fast state? |
| SLM plus fast `z` only | Does current-state information alone explain the result? |
| SLM plus `phi` and `z` | Does the combined representation improve on the simpler alternatives? |
| Same trained model with participant state shuffled within appropriate context strata | Are outputs meaningfully dependent on the correct participant's state? |

Hold language checkpoint, tokenizer, prompt construction, available past
information, split, bridge training budget, and seeds fixed. Treat state
ablations as separately trained variants; use post-training shuffling as an
additional reliance diagnostic. Report effective parameter counts for controls.

Propose held-out response-token negative log likelihood, averaged per
participant, as the primary language endpoint. Include unknown-token rate and
response length. Where real choice targets exist, add choice accuracy and a
proper probabilistic score such as log loss or Brier score. Use blinded
participant judgments of fidelity as a secondary measure, not the only evidence.
Lower text loss does not by itself validate personality as a psychological
construct.

Use multiple fixed seeds and participant-level uncertainty estimates rather
than treating every sentence as an independent person. Before confirmatory
evaluation, choose a minimum practically meaningful improvement and a
participant-clustered confidence-interval rule. The scientific gate cannot be
marked passed until these are prespecified and met; there is no justified
numerical real-person success threshold yet.

### Keep the Geometry Claim Separate

Passing `phi` and `z` to an SLM uses a learned representation. It does **not**
mean the SLM uses the Manifold metric, follows a geodesic, or benefits from curvature.
The initial plan deliberately excludes metric eigenvalues from token inputs.

If a later study compares Euclidean-identity and decoder-path-identity Manifold
checkpoints, use corrected v2 provenance, fit a fresh bridge to each checkpoint,
and match the data and training protocol. Evaluate the same held-out outcome.
The current distance is sampled straight-coordinate-path length, not an
optimized geodesic. Existing results do not establish a distinct geometry
advantage; see
[Ethan/PSYCHARCHIVES_GEOMETRY_TRAINING_RESULTS.md](Ethan/PSYCHARCHIVES_GEOMETRY_TRAINING_RESULTS.md)
and
[Ethan/PSYCHARCHIVES_FEATURE_GROUP_RESULTS.md](Ethan/PSYCHARCHIVES_FEATURE_GROUP_RESULTS.md).

If no-state or simple summaries match the new model, report no demonstrated
incremental benefit. If a prefix is needed to express context-dependent effects,
test it as a conditioning-mechanism change, not evidence that the original
manifold improved. If Euclidean and decoder-path versions are indistinguishable
within uncertainty, do not claim geometric superiority.

## 10. Deliverables and Review Checklist

After implementation, produce the tested integration entry point, synthetic
fixtures, test results keyed to C/B/L/E IDs, a versioned bridge artifact, and an
aggregate run report. The report should identify condition, seeds, code and
model fingerprints, preprocessing/tokenizer versions, dataset/split provenance,
training settings, device, tolerances, metrics, resource measurements, and any
failed or skipped gate with its reason. Never populate missing results with
assumed success or publish participant-level evidence in that report.

Before beginning code, resolve:

- [ ] Confirm the plain SLM is the initial target; otherwise explicitly specify
   the affective-model variant and independent VAD policy.
- [ ] Select the supported Manifold producer/checkpoint and approve a narrow import
   compatibility change if required.
- [ ] Approve the separate wrapper, initial frozen-model policy, and simple
   projection as an integration baseline.
- [ ] Agree on synthetic fixtures, numerical tolerances, toy learning threshold,
   and selected Python environment.
- [ ] Record that Transformer modernization is a separate experiment, not
   removed from the research direction.
- [ ] Confirm licensed-data handling and whether participant-linked language is
   available; keep the real-person stage blocked when it is not.
- [ ] Prespecify the real evaluation target, split, meaningful effect, and
   uncertainty rule before any confirmatory test is opened.

**Ready-to-code scope:** implement and verify the connection with synthetic
data first. **Not yet justified:** claiming that the connected SLM represents
a complete personality or that manifold geometry improves its predictions.