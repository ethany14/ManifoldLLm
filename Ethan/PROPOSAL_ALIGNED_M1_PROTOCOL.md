# Proposal-aligned M1 protocol / 与 proposal 对齐的 M1 实验协议

Status: **prospective protocol, not a completed experiment**. This document follows Sections 3, 5–7, and Appendix A of `Prereport_Awakening_AI_Mind_Digital_Personality_Manifold_CN (1).pdf` (25 August 2026). The proposal is a research plan; its success criteria are not findings.

## Decision / 方法决定

The project's first formal manifold model is **M1: a decoder-induced Riemannian personality manifold with separate slow person variable `phi` and fast state variable `z`**. It is not enough to use an autoencoder, draw a UMAP plot, or add a graph-distance penalty and call the result M1. The current Beck graph-geodesic regularizer and [decoder-metric development check](BECK_DECODER_METRIC_DEV.md) are exploratory engineering proxies, not M1.

本项目首个正式流形模型应是 **M1：具有 `phi`–`z` 慢快分离、由解码器诱导局部度量的黎曼人格流形**。普通 AE、二维可视化或图距离正则项本身都不能替代 M1。当前 Beck 图测地实验仅是方法原型。

## Exact comparison ladder / 原文对照顺序

| ID | Model | Increment isolated | Readiness |
|---|---|---|---|
| B0 | Big Five + linear/mixed-effects | Psychometric/vector baseline | Requires separately scored Big Five anchors; sparse momentary items are not automatically a validated trait score. |
| B1 | Flat VAE + static personality latent | Nonlinear probabilistic compression | Not yet implemented on person-linked Beck data. |
| B2 | Flat VAE + `phi`–`z` slow–fast decomposition | Stable person vs momentary state | Prior deterministic slow–fast pilots are approximations, not this VAE. |
| **M1** | **Riemannian personality manifold + `phi`–`z`** | Decoder-induced local metric and geodesic use | Formal target of the next model experiment. |
| M2 | Multi-chart manifold + `phi`–`z` | Multiple local coordinate charts | Only if single-chart M1 fails coverage/transition diagnostics. |
| M3 | Hybrid/stratified manifold + event dynamics | Discrete modes and event bridges | Only with observed role/event transitions and sufficient data. |

Isomap/graph Laplacian are useful **diagnostics or alternative geometry estimators**, not substitutes for this Appendix A model ladder.

## Data gate / 数据门槛

The proposal's proof of concept calls for consented longitudinal observations with context, psychological anchors, and independently assessed or repeatable behavioral outcomes; it sketches 60–100 volunteers for 4–6 weeks with 3–5 short ESM prompts per day, weekly measures, structured tasks, and optional language data. The exact sample size requires a power analysis after the primary endpoint is fixed. These are proposal suggestions, not completed recruitment.

Beck provides repeated people, situations, affect, and **self-reported** future activities. It can support an exploratory B2–M1 engineering comparison, but lacks text and independently observed action; its current test participants have already been examined. EmoBank and dair-ai/emotion contain isolated sentences and cannot test person-level identity. PersDyn provides repeated self-rated personality states but no independent behavior. Thus none of our existing results meets the full proposal's first-paper success criteria.

## Minimum M1 implementation / 最小 M1 实现

1. `h_it = encoder(x_it, c_it)`. From observations available **before prediction time**, infer a posterior `q(phi_i | history_<=t)` for a slow person variable and `q(z_it | phi_i, h_i,<=t, c_it)` for a fast state. Use one fixed dimension and matched input information across B1/B2/M1 for the first comparison. The person's posterior, not a single point, is the uncertainty-bearing identity representation.
2. Train an observation/state decoder `g(phi,z,c)` and a future-behavior head `p(y_i,t+1 | phi,z_it,c_it)`; add a short-window stability penalty on `phi`, while allowing `z` to respond to context. Keep future outcomes and future context out of the encoder at prediction time. A simple GRU/state-space transition is sufficient initially.
3. For M1, define a positive-definite local metric on the **personality coordinate** from the trained decoder, for example `G(phi) = E_(z,c)[J_phi g(phi,z,c)^T W J_phi g(phi,z,c)] + eps I`. Predefine the reference context distribution, weights, and `eps`; check Jacobian rank, eigenvalues, condition number, and behavior in low-density regions. A single scalar Bernoulli outcome is insufficient to identify a full-rank metric, so use a multidimensional state/behavior decoder or several outcomes and contexts.
4. Approximate geodesic distance by integrating path length under `G`, with path optimization or a local graph whose **edge lengths use the learned metric**. Report approximation accuracy and disconnected/low-density cases. Use that distance in a prespecified M1 prediction or regularization component; otherwise the same B2 predictor with a post-hoc distance analysis is only a geometry diagnostic, not an incremental model test.
5. Train B2 and M1 from matched seeds with the same encoder/decoder capacity, information, optimizer budget, and selection rule. M1 differs only in its geometric operation. Record ablations replacing geodesic distance with Euclidean distance and disabling the metric component.

This pull-back construction is a **proposed project implementation**, consistent with the geometric idea in Arvanitidis et al. (2018), not a result that paper established for personality. M2/M3 add complexity only after M1's limitations are observed, rather than being chosen by attractive plots.

## Locked evaluation / 预注册式评价

- Primary endpoint: held-out **future behavior** likelihood/Brier (for binary outcomes) or a predeclared task-appropriate score. Report class prevalence and participant-cluster confidence intervals. The proposal also calls for a separate within-person future-time test and new-person generalization.
- Primary contrasts: `M1 − B2` isolates the geometric increment; `B2 − B1` isolates slow–fast structure; `M1 − B0/B1` tests whether the full model improves over basic controls. Use paired predictions on the same people/time windows.
- Geometry validity: compare geodesic vs Euclidean person distances on an **external behavioral-similarity task** not used to fit the metric. Also report neighborhood preservation, stability across time, and uncertainty/OOD behavior. A 2-D plot is not evidence of curvature or behavioral validity.
- Secondary tests: `phi` stability over short windows, `z` responsiveness to context, psychometric convergent/discriminant validity, and counterfactual consistency where interventions are actually observed. M2 requires chart-transition consistency; M3 requires event/change-point evidence.
- Model selection uses development people only. The existing Beck test has been reused and is **exploratory**. A final claim requires newly held-out participants or an independent cohort, with data-collection and use consent.

## Current decision rule / 当前结论规则

We **will implement M1** because manifold is a project requirement. We will **not claim manifold superiority** unless M1 improves future-behavior prediction over B2 and B0/B1 on an untouched evaluation and geodesic distance adds external validity over Euclidean distance. If M1 fails, the publishable finding is a carefully tested limit of manifold benefit under the specified data/task, not a relabeling of a flat latent model.

我们**会实现 M1**，因为 manifold 是项目要求；但只有在未使用过的样本上超过匹配的 B2 与基础对照，并证明测地距离有外部行为增量效度，才主张“流形更好”。若未通过，应如实报告局限。

## Provenance

- Project proposal: *Awakening AI Mind: 数字人格流形与行为—认知数字孪生*, 25 August 2026, especially Sections 3, 5–7 and Appendix A.
- Existing results: `BECK_BEHAVIOR_PILOT.md`, `BECK_GRAPH_GEOMETRY_RESULT.md`, `PERSDYN_SLOW_FAST_COMPARISON.md`, `SUPERVISOR_METHOD_REPORT_BILINGUAL.md`.
- Geometry methods: [Tenenbaum et al., Isomap (2000)](https://doi.org/10.1126/science.290.5500.2319); [Arvanitidis et al., Latent Space Oddity (2018)](https://arxiv.org/abs/1710.11379); [Arvanitidis et al., Pulling Back Information Geometry (2022)](https://proceedings.mlr.press/v151/arvanitidis22b.html).
