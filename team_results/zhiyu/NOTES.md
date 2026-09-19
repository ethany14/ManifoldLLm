# Zhiyu Baseline

- Dataset: complete EmoBank processed set
- Split: official 8,062 train / 999 dev / 1,000 test
- Frozen model: Qwen/Qwen3-4B
- Representation dimension: 16
- Seed: 42
- Result source: `artifacts/emobank_full/ablation_seed_42`

The strongest predictive result was AE + VAD + Geometry (mean VAD R2 = 0.302),
closely followed by AE + VAD (0.301). The geometry term improved neighbourhood
trustworthiness but supplied little incremental affect-prediction value.
