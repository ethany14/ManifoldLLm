# Personality–Affect Dataset Bridge — YinpingYin

**An English-only, GitHub-ready research catalog of 63 dataset leads and reproducible schema tools.**

## Objective
Connect a **stable personality representation** (questionnaire-based traits or personality-derived embeddings) with a **dynamic affective representation** (emotion categories, VAD coordinates, conversational trajectories). Evaluate whether joint representations improve emotion prediction and behavior modeling compared with independent or concatenated baselines.

**Scope:** This repository is an index and experimental scaffold, **not 63 redistributed raw datasets**, and it is **not a trained or validated joint manifold**. Existing encoder interfaces have not been integration-tested.

## Files
- `datasets/catalog.csv`, `catalog.json`: 63 candidate resources, source links, fit, risks, verification status and data permissions caveats.
- `docs/YinpingYin_Dataset_Explanation_EN.pptx`: English research presentation.
- `docs/DATA_DICTIONARY.md`: output fields and data provenance rules.
- `docs/SOURCES_AND_ETHICS.md`: access, copyright and consent requirements.
- `scripts/inspect_catalog.py`: category filtering.
- `scripts/normalize_csv.py`: local conversion of an authorized dataset using explicit column mappings.
- `scripts/validate_bridge.py`: label and schema checks.
- `examples/synthetic_bridge.csv`: artificial rows for testing only.

## Dataset tiers
1. **Direct joint evidence:** ETC Dataset, CPED, PELD, and other genuinely speaker-linked resources. These differ greatly in whether trait ratings come from human questionnaires or fictional-character annotation.
2. **Personality / dialogue transfer:** RealPersonaChat and personality/text corpora. Emotion may be missing and must remain missing unless estimated by a model, then labeled `predicted`.
3. **Affect-only resources:** EmoBank, GoEmotions, MELD, RECCON and other affect benchmarks. These **do not automatically share identities** with the personality data.
4. **Repeated / multimodal measures:** Additional candidates for follow-up experiments, subject to access and field verification.

## How to use
```bash
python scripts/inspect_catalog.py --category joint
python scripts/validate_bridge.py examples/synthetic_bridge.csv
python scripts/normalize_csv.py --input my_authorized_data.csv --output data/processed/bridge.csv --dataset ETC --person-id participant_id --text utterance --emotion emotion_label --emotion-kind observed --personality big_five_json
```

## Experimental design
- Split by participant or by independent fictional character **before** model fitting to avoid leakage.
- Fit separate personality and affect encoders; compare emotion-only, personality-only, concatenation, MLP fusion, and geometry-aware fusion.
- Keep *measured*, *annotated*, *predicted*, and *missing* targets distinct; do not convert a categorical label to a unique VAD coordinate without a validated mapping.
- Report macro F1, VAD MAE/CCC (when measured VAD exists), identity-holdout transfer, calibration and uncertainty.
- Do not claim digital-personality validation or superior manifold geometry without held-out results.

## Publication and attribution
This project contains **metadata and tools**, not third-party raw corpora. Each source retains its own terms. The project-level MIT license does not apply to upstream datasets. Do not commit sensitive content or dataset files that cannot be redistributed. Upload the extracted folder to `ManifoldLLm/YinpingYin_Personality_Affect_Bridge_EN/` using a review branch and pull request.
