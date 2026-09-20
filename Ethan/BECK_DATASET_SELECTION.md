# Dataset for the next person-situation experiment

## Decision

Use the openESM harmonization of Beck and Jackson (2022), *Personalized Prediction of Behaviors and Experiences: An Idiographic Person-Situation Test*, as the next **non-text longitudinal benchmark**. It is closer to our research question than EmoBank or PersDyn because it links repeated person-level personality and affect states, situation ratings, timestamps, and self-reported activities/experiences. It is still **not** a dataset of conversations or independently observed behavior, so it cannot by itself validate an LLM digital personality.

Primary sources: [paper](https://doi.org/10.1177/09567976221093307), [original open materials](https://osf.io/8ebyx/), [openESM dataset page](https://openesmdata.org/datasets/0060_beck/), and [versioned Zenodo record](https://doi.org/10.5281/zenodo.17361673). Cite both the original study and openESM/Zenodo when using the harmonized file. The Zenodo record states **CC BY-NC 4.0**; verify that the team's intended use is noncommercial. The original OSF node does not show a license in its API metadata, so do not assume the raw archive has the same redistribution terms. Raw and harmonized downloads remain gitignored.

## Files already placed locally

All are under `data/person_situation/source/` and excluded by the repository `.gitignore`:

| File | Role | Verification |
|---|---|---|
| `0060_beck_ts.tsv` | Preferred modeling table from openESM | 2,473,213 bytes; MD5 `eb1a2855b1db31c4af57be1515ce15b1` |
| `0060_beck_codebook.xlsx` | Harmonized variable definitions | MD5 `96c1413dfb442b49fb26b81bbf3a441c` |
| `materials_and_data.zip` | Original OSF archive with scripts and raw deidentified data | 65,550,997 bytes; SHA-256 `822493eb024df0a362bef4597deb9c77431c84a24506f4c3b4febad3b941dcfe` |
| `codebook.xlsx` | Original OSF codebook | SHA-256 `1a9238c6a53380a3a7bd83b7d0482bc005d74b021e9a7e8c58681286934b39d6` |
| `esm_cleaned_combined_2021-04-07.RData`, `baseline_05.07.20.csv` | Selected original archive files for provenance checks | Extracted unchanged from `materials_and_data.zip` |

The preferred TSV was inspected read-only: **8,672 rows, 107 columns, 199 participant IDs**, with no duplicate `(id, date, hour, minute)` keys. There are 8,665 nonmissing entries for each of `procrastinating`, `studying`, and `lonely`. Relevant fields also include `day`, `date`, `hour`, `minute`, `hour_block`, momentary Big Five items, ten affect items, eight DIAMONDS situation ratings, and binary daily activity/social-context indicators. The [openESM codebook](https://openesmdata.org/datasets/0060_beck/) defines those outcomes as binary self-reports, not sensor-confirmed actions.

**Count reconciliation is required before comparison with the article.** The published analysis reports 104 selected participants and 5,971 assessments, whereas the harmonized TSV contains 199 IDs and 8,672 rows. The original raw RData likewise contains 199 IDs before the article's analysis exclusions. We must reproduce the article's inclusion and timing rules from its scripts or explicitly define a new cohort and report that it differs. Do not quote the paper's N as the size of our unfiltered TSV.

## Proposed locked task before model training

Primary outcome: `studying` at the **next observed prompt**, because it is a behavior report rather than an affect rating. Secondary: `procrastinating`; `lonely` is an experience, not a behavior. Inputs at time t may include only observations available by t: current and prior personality/affect/situation reports, preceding binary activities, and known time features. Do **not** use any t+1 situation or same-prompt target item as a feature. Keep a per-person chronological sequence; check long gaps and missing prompts rather than treating every adjacent recorded row as equal elapsed time.

Reserve whole participants for a new-person test and future windows within development participants for temporal forecasting. If the model receives a test person's first several prompts as calibration, label the task *few-shot new-person prediction*, not zero-shot. Freeze the cohort, split IDs, target, and primary metric before comparing (1) history/prevalence and logistic baselines, (2) Euclidean slow-fast model, (3) same model with geometry. Match information, latent dimension, and training budget. Report participant-level intervals, calibration, and class prevalence, not accuracy alone. The original study's personalized models are useful reference baselines, but our held-out-person task is different from its idiographic evaluation.

## Main limitations

The cohort is university students, so transfer to a general population is uncertain. All outcomes here are self-reported; the data do not independently verify what a person did. There is no open dialogue text in the harmonized file, so an LLM conditioning or generation experiment still needs a separate, consented language dataset. Geometry can be claimed useful only if it adds held-out prediction beyond a matched non-geometric model; a better-looking embedding plot is insufficient.
