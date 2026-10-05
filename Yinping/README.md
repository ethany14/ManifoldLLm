# Yinping - October 3 Dataset Collection

This folder contains my dataset collection and preprocessing work for the ManifoldLLm research project.

The work focuses only on the first three October 3 tasks:

1. Select a main dataset
2. Document the dataset
3. Clean and structure the dataset

Model training, PCA, Isomap, VAE, embeddings, and manifold visualization are not included in this folder.

---

## 1. Main Dataset

### Public Discourse Corpus (PDC)

The Public Discourse Corpus was selected as the primary dataset.

It was selected because it contains:

- repeated observations from publicly documented individuals
- sentence-level speech
- speaker identity
- video identity
- temporal metadata
- pre-existing affective valence labels

The original dataset contains approximately:

- 186,642 sentences
- 998 videos
- 100 public speakers

The original source files used are:

```text
main.jsonl
speakers.jsonl
videos.jsonl