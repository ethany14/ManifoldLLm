# Learned Affective Space

This folder generates a PNG from the scratch-trained model's actual 3D affect
posterior means. It does not use an LLM, PCA, Isomap, or invented surface geometry.

## Run From the Repository Root

```sh
python -m pip install -r Alexander/small_language_model/visualization/requirements.txt
python -m Alexander.small_language_model.visualization.plot_manifold
```

The default checkpoint is `../artifacts/emobank/model.pt` relative to this folder.
Train it first using `python -m Alexander.small_language_model.train --epochs 20`,
or supply an existing checkpoint with `--checkpoint /path/to/model.pt`.
The script uses the CSV and tokenization settings saved with that checkpoint.
If the CSV has moved, pass `--csv /new/path/to/original.csv`; its hash must match.
The reconstructed training vocabulary must also match the saved vocabulary exactly.

Development data is plotted by default, keeping test data for final evaluation.
Select another official split with `--split train` or `--split test`.
Use `--output-dir` to keep images from different checkpoints separate.

## Outputs

- `output/manifold.png`: learned and human VAD point clouds in 3D and their 2D
  valence/arousal projections. Each point is one sentence. Axes use matching
  limits of at least [-1, 1], expanding if predictions fall outside that range.
- `output/coordinates.csv`: original sentence ID/text and all six learned/human
  coordinates, allowing every point to be inspected.
- `output/metadata.json`: checkpoint/CSV hashes, split, epoch, sample counts,
  attribution and interpretation notes.

Color represents human valence in both panels. It is not a predicted emotion
category. Model inference receives zero conditioning vectors; the affect encoder
depends only on text. Posterior means make inference deterministic.

The included `pilot/` image comes from the two-epoch pilot checkpoint trained on
800 sentences and shows 100 development sentences. Its metadata records the
checkpoint location, which is temporary on the development machine. Recreate a
pilot checkpoint with the training command in the parent README before plotting
on another machine. The exported image/CSV do not require that checkpoint to view.

A compact cloud can indicate weak predictions or collapse, not successful
disentanglement. These plots show samples of a learned representation, not proof
of a smooth or scientifically validated manifold. No interpolated surface is drawn.
Data credit: EmoBank, Buechel and Hahn (2017), CC BY-SA 4.0.