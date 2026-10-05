# Alexander's Research

All affective-manifold research lives here. The reusable language backbone and
EmoBank loader live in [small_language_model](../small_language_model/README.md).
The dependency points one way: Alexander imports the shared package; the shared
package does not import Alexander or any manifold mathematics.

## Where to Make Math Changes

| File | Owns |
| --- | --- |
| [manifold_losses.py](manifold_losses.py) | Reconstruction, VAD anchoring helpers, covariance, KL, and weighted paper objective |
| [model.py](model.py) | Affect/private Gaussian heads, decoder, conditioning, and combined training loss |
| [geometry.py](geometry.py) | Neighborhood loss, decoder Jacobian, induced metric, and graph distances |
| [train.py](train.py) | Loss weights, optimization, evaluation, model selection, and checkpoints |
| [compare_manifold_variants.py](compare_manifold_variants.py) | Matched experiments and source snapshots |
| [visualization](visualization/README.md) | Coordinate plots and geometry diagnostics |

`EmotionLanguageModel` extends `small_language_model.core.GRUBackbone`. It keeps
the research heads here and uses `encode_tokens` from the shared core. Changes
to the loss equations or geometry do not require editing the shared package.

## Run From the Repository Root

Use Python 3.10 or newer (validated with 3.13):

```sh
python -m pip install -r Alexander/visualization/requirements.txt
python -m unittest discover -s Alexander -t . -v
python -m unittest discover -s small_language_model -t . -v
python -m Alexander.train --csv Ethan/data/emobank/emobank_pilot_1000.csv --epochs 2 --skip-test --output-dir Alexander/artifacts/my_pilot
```

Read the [full model and experiment guide](MANIFOLD_GUIDE.md) and
[article differences register](ARTICLE_DIFFERENCES.md) before changing methods.
Record new departures from the paper in the register.

## Research Outputs and Migration

Checkpoints are in `artifacts/` (ignored by Git). Saved comparisons are in
`manifold_results/`; plots and diagnostic reports are also under `visualization/`.
Existing archived file contents are unchanged, including historical paths. New
comparison snapshots preserve both packages and record their source hashes.

Use `from Alexander.model import EmotionLanguageModel` and
`from Alexander.manifold_losses import LossWeights`. Research commands formerly
under `small_language_model` now run under `Alexander`; no duplicate implementation
or forwarding modules remain in the shared package. Version-2 state-dict checkpoints
keep the same parameter keys and can be loaded by the relocated affective model.