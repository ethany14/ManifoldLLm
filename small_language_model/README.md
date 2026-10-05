# Shared Small Language Model

Reusable, scratch-trained language components for all students. No pretrained
weights, external LLM calls, or imports from personal research folders are used.

## Shared API

- [core.py](core.py): `GRUBackbone` provides embeddings and causal token states;
  `CausalLanguageModel` adds a plain next-token output head.
- [data.py](data.py): tokenizer, train-only vocabulary, and validated EmoBank CSV
  loading. The default CSV is in Ethan's data folder; no Ethan code is imported.

Use Python 3.10 or newer and run from the repository root:

```sh
python -m pip install -r small_language_model/requirements.txt
python -m unittest discover -s small_language_model -t . -v
```

```python
import torch
from small_language_model.core import CausalLanguageModel

model = CausalLanguageModel(vocabulary_size=100)
logits = model(torch.tensor([[2, 4, 5]]))
assert logits.shape == (1, 3, 100)
```

The plain model is randomly initialized, not a trained conversational model.
Train its logits against next-token targets with cross-entropy, ignoring padding
ID 0. Students can subclass `GRUBackbone` and add their own heads without coupling
the shared code to another student's research.

## Alexander's Manifold Research

The affective model, loss equations, geometry, training CLI, tests, plots, and
research outputs live in [Alexander](../Alexander/README.md). They import this
package's backbone and data loader. Use `python -m Alexander.train` for that
experiment, and `from Alexander.model import EmotionLanguageModel` to load its
checkpoints. The old `small_language_model.model`, `.train`, `.geometry`, and
`.manifold_losses` research modules have moved; they are not shared-core APIs.