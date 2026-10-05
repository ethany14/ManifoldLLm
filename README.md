# ManifoldLLm

Research code for asking **how a manifold can represent an updateable digital-personality state** from repeated observations of the same people.

The current person-level method is a longitudinal slow/fast generative model: an ordered history estimates a slow person coordinate `phi`, current cues estimate a fast state `z`, and a nonlinear decoder defines a local metric on the slow coordinate. Decoder-path distance has been tested inside a same-person contrastive objective. This is a working method prototype, **not** a validated digital personality or a demonstrated advantage over Euclidean geometry.

## Person-Level Research

- [Current method, commands, and data handling](Ethan/README.md)
- [Final report (PDF)](Ethan/output/pdf/PSYCHARCHIVES_CORE_METHOD_REPORT_EN.pdf)
- [PsychArchives implementation details](Ethan/PSYCHARCHIVES_MANIFOLD_IMPLEMENTATION.md)
- [Geometry-training comparison](Ethan/PSYCHARCHIVES_GEOMETRY_TRAINING_RESULTS.md)
- [Feature-group and simple-baseline check](Ethan/PSYCHARCHIVES_FEATURE_GROUP_RESULTS.md)

The active person-level study is in `Ethan/`. Earlier EmoBank, DAIR, Beck, and PersDyn scripts there are retained for reproducibility, but are not evidence for the current method conclusion. Licensed PsychArchives source files and individual-level outputs must not be committed or redistributed.

## Shared Emotion Language Model

Student research folders remain independent. The shared, scratch-trained emotion
language model lives in [small_language_model](small_language_model/README.md)
and is available for everyone to use and extend. It does not load pretrained
language models or call an LLM service.

## Shared Model Quick Start

Use Python 3.10 or newer (validated with 3.13). Run from the repository root;
the separate environment below leaves any existing student environment untouched:

```sh
python3.13 -m venv .venv-slm
source .venv-slm/bin/activate
python -m pip install -r small_language_model/visualization/requirements.txt
python -m unittest discover -s small_language_model -t . -v
python -m small_language_model.train --csv Ethan/data/emobank/emobank_pilot_1000.csv --epochs 2 --skip-test --output-dir small_language_model/artifacts/my_pilot
```

The pilot is a pipeline check, not evidence of a recovered affective manifold.
Use a separate output directory for each experiment. The shared model reads
EmoBank CSV data from Ethan's folder by default; it does not import Ethan's code
or use his embeddings. Supply `--csv` to use another compatible dataset.

See the [module guide and research limitations](small_language_model/README.md)
and the [article differences register](small_language_model/ARTICLE_DIFFERENCES.md)
before changing the research methods.
