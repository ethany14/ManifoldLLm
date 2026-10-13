# ManifoldLLm

Research code for asking **how a manifold can represent an updateable digital-personality state** from repeated observations of the same people.

Ethan's selected representation method is a learned nonlinear encoder-decoder with decoder-induced Riemannian geometry, evaluated on 300-item IPIP personality responses. A 16-dimensional coordinate is the provisional geometry-focused setting. The earlier longitudinal slow/fast model remains a separate temporal prototype. Neither is a validated digital personality or a demonstrated general advantage over Euclidean geometry.

## Latest English Method Report

[Read the report: Learned Manifolds for Digital Personality - Method and Rationale (PDF)](Ethan/output/pdf/LEARNED_MANIFOLD_METHOD_AND_RATIONALE_EN.pdf)

This 14-page illustrated report explains why the encoder-decoder manifold was selected, its relationship to Isomap, variational autoencoders and decoder-induced Riemannian geometry, the mathematical specification, and the IPIP dimension and geometry-objective comparisons. It includes model-flow diagrams, result charts, actual trained-model 3D decoder slices and an aggregated personality-coordinate distribution. It concludes with the implementation pathway from personality evidence to learned coordinates, explicit geometry and a future personality-conditioned behavioral model. Core equations use LaTeX-style typesetting; projection variance and visualization limits are stated explicitly.

The report distinguishes completed representation experiments from planned behavioral generation. The 16D choice is an engineering tradeoff, current paths are not optimized geodesics, and no personality-conditioned LLM has yet been validated.

## Person-Level Research

- [Current method, commands, and data handling](Ethan/README.md)
- [Earlier longitudinal PsychArchives report (PDF)](Ethan/output/pdf/PSYCHARCHIVES_CORE_METHOD_REPORT_EN.pdf)
- [PsychArchives implementation details](Ethan/PSYCHARCHIVES_MANIFOLD_IMPLEMENTATION.md)
- [Geometry-training comparison](Ethan/PSYCHARCHIVES_GEOMETRY_TRAINING_RESULTS.md)
- [Feature-group and simple-baseline check](Ethan/PSYCHARCHIVES_FEATURE_GROUP_RESULTS.md)

The active person-level study is in `Ethan/`. Earlier EmoBank, DAIR, Beck, and PersDyn scripts there are retained for reproducibility, but are not evidence for the current method conclusion. Licensed PsychArchives source files and individual-level outputs must not be committed or redistributed.

## Shared Language Core and Manifold Research

The reusable, scratch-trained GRU backbone, plain language model, and data loader
live in [small_language_model](small_language_model/README.md) for all students.
[Alexander](Alexander/README.md) owns the affective model, manifold mathematics,
training experiments, and visualizations, importing that shared core. Neither
package loads pretrained language models or calls an LLM service.

## Alexander's Manifold Quick Start

Use Python 3.10 or newer (validated with 3.13). Run from the repository root;
the separate environment below leaves any existing student environment untouched:

```sh
python3.13 -m venv .venv-slm
source .venv-slm/bin/activate
python -m pip install -r Alexander/visualization/requirements.txt
python -m unittest discover -s small_language_model -t . -v
python -m unittest discover -s Alexander -t . -v
python -m Alexander.train --csv Ethan/data/emobank/emobank_pilot_1000.csv --epochs 2 --skip-test --output-dir Alexander/artifacts/my_pilot
```

The pilot is a pipeline check, not evidence of a recovered affective manifold.
Use a separate output directory for each experiment. The shared loader reads
EmoBank CSV data from Ethan's folder by default; it does not import Ethan's code
or use his embeddings. Supply `--csv` to use another compatible dataset.

See the [module guide and research limitations](Alexander/MANIFOLD_GUIDE.md)
and the [article differences register](Alexander/ARTICLE_DIFFERENCES.md)
before changing the research methods.
