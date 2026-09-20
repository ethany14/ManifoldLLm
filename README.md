# ManifoldLLm

Research code for asking **how a manifold can represent an updateable digital-personality state** from repeated observations of the same people.

The current method is a longitudinal slow/fast generative model: an ordered history estimates a slow person coordinate `phi`, current cues estimate a fast state `z`, and a nonlinear decoder defines a local metric on the slow coordinate. Decoder-path distance has been tested inside a same-person contrastive objective. This is a working method prototype, **not** a validated digital personality or a demonstrated advantage over Euclidean geometry.

## Start here

- [Current method, commands, and data handling](Ethan/README.md)
- [How-focused supervisor report (PDF)](Ethan/output/pdf/PSYCHARCHIVES_CORE_METHOD_REPORT_EN.pdf)
- [PsychArchives implementation details](Ethan/PSYCHARCHIVES_MANIFOLD_IMPLEMENTATION.md)
- [Geometry-training comparison](Ethan/PSYCHARCHIVES_GEOMETRY_TRAINING_RESULTS.md)
- [Feature-group and simple-baseline check](Ethan/PSYCHARCHIVES_FEATURE_GROUP_RESULTS.md)

The active person-level study is in `Ethan/`. Earlier EmoBank, DAIR, Beck, and PersDyn scripts are retained for reproducibility, but are not evidence for the current method conclusion. Licensed PsychArchives source files and individual-level outputs must not be committed or redistributed.
