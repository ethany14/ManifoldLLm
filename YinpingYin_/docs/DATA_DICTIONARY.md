# Canonical bridge schema

The normalized rows preserve `dataset`, `person_id`, `text`, `emotion`, `emotion_kind`, and `personality` using the existing normalizer. Not all corpora provide each field. An empty field means unknown, not zero. A person ID is meaningful only *within the same source*. Do not merge people across corpora by row number or by an arbitrary matching name.

`emotion_kind`: `observed`, `predicted`, or `missing`; model-predicted labels are never observed ground truth. A dataset's native label ontology should be kept separately; Big Five traits from self-report are not interchangeable with inferred character descriptions. Preserve conversation/turn order and timestamps whenever available.
