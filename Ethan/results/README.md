# Team result submissions

Create one subfolder per contributor under `Ethan/results/`. Keep scripts and aggregate comparisons reproducible, but do **not** commit licensed PsychArchives rows, participant IDs, person coordinates, model checkpoints, or individual predictions.

For the current PsychArchives method comparison, include a short `NOTES.md` and aggregate output table or JSON stating the dataset version, person split, input cues, preprocessing, model, loss, seed(s), training budget, selection rule, and whether each score is development or untouched test. Report next-valence performance, fixed trait-probe performance, disjoint-window matching, and a matched Euclidean geometry control. Preserve the 69-person test cohort for a frozen confirmatory analysis.

The existing `template/` and `zhiyu/` folders are **legacy EmoBank-format examples**, not the reporting schema for the person-level PsychArchives study. Their model and score fields should not be copied into a PsychArchives submission. For the active scientific interpretation and commands, start with [Ethan/README.md](../README.md).
