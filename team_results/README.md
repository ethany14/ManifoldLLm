# Team Result Comparison

Create one subfolder per contributor:

```text
team_results/
  template/
  zhiyu/
  teammate_name/
```

Copy the template folder, rename it to your GitHub username, and record every run in
that folder. Keep raw datasets, virtual environments, model weights, and embedding
arrays outside `team_results`.

For a directly comparable run, use the settings in `REPRODUCIBILITY.md`. Each result
submission should include:

- `experiment_config.json`: model, layer, pooling, split, seed, and loss weights;
- `results.csv`: held-out test metrics;
- `NOTES.md`: hardware, deviations, failures, and interpretation.

Do not tune a model against the official test set. Use the development split for model
selection and report the test result once the setup is fixed.
