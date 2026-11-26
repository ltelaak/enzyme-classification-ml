# Historical clustering-threshold comparisons

These 11 TSV files are unmodified text exports from `task_1/results/performance_eval/threshold_comparison/` in the original PBL repository. They compare alternative sequence identity thresholds, length filters, undersampling and classifiers. They are exploratory cross-validation results, not final held-out test results. Conditions differ between files; avoid pooling scores or attributing differences to clustering alone. Filenames are historical labels, not independently verified execution metadata.

## Example: AAindex + Random Forest (unfiltered threshold comparison)

| Sequence identity threshold | Dataset size | Mean CV accuracy |
|---:|---:|---:|
| 30% | 40,426 | 0.7838 |
| 50% | 56,907 | 0.7890 |
| 70% | 71,800 | 0.7970 |
| 90% | 87,446 | 0.8098 |
| 100% | 101,688 | 0.8343 |

Source: [`cthresh_comp_cv_aaindex_rf.tsv`](cthresh_comp_cv_aaindex_rf.tsv), rounded for display. This is an earlier exploratory dataset, not the final Task 1 dataset. Higher identity thresholds retain more sequences and may allow more similarity between training and validation sequences; these scores do not establish which threshold generalizes best. The final project used 30% class-wise clustering to reduce homology-related redundancy, not because it maximized this particular accuracy measure.

See [data preparation notes](../../../docs/data_preparation.md).
