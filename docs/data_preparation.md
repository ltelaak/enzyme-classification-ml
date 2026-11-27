# Data preparation and protein representations

This is a retrospective of the original collaborative project, not a newly executed experiment. Source notebooks remain under their original task directories. Historical outputs are curated separately.

## Final datasets

| | Task 1 | Task 2 |
|---|---:|---:|
| Raw sequences (reported) | 104,729 | 41,406 |
| After preprocessing (reported) | 39,335 | 7,850 |
| Development set | 33,434 | 6,672 |
| Held-out test set | 5,901 | 1,178 |
| MMseqs2 identity threshold | 30% | 30% |
| Length percentile bounds | 2–98 | 2–98 |
| Length cutoffs in the report (amino acids) | 17–1,890 | 49–1,910 |
| Final undersampling | None (final dataset name) | Cap 2,000 per EC class |
| ProtT5 input dimensions | 1,024 | 1,024 |
| PCA dimensions | 128 | 128 |

The local raw tables and split files confirm the sequence counts in the report. The 6,672 Task 2 development sequences belong to the internal split. The final talk shows 6,581 after filtering against [Price-149](../results/price149/). The original task 1 preprocessing notebook also contains an undersampled exploratory branch; it must not be interpreted as the final non-undersampled configuration. Task 2 preprocessing clusters EC classes separately, caps majority classes, splits the dataset 85/15 and fits PCA on the development data before transforming the test data.

## Exploratory workflow

- Task 1: cluster thresholds 30, 50, 70, 90, 100%; alternative length filters and undersampling; classical sequence features and protein embeddings.
- Task 2 reduces each annotation to its distinct first-level EC numbers and expands them to one row per class. It then drops every copy of a repeated sequence, including sequences assigned to multiple main classes. Clustering within each class and length filtering precede undersampling. The cap of 2,000 leaves EC classes 1–3 with 2,000 proteins each. The archived sampling call has no fixed random seed.
- The embedding generator retains historical support for multiple protein language model families; the final project used ProtT5. Presence of an option is not evidence that it was benchmarked.
- PCA is fitted separately for the two tasks. Fitting PCA before internal cross-validation may expose fold-validation feature statistics; class-wise clustering does not establish a global pairwise identity guarantee. The historical evaluation is preserved rather than retroactively changed.

## Notebook index

| Notebook | Role |
|---|---|
| [Task 1 preprocessing](../task_1/notebooks/preprocessing.ipynb) | Raw sequence processing, exploratory branches, splitting and PCA |
| [Task 1 clustering](../task_1/notebooks/clustering.ipynb) | Early clustering threshold datasets |
| [Task 1 threshold comparison](../task_1/notebooks/cluster_thresh_comp.ipynb) | OHE/AAindex and classifier comparisons on alternative datasets |
| [Embedding generator](../task_1/notebooks/prott5_embedding_generator.ipynb) | Historical protein embedding generation |
| [Task 2 preprocessing](../task_2/notebooks/preprocessing.ipynb) | EC classes, class-wise clustering, cap 2,000, splitting and PCA |

Run archived notebooks from their own directories after supplying task-specific data and installing research dependencies. Some exploratory cells refer to earlier APIs, alternative execution environments or missing datasets. They have not been rerun. The portfolio includes both recovered PCA transformations. The original protein datasets are preserved privately. [Clustering threshold results](../results/data_analysis/task1_clustering/)
