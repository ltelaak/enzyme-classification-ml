# Enzyme classification with ProtT5 embeddings

**Rostlab · Joint B.Sc. Bioinformatics, LMU & TUM**

Given a protein sequence, can we detect whether it is an enzyme and predict its main Enzyme Commission (EC) class? We compared sequence encodings and classical classifiers, then trained two feed-forward networks on ProtT5 embeddings. Each network uses its own PCA to reduce 1,024 dimensions to 128. The combined model returns 0 for a predicted non-enzyme or a single EC class from 1 to 7.

## Experiments and results

| Experiment | Code and model selection | Historical evaluation |
|---|---|---|
| Task 1 · Enzyme detection | [Feature comparisons from one-hot encoding to ProtT5 + NN](task_1/notebooks/README.md) · [NN selection and run 131](results/task1/model_selection/) | **0.8203 MCC** on 5,901 proteins. The closest classical models in the final comparison reached about 0.74. [Test output](results/task1_test_metrics.txt) |
| Task 2 · EC main class | [Optuna and W&B notebook](task_2/notebooks/nn_model_selection.ipynb) · [Trial exports and validation folds](results/task2/) | **0.7416 macro-F1** reproduced on the original test set of 1,178 enzymes. [Test output](results/task2_test_metrics.txt) · [Baseline comparison](results/baselines/#task-2-ec-classification) |
| Task 3 · Combined prediction | [Hierarchical model](task_3/src/models.py) | **0.6953 macro-F1** reproduced across eight classes on the historical notebook test set (5,111 proteins). [Internal evaluation](results/task3/) |
| Price-149 · External evaluation of Task 2 | Training data filtered for similarity to Price-149 | **0.72 macro-F1 and 0.72 weighted F1** on 149 enzymes, as reported in the project report. [Benchmark figure and scores](results/price149/) |

### Data and model choices

Task 1 uses binary enzyme labels. Task 2 starts with experimentally annotated enzymes, clusters sequences within each EC main class and caps the largest classes at 2,000 examples before splitting. The [data preparation notes](docs/data_preparation.md) give the sequence counts and filtering steps for both tasks.

For Task 1, ProtT5 performed best across three classical classifiers in the presentation's encoding comparison. On the final ProtT5 test set, the **128–128–64–1 neural network** scored highest. An older [random forest grid-search export](results/feature_comparison/gridsearch_model_comparison/prott5/prott5_rf.tsv) records a higher MCC under an unverified split. The [comparison notes](results/baselines/) explain which scores can be compared.

For Task 2, the selected **128–256–7 network** matches the parameters of Optuna trial 218 in the Luis database. Optuna records that trial as failed. The five stored W&B folds give a mean macro-F1 of 0.7502. Cross-validation used a learning rate of 0.0104079 and the final AdamW configuration records 0.001. The implementation retains the final sigmoid, which differs from the linear output described in the report. [Task 2 experiment record](results/task2/README.md)

## Installation and inference

From the repository root, set up a Python environment
(tested with Python 3.12):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.smoke_test
```

The smoke test checks the three model interfaces using the recovered
PCA transformations, synthetic inputs and randomly initialized networks.

The repository includes both original trained checkpoints, their
configurations and the corresponding PCA transformations.
See [checkpoint usage](checkpoints/README.md).

For the original research notebooks, training and FASTA-based inference,
install the extended dependencies:

```bash
python -m pip install -r requirements-research.txt
```

The model and research environments were tested through clean installations.

## Reproducibility

The original trained classifiers were evaluated using the recovered
historical test embeddings.

| Metric | Task 1 | Task 2 | Task 3 |
|---|---:|---:|---:|
| Test samples | 5,901 | 1,178 | 5,111 |
| Macro-F1 | 0.9101 | 0.7416 | 0.6953 |
| MCC | 0.8203 | 0.7239 | 0.7469 |
| Balanced accuracy | 0.9114 | 0.7440 | 0.7325 |

The Task 1 PCA was reconstructed from matching raw and projected
training embeddings. The original Task 2 PCA was recovered and
exported to NPZ format.

Both transformations reproduce the stored test embeddings with
an RMSE below 3e-9 and identical classifier predictions.

The [validation script](scripts/validate_checkpoints.py)
checks all three classification tasks using the historical test data:

```bash
python -m scripts.validate_checkpoints --help
```

The original datasets are preserved privately and are not distributed
with this repository.

Task 3 reproduces the historical notebook evaluation rather than
the separately described filtered report dataset.
Price-149 remains a historical external benchmark reported in
the original project and is outside the verified reproduction scope.

Full retraining and fresh ProtT5 embedding generation have not
been rerun as part of the portfolio preparation.

## Authors and contributions

This project was developed collaboratively by Neo Mehmedinovic, Luis Telaak, David H. Nar and Daniel Poock. The models, evaluation and results are a team achievement.

Luis's project work included dataset preparation, neural network architecture development and experiment tracking with Weights & Biases.

This repository documents the shared academic work; the portfolio-specific documentation and portability changes are separate from the original experiments.
