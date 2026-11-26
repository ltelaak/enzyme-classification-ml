# Enzyme classification with ProtT5 embeddings

**Rostlab · Joint B.Sc. Bioinformatics, LMU & TUM**

Given a protein sequence, can we detect whether it is an enzyme and predict its main Enzyme Commission (EC) class? We compared sequence encodings and classical classifiers, then trained two feed-forward networks on ProtT5 embeddings. Each network uses its own PCA to reduce 1,024 dimensions to 128. The combined model returns 0 for a predicted non-enzyme or a single EC class from 1 to 7.

## Experiments and results

| Experiment | Code and model selection | Historical evaluation |
|---|---|---|
| Task 1 · Enzyme detection | [Feature comparisons from one-hot encoding to ProtT5 + NN](task_1/notebooks/README.md) · [NN selection and run 131](results/task1/model_selection/) | **0.8203 MCC** on 5,901 proteins. The closest classical models in the final comparison reached about 0.74. [Test output](results/task1_test_metrics.txt) |
| Task 2 · EC main class | [Optuna and W&B notebook](task_2/notebooks/nn_model_selection.ipynb) · [Trial exports and validation folds](results/task2/) | **0.7416 macro-F1** from the archived predictions for 1,178 enzymes. [Test output](results/task2_test_metrics.txt) · [Baseline comparison](results/baselines/#task-2-ec-classification) |
| Task 3 · Combined prediction | [Hierarchical model](task_3/src/models.py) | **0.6953 macro-F1** across eight classes. [Internal evaluation](results/task3/) |
| Price-149 · External evaluation of Task 2 | Training data filtered for similarity to Price-149 | **0.72 macro-F1 and 0.72 weighted F1** on 149 enzymes, as reported in the project report. [Benchmark figure and scores](results/price149/) |

### Data and model choices

Task 1 uses binary enzyme labels. Task 2 starts with experimentally annotated enzymes, clusters sequences within each EC main class and caps the largest classes at 2,000 examples before splitting. The [data preparation notes](docs/data_preparation.md) give the sequence counts and filtering steps for both tasks.

For Task 1, ProtT5 performed best across three classical classifiers in the presentation's encoding comparison. On the final ProtT5 test set, the **128–128–64–1 neural network** scored highest. An older [random forest grid-search export](results/feature_comparison/gridsearch_model_comparison/prott5/prott5_rf.tsv) records a higher MCC under an unverified split. The [comparison notes](results/baselines/) explain which scores can be compared.

For Task 2, the selected **128–256–7 network** matches the parameters of Optuna trial 218 in the Luis database. Optuna records that trial as failed. The five stored W&B folds give a mean macro-F1 of 0.7502. Cross-validation used a learning rate of 0.0104079 and the final AdamW configuration records 0.001. The implementation retains the final sigmoid, which differs from the linear output described in the report. [Task 2 experiment record](results/task2/README.md)

## Run the models

From the repository root, set up Python 3.11 or 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.smoke_test
```

The smoke test checks the three model interfaces using synthetic embeddings, newly fitted PCA transforms and randomly initialized networks. To run the **two original trained classifiers**, use the [checkpoint example](checkpoints/README.md) in the same environment.

With the original test embeddings supplied locally, the Task 1 checkpoint reproduces the reported MCC and confusion matrix. For Task 2, the archived predictions confirm the score. The available embedding file contains a different set of proteins. [Task 2 verification](results/task2/#held-out-test-verification)

This repository includes both checkpoints and their configurations. The test data and the two fitted PCA transforms are outside the portfolio. The [research dependencies](requirements-research.txt) cover the broader notebook workflow.

## Authors and contributions

This project was developed collaboratively by Neo Mehmedinovic, Luis Telaak, David H. Nar and Daniel Poock. The models, evaluation and results are a team achievement.

Luis's project work included dataset preparation, neural network architecture development and experiment tracking with Weights & Biases.

This repository documents the shared academic work; the portfolio-specific documentation and portability changes are separate from the original experiments.
