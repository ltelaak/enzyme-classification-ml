# EC classification notebooks

[Preprocessing](preprocessing.ipynb) builds the EC dataset. [Random forest](task2_random_forest.ipynb) covers the baseline, and [model selection](nn_model_selection.ipynb) follows the neural network from the search to final training. [Trial exports, fold scores and final results](../../results/task2/)

These notebooks use the same [archive conventions as Task 1](../../task_1/notebooks/README.md). In the model selection notebook, the last saved search cell uses a different study from the selected W&B configuration. The evaluation cell was adapted to load the portfolio checkpoint.

## Original notebook versions

| Notebook | Git blob in the original project |
|---|---|
| [EC classifier selection and training](nn_model_selection.ipynb) | `30147bb905601497593f33a169eb528d3c985256` |
| [Preparing the EC classification dataset](preprocessing.ipynb) | `a3a1665b3f1cc6fa37504fc2782c3558fcc30501` |
| [Random forest baseline for EC classification](task2_random_forest.ipynb) | `b0c43e84918cf8729fe88182c507c119b1f5206b` |
