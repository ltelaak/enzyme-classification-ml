# Task 2 · EC main class prediction

The EC classifier assigns an enzyme to one of seven main classes using 128 PCA-reduced ProtT5 features. The final model has one hidden layer of 256 units. This record follows the parameters from the historical search through validation and held-out testing.

| Stage | Evidence | Recorded result |
|---|---|---|
| Parameter search | [Luis database export](optuna/luis_trials.csv), study `broad_search_multip_datasetv2`, trial 218 | Status `FAIL`, no Optuna objective value. The dataset cap (2,000), hidden width, dropout, learning rate and weight decay match the W&B run. The cause of the failed status is unknown. |
| Five-fold validation | [Fold results](wandb/folds.csv) and [selected W&B summary fields](wandb/summary.json) | Mean macro-F1 0.75018, MCC 0.72671, balanced accuracy 0.75439. These means can be calculated from the five rows. |
| Final training | [Research notebook](../../task_2/notebooks/nn_model_selection.ipynb) and [saved configuration](../../configs/task2.yaml) | AdamW, dropout 0.11765, weight decay 0.03425. The manual CV cell passes learning rate 0.0104079. The final cell sets that variable but omits it from the AdamW call. The saved training configuration records 0.001. |
| Held-out evaluation | [Original test metrics](../task2_test_metrics.txt) and [confusion matrix](final/confusion_matrix.png) | On 1,178 enzymes: macro-F1 0.74162, MCC 0.72385, balanced accuracy 0.74401. |

### Reading the search exports

Each CSV preserves one database snapshot, including unsuccessful trials. [db_trials.csv](optuna/db_trials.csv) has 4,707 rows from 41 studies. [luis_trials.csv](optuna/luis_trials.csv) has 2,873 rows from 32 studies. The snapshots overlap. Identify records by database, study and trial number. A different trial 218 in the first database belongs to `broad_search_one_layer_every_dataset_v1`.

The columns record Optuna state, stored numerical objectives by index, and sampled parameters with categorical choices decoded. Empty parameter fields indicate that no value is recorded for that trial. An empty objective is preserved as missing. Search spaces and objectives varied across studies. The final saved search cell in the notebook describes a later study. The manual CV execution preserved in the original notebook was interrupted in fold 2, so the subsequent displayed value of 0.7170 cannot be assigned to that execution.

### Training artifacts and provenance

[Loss curve](final/loss_curve.png) · [Macro-F1 curve](final/macro_f1_curve.png) · [Task 2 weights](../../checkpoints/task2.pth)

The two training curves, confusion matrix and state dictionary are byte-identical to the files in the original model directory for the usamp2000 dataset. The checkpoint loads into the archived network and produced finite outputs of shape (2, 7) for two synthetic 128-dimensional inputs. Reproducing the test scores requires the historical data and fitted PCA.

The CSVs were exported from the original SQLite files (Git blobs `53ff546811eb6e05443fd458e9fb92ded20153b8` and `03d16438f560e02aa589038c1dfede19a910d001`). The W&B fold table matched its recorded SHA-256. The three locally available W&B curve images differed from the hashes in that run summary and are excluded. The training figures linked above come from the final model directory. The curated notebook retains the original research cells with outputs cleared, portable data paths and a checkpoint evaluation cell adapted to the portfolio layout.
