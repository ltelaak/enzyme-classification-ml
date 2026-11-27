# Task 2 · EC main class prediction

The EC classifier assigns an enzyme to one of seven main classes using 128 PCA-reduced ProtT5 features. The final model has one hidden layer of 256 units. This record follows the parameters from the historical search through validation and held-out testing.

| Stage | Evidence | Recorded result |
|---|---|---|
| Parameter search | [Luis database export](optuna/luis_trials.csv), study `broad_search_multip_datasetv2`, trial 218 | Status `FAIL`, no Optuna objective value. The dataset cap (2,000), hidden width, dropout, learning rate and weight decay match the W&B run. The cause of the failed status is unknown. |
| Five-fold validation | [Fold results](wandb/folds.csv) and [selected W&B summary fields](wandb/summary.json) | Mean macro-F1 0.75018, MCC 0.72671, balanced accuracy 0.75439. These means can be calculated from the five rows. |
| Final training | [Research notebook](../../task_2/notebooks/nn_model_selection.ipynb) and [saved configuration](../../configs/task2.yaml) | AdamW, dropout 0.11765, weight decay 0.03425. The manual CV cell passes learning rate 0.0104079. The final cell sets that variable but omits it from the AdamW call. The saved training configuration records 0.001. |
| Held-out evaluation | [Original test metrics](../task2_test_metrics.txt) and [confusion matrix](final/confusion_matrix.png) | On 1,178 enzymes: macro-F1 0.74162, MCC 0.72385, balanced accuracy 0.74401. |

### Held-out test verification

The original inference export (`8351126b089050e908d8b788a645133b1c22e1c0`) records the protein ID, true EC class and predicted EC class for each of the 1,178 enzymes. Recounting those predictions reproduces all 49 cells of the saved confusion matrix and a macro-F1 of 0.74162. The report’s [bootstrap comparison with four baselines](../baselines/#task-2-ec-classification) rounds the network result to 0.74. The report’s class-wise recall values also differ in places from the saved matrix. For example, EC 1 has 261 correct predictions out of 300 (87%) in the saved run.

The matching historical dataset and original fitted PCA
were recovered from the original shared project data.
All 1,178 test protein IDs match the archived inference export.

Inference using the recovered raw embeddings and original PCA
reproduces all archived predictions, with macro-F1 of
0.7416219455171614 and MCC of 0.723851024160418.

The original PCA was exported to portable NPZ format.
The reconstructed test embeddings match the stored projected
embeddings with an RMSE of 2.0757e-09.

### Reading the search exports

Each CSV preserves one database snapshot, including unsuccessful trials. [db_trials.csv](optuna/db_trials.csv) has 4,707 rows from 41 studies. [luis_trials.csv](optuna/luis_trials.csv) has 2,873 rows from 32 studies. The snapshots overlap. Identify records by database, study and trial number. A different trial 218 in the first database belongs to `broad_search_one_layer_every_dataset_v1`.

The columns record Optuna state, stored numerical objectives by index, and sampled parameters with categorical choices decoded. Empty parameter fields indicate that no value is recorded for that trial. An empty objective is preserved as missing. Search spaces and objectives varied across studies. The final saved search cell in the notebook describes a later study. The manual CV execution preserved in the original notebook was interrupted in fold 2, so the subsequent displayed value of 0.7170 cannot be assigned to that execution.

### Training artifacts and provenance

[Loss curve](final/loss_curve.png) · [Macro-F1 curve](final/macro_f1_curve.png) · [Task 2 weights](../../checkpoints/task2.pth)

The two training curves, confusion matrix and state dictionary are byte-identical to the files in the original model directory for the usamp2000 dataset. The checkpoint loads into the archived network and produces seven scores per enzyme. Inference from raw ProtT5 embeddings requires the PCA fitted alongside those weights.

The CSVs were exported from the original SQLite files (Git blobs `53ff546811eb6e05443fd458e9fb92ded20153b8` and `03d16438f560e02aa589038c1dfede19a910d001`). The portfolio includes the selected W&B run. Its exported summary fields and all five fold rows match the original run, and the original fold table matches the SHA-256 in its tracking record. The three locally available W&B curve images differed from the hashes in that run summary and are excluded. The training figures linked above come from the final model directory. The curated notebook retains the original research cells with outputs cleared, portable data paths and a checkpoint evaluation cell adapted to the portfolio layout.
