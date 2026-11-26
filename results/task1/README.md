# Task 1: final binary classifier

The checkpoint reproduced the archived test accuracy, MCC and confusion matrix on all 5,901 proteins when evaluated with the original 128-dimensional test embeddings.

- [Model selection and original CV archive](model_selection/)
- [Final model configuration](../../configs/task1.yaml)
- [Original checkpoint](../../checkpoints/task1.pth)
- [Original held-out test metrics](../task1_test_metrics.txt)

| Metric | Recorded value |
|---|---:|
| Test accuracy | 0.9198440942213184 |
| Test MCC | 0.8203024040584289 |
| Reported AUC | 0.9114342058752395 |

The AUC in the original inference code is calculated from rounded class predictions, not continuous scores. The training code uses early stopping on a 90/10 development split. It records `best_model_state = model.state_dict()` without a deep copy and saves weights before the final restoration call. Therefore the exact relationship between the saved checkpoint and the best validation epoch cannot be established from the code alone; the checkpoint nevertheless reproduces the recorded test predictions at the level of the confusion matrix.

![Original final training accuracy](training/accuracy.png)

![Original final training loss](training/loss.png)

![Original held-out confusion matrix](test/confusion_matrix.png)

The original checkpoint requires 128-dimensional input. Raw 1024-dimensional ProtT5 embeddings require the original fitted PCA, not a newly fitted arbitrary transformer. Fitted PCA files and the original datasets are not included. Do not equate the current synthetic smoke test with reproduction of the historical metrics.
