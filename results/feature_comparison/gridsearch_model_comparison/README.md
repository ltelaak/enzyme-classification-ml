# Historical feature and classifier search results

All 19 TSV exports from the original `task_1/results/performance_eval/gridsearch_model_comparison/` directory are retained byte-for-byte, with their original subdirectories and filenames. The [summary CSV](summary.csv) is a new transcription of the maximum recorded CV MCC and nonempty final-test MCC fields per file, not a new experiment. Duplicate identical test-MCC rows in the hybrid PseAAC SVM export are represented once. 

The original `calculate_search_grid` helper uses `RandomizedSearchCV` or `GridSearchCV`, selects/refits by MCC, and records accuracy and MCC. It also compares the default estimator for randomized searches. The `final_test_acc` column is attached by accuracy rank even though the estimator is selected by MCC, so it may appear on a different row. Interpret the test fields as file-level results, not necessarily as independent measurements of every configuration.

## Recorded results (rounded MCC)

| Feature | Classifier | Best CV MCC | Recorded test MCC |
|---|---|---:|---:|
| AAC | [Logistic Regression](aac/aac_logr.tsv) | 0.3979 | 0.3922 |
| AAC | [Random Forest](aac/aac_rf.tsv) | 0.6568 | 0.6630 |
| AAC | [Linear SVC](aac/aac_svm.tsv) | 0.4004 | 0.3943 |
| AAindex | [Logistic Regression](aaindex/aaindex_logr.tsv) | 0.3984 | 0.3919 |
| AAindex | [Random Forest](aaindex/aaindex_rf.tsv) | 0.6687 | 0.6731 |
| AAindex | [Linear SVC](aaindex/aaindex_svm.tsv) | 0.3984 | 0.3917 |
| AAindex + n-grams | [Logistic Regression](hybrid_aaindex_ngrams/hybrid_aaindex_ngrams_logr.tsv) | 0.5254 | 0.5313 |
| AAindex + n-grams | [Random Forest](hybrid_aaindex_ngrams/hybrid_aaindex_ngrams_rf.tsv) | 0.6806 | 0.6785 |
| AAindex + n-grams | [Linear SVC](hybrid_aaindex_ngrams/hybrid_aaindex_ngrams_svm.tsv) | 0.5246 | 0.5320 |
| AAindex + PseAAC | [Logistic Regression](hybrid_aaindex_pseaac/hybrid_aaindex_pseaac_logr.tsv) | 0.5453 | 0.5458 |
| AAindex + PseAAC | [Random Forest](hybrid_aaindex_pseaac/hybrid_aaindex_pseaac_rf.tsv) | 0.6806 | 0.6892 |
| AAindex + PseAAC | [Linear SVC](hybrid_aaindex_pseaac/hybrid_aaindex_pseaac_svm.tsv) | 0.5464 | 0.5427 |
| One-hot | [Logistic Regression](one_hot_encoding/one_hot_encoding_logr.tsv) | 0.4348 | 0.4397 |
| One-hot | [Random Forest](one_hot_encoding/one_hot_encoding_rf.tsv) | 0.3875 | 0.4218 |
| One-hot | [Linear SVC](one_hot_encoding/one_hot_encoding_svm.tsv) | 0.3782 | 0.4139 |
| ProtT5 | [Random Forest](prott5/prott5_rf.tsv) | 0.8863 | 0.8930 |
| PseAAC | [Logistic Regression](pseaac/pseaac_logr.tsv) | 0.4793 | 0.4734 |
| PseAAC | [Random Forest](pseaac/pseaac_rf.tsv) | 0.6622 | 0.6777 |
| PseAAC | [Linear SVC](pseaac/pseaac_svm.tsv) | 0.4876 | 0.4810 |

## Interpretation and limitations

The ProtT5 RF export records CV MCC 0.8863 and test MCC 0.8930, higher than the later final FNN test MCC 0.8203. The TSV does not store the exact dataset/split identity or original execution state; the current notebook has been edited across experiments. These older search scores cannot be assumed to share the final protocol. We retain them rather than claim the FNN beat every historical configuration. The later same-notebook ProtT5 baseline comparison is documented [separately](../../baselines/README.md). Classical feature encodings, hybrid features and ProtT5 were investigated, and the final design prioritized a compact learned classifier with the ProtT5/PCA input. No missing ProtT5 logistic/SVC TSVs are reconstructed; those classifiers are present in the notebook but the original folder has only an RF export.
