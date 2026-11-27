# Hierarchical classification

The [Task 3 model](../../task_3/src/models.py) applies the two fitted PCA transforms to each ProtT5 embedding. When the binary model's rounded output is zero, the pipeline returns class 0. For a predicted enzyme it returns the second model's highest-scoring EC class (1–7). Both classifiers reuse the trained weights from Tasks 1 and 2.

The original evaluation combined non-enzymes from the Task 1 test set with enzymes from the Task 2 test set. Section 2.4.1 of the team's final report gives 4,191 sequences and describes filtering enzymes against the Task 1 training set at 30% identity. Slide 28 also reports 4,191 sequences and rounds macro-F1 to 0.70. Its notes give macro-recall of 0.73 and MCC of 0.75. The original notebook takes all label-0 rows from its Task 1 input and all rows from its Task 2 input. It records neither the filtering step nor the resulting sample IDs or count. Combining the documented unfiltered splits directly would give 3,933 + 1,178 = 5,111 proteins. The exact membership of the reported 4,191 therefore remains unverified. The [recorded macro-F1](metrics.csv) averages the eight classes equally. The saved notebook output is 0.6953260225627327. The recovered PCA transformations and original checkpoints
reproduce the saved notebook macro-F1 exactly on 5,111 proteins.
The verified MCC is 0.7469349471557587 and balanced accuracy
is 0.7324765099507662.
The separately reported filtered dataset remains unverified.

![Original error percentages by true and predicted class](errors_by_class.png)

Rows are normalized by true class. The plotting cell sets the diagonal to zero to highlight errors. Diagonal zeros do not represent class recall. Source: original Task 3 test notebook, Git blob `dcdd1740ed365b2ad73d17299a82ddc00cbe9331`, cells 7–8 (zero-based).
