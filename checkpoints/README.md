# Historical model checkpoints

`task1.pth` is the original saved Task 1 state dictionary, copied byte-for-byte from the final c30 / 128-dimensional model directory. Use the paired `configs/task1.yaml`. The checkpoint has not been loaded or revalidated in this curation step; only its Git blob hash has been verified. Use safe weights-only loading. The original fitted PCA is required for raw embeddings. Repository remains private pending team approval; publication of weights is subject to that approval.

`task2.pth` is the original Task 2 state dictionary for the usamp2000, 128-dimensional input model. Its Git blob matches the original. It loaded successfully into the archived Task2Model architecture and returned seven finite scores per synthetic input. See [the configuration](../configs/task2.yaml) and [evaluation record](../results/task2/README.md).
