# Enzyme detection notebooks

| Comparison | Saved results |
|---|---|
| Classical features × classifier | [All 19 grid-search tables](../../results/feature_comparison/gridsearch_model_comparison/) and [score summary](../../results/feature_comparison/gridsearch_model_comparison/summary.csv) |
| ProtT5 baselines and neural network | [Five-fold CV](../../results/baselines/task1/cv_mcc.png) and [held-out test](../../results/baselines/task1/test_mcc_bootstrap.png) |
| Early neural networks | [AAindex runs 0–15](../../results/task1/model_selection/early_experiments/v2_aaindex/) |
| ProtT5/PCA neural network | [Selected run 131](../../results/task1/model_selection/run_131/), [CV archive](../../results/task1/model_selection/crossval_runs/) and [final training and test figures](../../results/task1/) |

The notebooks come from the original project. Local paths are relative to each notebook, and saved outputs were cleared. Re-running the experiments requires the original datasets and embeddings. Some exploratory cells write files or use older APIs. See [data preparation](../../docs/data_preparation.md) for the splits and feature definitions.

## Original notebook versions

| Notebook | Git blob in the original project |
|---|---|
| [Clustering threshold comparison](cluster_thresh_comp.ipynb) | `a0b4cad08958c90ac231fc04806806c3054148cd` |
| [Sequence identity thresholds](clustering.ipynb) | `6cbf88cc68da955afed38a6084d009a843fcb35b` |
| [Amino acid composition](gs_comp_aac.ipynb) | `d4e988d43b5df53ef3cdbf8fe9d660b498ad1f16` |
| [AAindex descriptors](gs_comp_aaindex.ipynb) | `d01fe676fb312f132ab69b4021ae50dcf3568f33` |
| [AAindex and n-grams](gs_comp_hybr_aaindex_ngrams.ipynb) | `e5daa8bd552f2579fa12ee0635edaa35df36a94e` |
| [AAindex and PseAAC](gs_comp_hybr_pseaac_aaindex.ipynb) | `36a56ee91f74a58bc41626d6bdaa18de3375f17e` |
| [One-hot encoding](gs_comp_ohe.ipynb) | `a6ee42e947a4c809c5604e03193cce49260b3d9d` |
| [Enzyme detection with ProtT5 embeddings](gs_comp_prott5.ipynb) | `47203df43ecdb278fe3c1df453a28a53dbf7e899` |
| [Pseudo amino acid composition](gs_comp_pseaac.ipynb) | `7d5292dbc0d0693627a2deb83dd58d469344f020` |
| [Neural networks with AAindex features](nn_comp_aaindex.ipynb) | `c8248016dbd3a285ea80f9acbff636a66e443b0d` |
| [Preparing the enzyme detection dataset](preprocessing.ipynb) | `4c816aa5ae4493ace4141a631e3cfa904d092a14` |
| [Protein embeddings with ProtT5](prott5_embedding_generator.ipynb) | `5ddad5c564c1120a7c7e55d94bb9c152674e4617` |
