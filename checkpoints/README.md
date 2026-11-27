# Pretrained models

- `task1.pth`: Binary enzyme classifier.
- `task2.pth`: EC main class classifier.
- `pca/`: Reconstructed Task 1 and original Task 2 PCA transformations.

## Usage

```python
import torch

from pca_transform import NpzPCA
from task_1.src.models import build_model_from_pretrained as load1
from task_2.src.models import build_model_from_pretrained as load2
from task_3.src.models import Task3Model

model = Task3Model(
    load1().cpu().eval(),
    load2().cpu().eval(),
    NpzPCA("checkpoints/pca/task1_pca_reconstructed.npz"),
    NpzPCA("checkpoints/pca/task2_pca_original.npz"),
)

# Synthetic example
predictions = model(torch.zeros(2, 1024))
print(predictions)
```

Replace the synthetic input with real 1024-dimensional ProtT5 embeddings.
The model returns 0 for non-enzymes or an EC main class from 1 to 7.

See the [main README](../README.md) for installation,
evaluation results and reproducibility details.
