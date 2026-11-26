# Load the trained classifiers

[task1.pth](task1.pth) contains the final enzyme detector. [task2.pth](task2.pth) contains the final EC classifier. Both files match the original state dictionaries.

Run this example from the repository root after installing the [model dependencies](../requirements.txt). It loads the weights and passes two synthetic 128-dimensional inputs through each network.

```python
from pathlib import Path

import torch
from task_1.src.models import build_model as build_detector
from task_2.src.models import build_model as build_ec_classifier

for task, build in [("task1", build_detector), ("task2", build_ec_classifier)]:
    model = build(Path("configs") / f"{task}.yaml").cpu().eval()
    weights = torch.load(Path("checkpoints") / f"{task}.pth",
                         map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    with torch.no_grad():
        scores = model(torch.zeros(2, 128))
    assert torch.isfinite(scores).all()
    print(task, tuple(scores.shape))
```

Expected output:

```text
task1 (2, 1)
task2 (2, 7)
```

Each classifier expects real embeddings projected with its own training-set PCA. The original fitted transformers are unavailable here. The historical evaluations are archived under [Task 1](../results/task1/) and [Task 2](../results/task2/).
