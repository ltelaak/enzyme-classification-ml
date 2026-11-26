"""Check model interfaces with synthetic inputs; not a performance evaluation."""

from pathlib import Path
import numpy as np
import torch
from pca_transform import NpzPCA
from task_1.src.models import build_model as build_task1
from task_2.src.models import build_model as build_task2
from task_3.src.models import Task3Model


def main():
    root = Path(__file__).resolve().parents[1]
    torch.manual_seed(42)

    first = build_task1(root / "configs/task1.yaml").eval()
    second = build_task2(root / "configs/task2.yaml").eval()
    device = next(first.parameters()).device

    with torch.no_grad():
        out_one = first(torch.zeros(4, 128, device=device))
        out_two = second(torch.zeros(4, 128, device=device))

        assert out_one.shape == (4, 1)
        assert out_two.shape == (4, 7)
        assert torch.isfinite(out_one).all()
        assert torch.isfinite(out_two).all()

    # Load the recovered PCA transforms.
    pca_one = NpzPCA(root / "checkpoints/pca/task1_pca_reconstructed.npz")
    pca_two = NpzPCA(root / "checkpoints/pca/task2_pca_original.npz")

    # Check PCA transformations for numerical validity.
    test_input = torch.randn(4, 1024)

    for pca in (pca_one, pca_two):
        transformed = pca.transform(test_input.numpy())
        assert transformed.shape == (4, 128)
        assert np.isfinite(transformed).all()

    # Check the combined classification pipeline.
    pipeline = Task3Model(first, second, pca_one, pca_two)
    output = pipeline(test_input)

    assert output.shape == (4,)
    assert torch.isfinite(output).all()
    assert ((output >= 0) & (output <= 7)).all()

    print("Model interface smoke test passed (synthetic inputs only).")


if __name__ == "__main__":
    main()
