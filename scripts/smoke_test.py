"""Check model interfaces with synthetic inputs; not a performance evaluation."""
from pathlib import Path
import torch
from sklearn.decomposition import PCA
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
        assert first(torch.zeros(4, 128, device=device)).shape == (4, 1)
        assert second(torch.zeros(4, 128, device=device)).shape == (4, 7)
    # Fit synthetic PCA transforms only to test the interface.
    samples = torch.randn(140, 1024).numpy()
    pca_one = PCA(n_components=128, random_state=42).fit(samples)
    pca_two = PCA(n_components=128, random_state=43).fit(samples)
    pipeline = Task3Model(first, second, pca_one, pca_two)
    output = pipeline(torch.randn(4, 1024))
    assert output.shape == (4,)
    assert ((output >= 0) & (output <= 7)).all()
    print("Model interface smoke test passed (synthetic inputs only).")


if __name__ == "__main__":
    main()
