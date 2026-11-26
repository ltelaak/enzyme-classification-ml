"""Validate trained checkpoints against the historical test datasets."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)

from pca_transform import NpzPCA
from task_1.src.models import build_model as build_task1
from task_2.src.models import build_model as build_task2
from task_3.src.models import Task3Model


ROOT = Path(__file__).resolve().parents[1]


def read_embeddings(path):
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys())
        X = np.stack([f[key][:] for key in keys])

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite embeddings: {path}")

    return keys, X


def load_model(builder, task):
    model = builder(ROOT / f"configs/{task}.yaml").cpu()

    model.load_state_dict(
        torch.load(
            ROOT / f"checkpoints/{task}.pth",
            map_location="cpu",
            weights_only=True,
        )
    )

    return model.eval()


@torch.no_grad()
def predict(model, X, binary=False):
    outputs = []

    for start in range(0, len(X), 256):
        batch = torch.as_tensor(
            X[start:start + 256], dtype=torch.float32
        )

        scores = model(batch)

        if binary:
            pred = scores.round().reshape(-1)
        else:
            pred = scores.argmax(dim=1) + 1

        outputs.append(pred.cpu().numpy())

    return np.concatenate(outputs).astype(int)


def validate_pca(pca, raw, projected, task):
    raw_keys, X = read_embeddings(raw)
    projected_keys, Z = read_embeddings(projected)

    assert raw_keys == projected_keys, "Embedding IDs differ"

    recovered = pca.transform(X)
    rmse = np.sqrt(np.mean((recovered - Z) ** 2))

    print(f"{task} PCA RMSE: {rmse:.10e}")

    np.testing.assert_allclose(
        recovered, Z, atol=1e-6, rtol=0
    )

    return raw_keys, X, Z, recovered


def report(name, y, pred):
    print(f"\n{name}")
    print("Samples:", len(y))
    print("Macro-F1:", f1_score(y, pred, average="macro"))
    print("MCC:", matthews_corrcoef(y, pred))
    print("Balanced accuracy:", balanced_accuracy_score(y, pred))
    print("Confusion matrix:")
    print(confusion_matrix(y, pred))


def main():
    parser = argparse.ArgumentParser()

    for task in ("task1", "task2"):
        parser.add_argument(f"--{task}-raw", required=True)
        parser.add_argument(f"--{task}-projected", required=True)

    args = parser.parse_args()

    first = load_model(build_task1, "task1")
    second = load_model(build_task2, "task2")

    pca1 = NpzPCA(
        ROOT / "checkpoints/pca/task1_pca_reconstructed.npz"
    )

    pca2 = NpzPCA(
        ROOT / "checkpoints/pca/task2_pca_original.npz"
    )

    k1, X1, Z1, R1 = validate_pca(
        pca1, args.task1_raw, args.task1_projected, "Task 1"
    )

    k2, X2, Z2, R2 = validate_pca(
        pca2, args.task2_raw, args.task2_projected, "Task 2"
    )

    assert len(k1) == 5901
    assert len(k2) == 1178

    assert all(
        "Enzyme=True" in k or "Enzyme=False" in k
        for k in k1
    )

    y1 = np.array([int("Enzyme=True" in k) for k in k1])
    y2 = np.array([int(k.split("EC=")[1]) for k in k2])

    assert set(y2) == set(range(1, 8))

    pred1 = predict(first, R1, binary=True)
    pred2 = predict(second, R2)

    np.testing.assert_array_equal(
        pred1, predict(first, Z1, binary=True)
    )

    np.testing.assert_array_equal(
        pred2, predict(second, Z2)
    )

    report("Task 1", y1, pred1)
    report("Task 2", y2, pred2)

    np.testing.assert_array_equal(
        confusion_matrix(y1, pred1),
        [[3684, 249], [224, 1744]],
    )

    np.testing.assert_array_equal(
        confusion_matrix(y2, pred2),
        [
            [261, 13, 8, 10, 2, 3, 3],
            [8, 239, 23, 15, 8, 4, 3],
            [13, 26, 238, 17, 1, 3, 2],
            [14, 11, 8, 70, 10, 1, 1],
            [5, 3, 10, 11, 45, 1, 0],
            [2, 3, 2, 2, 3, 43, 0],
            [2, 2, 2, 0, 0, 2, 25],
        ],
    )

    # Reproduce the test composition in the historical
    # Task 3 notebook (not the filtered report dataset).
    X3 = np.concatenate([X1[y1 == 0], X2])
    y3 = np.concatenate([y1[y1 == 0], y2])

    assert len(y3) == 5111

    pipeline = Task3Model(first, second, pca1, pca2)

    pred3 = np.concatenate([
        pipeline(
            torch.as_tensor(
                X3[i:i + 256], dtype=torch.float32
            )
        ).numpy()
        for i in range(0, len(X3), 256)
    ])

    report("Task 3", y3, pred3)

    np.testing.assert_allclose(
        f1_score(y3, pred3, average="macro"),
        0.6953260225627327,
        atol=1e-10,
        rtol=0,
    )

    print("\nAll historical checkpoint checks passed.")


if __name__ == "__main__":
    main()
