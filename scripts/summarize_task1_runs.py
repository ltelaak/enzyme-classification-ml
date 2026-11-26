"""Summarize archived Task 1 CV YAML files without retraining any models."""
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "results/task1/model_selection"


def read_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def fold_mcc(fold):
    try:
        tp, tn, fp, fn = (float(fold[k]) for k in ("tp", "tn", "fp", "fn"))
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return (tp * tn - fp * fn) / denominator if denominator else 0.0
    except (KeyError, TypeError, ValueError):
        return None


def main():
    directories = list((ROOT / "crossval_runs").glob("run_*")) + [ROOT / "run_131"]
    fields = [
        "run_folder", "config_file", "metrics_file", "dataset", "dataset_size",
        "dims", "dropouts", "activation", "normalize", "optimizer", "lr",
        "weight_decay", "loss_fn", "epochs", "folds_with_metrics",
        "recorded_average_accuracy", "derived_mean_fold_mcc", "read_error",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    for folder in sorted(directories, key=lambda p: int(p.name.removeprefix("run_"))):
        config = next(iter(sorted(folder.glob("*_config.yaml"))), None)
        metrics = next(iter(sorted(folder.glob("*_metrics.yaml"))), None)
        row = {"run_folder": folder.name, "config_file": config.name if config else "",
               "metrics_file": metrics.name if metrics else ""}
        try:
            cfg = read_yaml(config) if config else {}
            model, train, data = (cfg.get(key, {}) for key in ("model", "training", "dataset"))
            opt = train.get("optimizer", {})
            act = model.get("activation", {})
            row.update(dataset=data.get("name"), dataset_size=data.get("size"),
                       dims=json.dumps(model.get("dims")), dropouts=json.dumps(model.get("dropouts")),
                       activation=act.get("name") if isinstance(act, dict) else act,
                       normalize=model.get("normalize"),
                       optimizer=opt.get("name") if isinstance(opt, dict) else opt,
                       lr=opt.get("lr") if isinstance(opt, dict) else None,
                       weight_decay=opt.get("weight_decay") if isinstance(opt, dict) else None,
                       loss_fn=train.get("loss_fn"), epochs=train.get("epochs"))
            if metrics:
                met = read_yaml(metrics)
                folds = [v for k, v in met.items() if re.fullmatch(r"Fold\s+\d+", str(k)) and isinstance(v, dict)]
                mccs = [score for fold in folds if (score := fold_mcc(fold)) is not None]
                row.update(folds_with_metrics=len(folds),
                           recorded_average_accuracy=met.get("Average accuracy"),
                           derived_mean_fold_mcc=statistics.mean(mccs) if mccs else None)
        except (OSError, yaml.YAMLError, TypeError, AttributeError, ValueError) as exc:
            row["read_error"] = type(exc).__name__
        writer.writerow(row)


if __name__ == "__main__":
    main()
