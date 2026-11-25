import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .models import build_model
from .utils import get_class_distribution_list, class_distribution_from_dataset
import torchmetrics
import pandas as pd
import copy
from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_weighted_sampler(labels_idx: torch.Tensor, num_classes: int, alpha: float = 1.0):
    labels_idx = labels_idx.cpu().long()
    class_counts = torch.bincount(labels_idx, minlength=num_classes).float()
    class_weights = 1.0 / torch.clamp(class_counts, min=1.0).pow(alpha)
    sample_weights = class_weights[labels_idx]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for feats, labels, _ in loader:
        feats, labels = feats.to(device=device), labels.to(device=device)
        optimizer.zero_grad()
        outputs = model(feats)
        loss = loss_fn(outputs, labels - 1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = round(total_loss / len(loader), 4)
    return avg_loss

def evaluate(model, loader, loss_fn):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for feats, labels, _ in loader:
            feats, labels = feats.to(device=device), labels.to(device=device)
            preds = model(feats)
            loss = loss_fn(preds, labels - 1)
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        return avg_loss

def cross_validate_wandb(model_in, dataset, optimizer_fn, weight_decay, loss_fn, n_splits, batch_size, lr, epochs, patience=0, sampler_alpha=1):
    train_loss_tracking = []
    val_loss_tracking = []

    train_f1_tracking = []
    val_f1_tracking = []

    train_mcc_tracking = []
    val_mcc_tracking = []

    f1_idx = 1
    mcc_idx = 2

    cv_metrics = list()
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    log_msg = ''

    for fold, (train_idx, val_idx) in enumerate(skfold.split(dataset.X, dataset.y)):
        log_msg += f'Fold {fold + 1}\n'
        print(f'Fold {fold + 1}')

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

        # train_labels = dataset.y[train_set.indices]
        # dist = get_class_distribution_list(train_set, relative=False, label_list=train_labels)
        # inverse_dist = [1 / n for n in dist]
        # weights = [inverse_dist[y - 1] for y in train_labels]

        train_labels_idx = (dataset.y[train_idx] - 1).long()  # 0..6
        sampler = make_weighted_sampler(train_labels_idx, 7, alpha=sampler_alpha)

        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

        train_loss_list = []
        val_loss_list = []

        train_f1_list = []
        val_f1_list = []

        train_mcc_list = []
        val_mcc_list = []

        best_val_loss = 10000000
        patience_counter = 0

        model = build_model(model_in)
        optimizer = optimizer_fn(model.parameters(), lr=lr, weight_decay=weight_decay)
        min_delta = 0.001
        best_model_state = copy.deepcopy(model.state_dict())

        for e in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
            val_loss = evaluate(model, val_loader, loss_fn)
            train_metrics = get_cv_metrics(model, train_loader)
            val_metrics = get_cv_metrics(model, val_loader)

            train_f1 = train_metrics[f1_idx]
            val_f1 = val_metrics[f1_idx]

            train_mcc = train_metrics[mcc_idx]
            val_mcc = val_metrics[mcc_idx]

            log_msg += f'Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}'
            print(f'Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')
            
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            train_f1_list.append(train_f1)
            val_f1_list.append(val_f1)

            train_mcc_list.append(train_mcc)
            val_mcc_list.append(val_mcc)

            if patience > 0:
                if val_loss >= best_val_loss - min_delta:
                    patience_counter += 1
                    if patience_counter >= patience:
                        log_msg += f'Early stopping!'
                        print('Early stopping!')
                        break
                else:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
            else:
                best_model_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_state)
        cv_metrics.append(get_cv_metrics(model, val_loader)[:3])

        train_loss_tracking.append(train_loss_list)
        val_loss_tracking.append(val_loss_list)

        train_f1_tracking.append(train_f1_list)
        val_f1_tracking.append(val_f1_list)

        train_mcc_tracking.append(train_mcc_list)
        val_mcc_tracking.append(val_mcc_list)

    df = pd.DataFrame(cv_metrics, columns=['balanced_accuracy', 'macro_f1', 'mcc'])
    df.insert(0, 'run', range(n_splits))
    return train_loss_tracking, val_loss_tracking, train_f1_tracking, val_f1_tracking, train_mcc_tracking, val_mcc_tracking, df

def get_cv_metrics(model, loader):
    balanced_accuracy = torchmetrics.classification.Accuracy(task='multiclass', num_classes=7, average='macro').to(device=device)
    macro_f1 = torchmetrics.classification.F1Score(task='multiclass', num_classes=7, average='macro').to(device=device)
    mcc = torchmetrics.classification.MatthewsCorrCoef(task='multiclass', num_classes=7).to(device=device)
    conf = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=7).to(device=device)

    model.eval()
    with torch.no_grad():
        for feats, labels, _ in loader:
            feats, labels = feats.to(device=device), labels.to(device=device)
            preds = model(feats)

            labels_as_idx = labels - 1
            balanced_accuracy.update(preds, labels_as_idx)
            macro_f1.update(preds, labels_as_idx)
            mcc.update(preds, labels_as_idx)
            conf.update(preds, labels_as_idx)
        return [balanced_accuracy.compute().cpu().item(), macro_f1.compute().cpu().item(), mcc.compute().cpu().item(), conf]
