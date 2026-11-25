import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from .preprocessing import split_dataset
from pathlib import Path
from .utils import visualize_training, save_config, class_distribution_from_dataset
from .crossval import get_cv_metrics, train_one_epoch, evaluate, make_weighted_sampler
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, model_name, dataset, optimizer, criterion, epochs=100, early_stopping=True, patience=10, show_training_curves=True, batch_size=32, sampler_alpha=1):
    train_set, val_set = split_dataset(dataset, train_size=0.9, stratify=True)

    train_labels_idx = (train_set.y - 1).long()
    sampler = make_weighted_sampler(train_labels_idx, 7, alpha=sampler_alpha)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    model.to(device)
    print('==== Training model ====')

    metric_idx = 1
    min_delta = 0.001

    train_losses = []
    val_losses = []

    train_metrics = []
    val_metrics = []

    patience_counter = 0
    best_val_loss = 100000

    best_model_state = copy.deepcopy(model.state_dict())

    for e in range(epochs):
        print('Epoch ' + str(e))

        train_loss = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, loss_fn=criterion)
        val_loss = evaluate(model=model, loader=val_loader, loss_fn=criterion)

        val_metric_list = get_cv_metrics(model=model, loader=val_loader)
        train_metric = get_cv_metrics(model=model, loader=train_loader)[metric_idx]
        val_metric = val_metric_list[metric_idx]

        #val_metric_list[3].plot()

        print(f'Training loss: {train_loss:.4f}; Training metric: {train_metric:.4f}')
        print(f'Validation loss: {val_loss:.4f}; Validation metric: {val_metric:.4f}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_metrics.append(train_metric)
        val_metrics.append(val_metric)

        if early_stopping:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping!')
                    break
        else:
            best_model_state = copy.deepcopy(model.state_dict())
    model_dir_path = (Path(__file__).parent / Path(f'../results/models/{model_name}')).resolve()
    model_dir_path.mkdir(exist_ok=True, parents=True)
    model.load_state_dict(best_model_state)
    save_config(path=model_dir_path, model=model, dataset=dataset, optimizer=optimizer, loss_fn=criterion, epochs=epochs, patience=patience)
    torch.save(model.state_dict(), model_dir_path / f'{model_name}.pth')
    if show_training_curves:
        loss_curve = visualize_training(train_list=train_losses, val_list=val_losses, metric='loss')
        metric_curve = visualize_training(train_list=train_metrics, val_list=val_metrics, metric='macro_f1')
        loss_curve.savefig(model_dir_path / 'loss_curve.png', format='png')
        metric_curve.savefig(model_dir_path / 'macro_f1_curve.png', format='png')
        return model, loss_curve, metric_curve, 
    return model
