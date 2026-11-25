import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from .preprocessing import create_dataset_from_h5
import numpy as np
import yaml

# Graphs the distribution of ec main classes in the given DataFrame as a bar chart
def class_distribution(df:pd.DataFrame, name='', show=False):   
    with plt.style.context('default'):  
        # Compute class counts
        class_dict = {ec: len([n for n in df['EC main class'] if n == ec]) for ec in range(1, 8)}
        keys = list(range(1, 8))
        values = [class_dict[i] for i in range(1, 8)]

        # Create figure
        fig, ax = plt.subplots()
        bars = ax.bar(keys, values)
        ax.grid(False)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, str(height), ha='center', va='bottom')

        ax.set_ylabel('Enzyme Count')
        ax.set_title(f'Distribution of EC main classes in Dataset {name}')
        if show:
            plt.show()
            plt.close(fig)
        else:
            plt.close(fig)
            return fig

def class_distribution_from_dataset(dataset:Dataset, name='', label_list=[], show=False):
    if type(dataset) == Subset:
        label_list = label_list
    else:
        label_list = dataset.y

    if len(label_list.shape) == 2:
        df = pd.DataFrame(np.argmax(label_list, axis=1) + 1, columns=['EC main class'])
    else:
        df = pd.DataFrame(label_list, columns=['EC main class'])
    return class_distribution(df, show=show)

# Shows distribution of the lengths of sequences in dataset as histogram
def length_distribution(df, name=''):
    len_list = [len(s) for s in df['Sequence']]
    sns.histplot(len_list, bins=50, kde=True)
    plt.title("Distribution of Sequence Lengths in Dataset " + name)
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.show()

def df_to_tsv(df, path):
    df.to_csv(path, sep='\t', index=False)

# Writes given DataFrame to a .fasta file at the given path 
def df_to_fasta(df, path):
    if set(['Entry', 'Entry Name', 'Sequence', 'EC main class']).issubset(set(df.columns)):
        with open(path, 'w') as outfile:
            columns = df.columns
            ent_idx = list(columns).index('Entry') + 1
            name_idx = list(columns).index('Entry Name') + 1
            seq_idx = list(columns).index('Sequence') + 1
            ez_idx = list(columns).index('EC main class') + 1

            for row in df.itertuples():
                ez = row[ez_idx]
                outfile.write('>sp|' + row[ent_idx] + '|' + row[name_idx] + ' ' + 'EC=' + str(ez) + '\n')
                outfile.write(row[seq_idx] + '\n')
        return
    elif set(['Entry', 'Sequence', 'EC main class']).issubset(set(df.columns)):
        with open(path, 'w') as outfile:
            columns = df.columns
            ent_idx = list(columns).index('Entry') + 1
            seq_idx = list(columns).index('Sequence') + 1
            ez_idx = list(columns).index('EC main class') + 1

            for row in df.itertuples():
                ez = row[ez_idx]
                outfile.write('>sp|' + row[ent_idx] + '|' + ' ' + 'EC=' + str(ez) + '\n')
                outfile.write(row[seq_idx] + '\n')
        return
    raise Exception("DataFrame doesn't contain the necessary columns.")

def clear_folder(path:str | Path):
    path = Path(path)
    if Path(path).is_dir():
        shutil.rmtree(path)
        path.mkdir()
    else:
        raise Exception('Given path is not a directory!')
    
def get_h5_length(h5_path):
    return len(create_dataset_from_h5(h5_path=h5_path))

def visualize_training(train_list, val_list, metric=''):
    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(train_list) + 1)
    ax.plot(epochs, train_list, label='Training loss')
    ax.plot(epochs, val_list, label='Validation loss')
    
    ax.set_xlabel('Epoch')
    if metric == '':
        ax.set_ylabel('Score')
    else:
        ax.set_ylabel(metric)
    ax.set_title(f'Training vs validation {metric} over epochs.')
    ax.legend()
    ax.grid(True)

    plt.close(fig) 
    
    return fig

def visualize_cv_spaghetti(train_list, val_list, metric='', only_means=False):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # --- Helper: Forward Fill ---
    def get_padded_mean(ragged_list):
        max_len = max(len(row) for row in ragged_list)
        padded_rows = []
        for row in ragged_list:
            padded_rows.append(row + [row[-1]] * (max_len - len(row)))
        return np.mean(padded_rows, axis=0)

    # --- 1. PLOT TRAINING (Blue) ---
    if not only_means:
        for i, row in enumerate(train_list):
            epochs = range(1, len(row) + 1)
            
            # Line
            ax.plot(epochs, row, color='blue', alpha=0.4, linewidth=1, 
                    label='Train Folds' if i == 0 else "")
            
            # Dot at the end (The "Stop" point)
            ax.scatter(epochs[-1], row[-1], color='blue', s=15, alpha=0.8)

    # Mean Train
    train_mean = get_padded_mean(train_list)
    ax.plot(range(1, len(train_mean)+1), train_mean, color='blue', linewidth=2.5, linestyle='--', label='Train Mean')

    # --- 2. PLOT VALIDATION (Red) ---
    if not only_means:
        for i, row in enumerate(val_list):
            epochs = range(1, len(row) + 1)
            
            # Line
            ax.plot(epochs, row, color='orange', alpha=0.4, linewidth=1,
                    label='Val Folds' if i == 0 else "")
            
            # Dot at the end
            ax.scatter(epochs[-1], row[-1], color='orange', s=15, alpha=0.8)

    # Mean Val
    val_mean = get_padded_mean(val_list)
    ax.plot(range(1, len(val_mean)+1), val_mean, color='orange', linewidth=2.5, linestyle='--', label='Val Mean')

    # --- 3. FORMATTING ---
    ax.set_title("Training vs Validation Cross Validation History")
    ax.set_xlabel("Epochs")
    if metric == '':
        ax.set_ylabel('Score')
    else:
        ax.set_ylabel(metric)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return fig

def calculate_gap_means(train_loss_list, val_loss_list, train_f1_list, val_f1_list):
    n_folds = len(train_loss_list)
    gap_list_loss = list()
    gap_list_f1 = list()

    for i in range(n_folds):
        before = 4
        after = 2

        best_idx = np.argmin(val_loss_list[i])

        window = slice(max(best_idx - before, 0), best_idx + after + 1)

        t_fold_loss_avg = np.mean(train_loss_list[i][window])
        v_fold_loss_avg = np.mean(val_loss_list[i][window])

        t_fold_f1_avg = np.mean(train_f1_list[i][window])
        v_fold_f1_avg = np.mean(val_f1_list[i][window])

        fold_gap_f1 = abs(t_fold_f1_avg / v_fold_f1_avg - 1)
        fold_gap_loss = abs(v_fold_loss_avg / (t_fold_loss_avg + 1e-7) - 1)

        gap_list_loss.append(fold_gap_loss)
        gap_list_f1.append(fold_gap_f1)

    return np.mean(gap_list_loss), np.mean(gap_list_f1)

def save_config(path, model, dataset=None, optimizer=None, loss_fn=None, epochs=None, patience=0):
    config = {
        'model': {
            'model_type': type(model).__name__,
            'dims': model.dims,
            'dropouts': model.dropouts,
            'activation': {
                'name': type(model.activation).__name__,
                'negative_slope': model.activation.negative_slope
            },
            'normalize': model.normalize, 
            'dtype': str(model.dtype)
        },
        'training': {
            'optimizer': {
                'name': type(optimizer).__name__,
                'lr': optimizer.param_groups[0]['lr'],
                'weight_decay': optimizer.param_groups[0]['weight_decay']
            },
            'loss_fn': type(loss_fn).__name__,
            'epochs': epochs,
            'patience': patience
        },
        'dataset': {
            'name': dataset.name,
            'size': len(dataset),
            'feature_dim': str(tuple((dataset[0][0].shape)))
        }
    }
    path = Path(path).resolve()
    if path.is_dir():
        file_path = path / f'{path.name}_config.yaml'
    else:
        file_path = path
    with open(file_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    return config

def get_class_distribution_list(dataset, relative=True, label_list=[]):
    if type(dataset) == Subset:
        label_list = label_list
    else:
        label_list = dataset.y
    class_list_abs = list()
    for c in range(1,8):
        class_list_abs.append(len([e for e in label_list if e == c]))
    n = sum(class_list_abs)
    class_list_rel = [item / n for item in class_list_abs]
    if relative:
        return class_list_rel
    else:
        return class_list_abs
