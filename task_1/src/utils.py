import matplotlib.pyplot as plt 
import seaborn as sns
import yaml
import torch.optim as optim
import os
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from pathlib import Path
import numpy as np
import shutil
from Bio import SeqIO

# Shows distribution of the lengths of sequences in dataset as histogram
def length_distribution(df, name=''):
    len_list = [len(s) for s in df['Sequence']]
    sns.histplot(len_list, bins=50, kde=True)
    plt.title("Distribution of Sequence Lengths in Dataset " + name)
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.show()

# Graphs the distribution of non-enzyme and enzyme entries in the given DataFrame as a bar chart
def class_distribution(df, name=''):     
    non_enzymes = [n for n in df['Enzyme'] if n == 0]
    enzymes = [n for n in df['Enzyme'] if n == 1]
    keys = ['Enzyme Entries', 'Non-enzyme entries']
    values = [len(enzymes), len(non_enzymes)]
    bars = plt.bar(keys, values)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, str(height), ha='center', va='bottom')
    plt.ylabel('Protein Count')
    plt.title('Distribution of Enzyme and Non-Enzyme Protein Entries in Dataset ' + name)
    plt.show()

def fasta_to_tsv(fasta_path:str | Path):
    fasta_path = Path(fasta_path)
    out_path = fasta_path.with_name(f'{fasta_path.stem}.tsv')
    temp_dict = {'Entry': list(), 'Entry Name': list(), 'Sequence': list(), 'Enzyme': list()}
    records = SeqIO.parse(fasta_path, 'fasta')
    for record in records:
        split = record.description.split('|')
        split2 = split[2].split(' ')
        temp_dict['Entry'].append(split[1])
        temp_dict['Entry Name'].append(split2[0])
        temp_dict['Sequence'].append(str(record.seq))
        temp_dict['Enzyme'].append(1 if 'Enzyme=True' in split2[1] else 0)
    df = pd.DataFrame(temp_dict)
    df.to_csv(out_path, sep='\t', index=False)

# Writes given DataFrame to a .fasta file at the given path 
def df_to_fasta(df, path):
    if set(['Entry', 'Entry Name', 'Sequence', 'Enzyme']).issubset(set(df.columns)):
        with open(path, 'w') as outfile:
            columns = df.columns
            ent_idx = list(columns).index('Entry') + 1
            name_idx = list(columns).index('Entry Name') + 1
            seq_idx = list(columns).index('Sequence') + 1
            ez_idx = list(columns).index('Enzyme') + 1

            for row in df.itertuples():
                ez = (row[ez_idx] == 1)
                outfile.write('>sp|' + row[ent_idx] + '|' + row[name_idx] + ' ' + 'Enzyme=' + str(ez) + '\n')
                outfile.write(row[seq_idx] + '\n')
        return
    raise Exception("DataFrame doesn't contain the necessary columns.")

# Writes given DataFrame to a .tsv file at the given path 
def df_to_tsv(df, path):
    df.to_csv(path, sep='\t', index=False)

def save_config(path, model, dataset=None, optimizer=None, loss_fn=None, epochs=None):
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
            'epochs': epochs
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

def create_run_folder():
    base_dir = Path(__file__).resolve().parent.parent
    # Create folder logs/
    log_path = base_dir / 'results' / 'performance_eval' / 'nn_crossval_logs'
    log_path.mkdir(exist_ok=True)
    # Create folder run_x/
    path = log_path / f'run_{len(list(log_path.iterdir()))}'
    path.mkdir(exist_ok=True)
    return path

def visualize_cv_training(path, tlm, vlm, tam, vam, num_epochs):
    file_path_loss = os.path.join(path, f'{os.path.basename(path)}_loss.png')
    file_path_acc = os.path.join(path, f'{os.path.basename(path)}_acc.png')
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, tlm, label='Training loss')
    plt.plot(epochs, vlm, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average training vs validation loss over splits')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path_loss)
    plt.show()

    plt.plot(epochs, tam, label='Training accuracy')
    plt.plot(epochs, vam, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Average training vs validation Accuracy over splits')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path_acc)
    plt.show()

def log_training(path, msg):
    msg_path = os.path.join(path, f'{os.path.basename(path)}_cv_training.log')
    with open(msg_path, 'w') as f:
        f.write(msg)

def log_metrics(path, metrics):
    file_path = os.path.join(path, f'{os.path.basename(path)}_metrics.yaml')
    with open(file_path, 'w') as f:
        yaml.dump(metrics, f, sort_keys=False)

def calculate_search_grid(estimator, X_train, y_train, X_test, y_test, n_iter, param_dist, randomized=True):
    mcc_scorer = make_scorer(matthews_corrcoef)

    if randomized:
        columns = ['mean_fit_time', 'params', 'mean_test_accuracy', 'std_test_accuracy', 'mean_test_mcc', 'std_test_mcc']
        rand_grid = RandomizedSearchCV(estimator, param_dist, n_iter=n_iter, scoring={'accuracy': 'accuracy', 'mcc': mcc_scorer}, refit='mcc')
        rand_grid.fit(X_train, y_train)
        param_columns = [c for c in rand_grid.cv_results_.keys() if 'param_' in c]
        df_rand_grid = pd.DataFrame(rand_grid.cv_results_, columns=param_columns + columns)

        # Add baseline in case RandomizedSearch didn't go over it
        baseline = RandomizedSearchCV(estimator, {k: [v] for k, v in estimator.get_params().items() if k in param_dist}, n_iter=1, scoring={'accuracy': 'accuracy', 'mcc': mcc_scorer}, refit='mcc')
        baseline.fit(X_train, y_train)
        df_baseline = pd.DataFrame(baseline.cv_results_, columns=df_rand_grid.columns)

        # Concatenate RandomizedSearch results with baseline and remove duplicates (removes baseline if already in RS results)
        # Add ranks in the end
        df = pd.concat([df_baseline, df_rand_grid])
        df.drop_duplicates(subset=param_columns, inplace=True)
        df['rank_test_mcc'] = df['mean_test_mcc'].rank(method='min', ascending=False)
        df['rank_test_accuracy'] = df['mean_test_accuracy'].rank(method='min', ascending=False)

        # For best classifier, calculate performance (accuracy and mcc) on held out test set and add as last 2 columns
        if baseline.best_score_ > rand_grid.best_score_: 
            best_estimator = baseline.best_estimator_
        else:
            best_estimator = rand_grid.best_estimator_
    else:
        columns = ['mean_fit_time', 'params', 'mean_test_accuracy', 'std_test_accuracy', 'mean_test_mcc', 'std_test_mcc', 'rank_test_accuracy', 'rank_test_mcc']
        grid = GridSearchCV(estimator=estimator, param_grid=param_dist, scoring={'accuracy': 'accuracy', 'mcc': mcc_scorer}, refit='mcc')
        grid.fit(X_train, y_train)
        param_columns = [c for c in grid.cv_results_.keys() if 'param_' in c]
        df = pd.DataFrame(grid.cv_results_, columns=param_columns + columns)
        best_estimator = grid.best_estimator_

    y_pred = best_estimator.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_mcc = matthews_corrcoef(y_test, y_pred)

    df['final_test_acc'] = np.nan
    df.loc[df['rank_test_accuracy'] == 1, 'final_test_acc'] = test_acc
    df['final_test_mcc'] = np.nan
    df.loc[df['rank_test_mcc'] == 1, 'final_test_mcc'] = test_mcc
    return df


def clear_folder(path:str | Path):
    path = Path(path)
    if Path(path).is_dir():
        shutil.rmtree(path)
        path.mkdir()
    else:
        raise Exception('Given path is not a directory!')
