import pandas as pd
import numpy as np
from protlearn import preprocessing
from pathlib import Path
from Bio import SeqIO
import h5py
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
from sklearn.decomposition import PCA
import pickle
from pyfaidx import Fasta
import matplotlib.pyplot as plt
from pyfaidx import Fasta
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel, T5EncoderModel, T5Tokenizer
from pathlib import Path
import os

class Task2Dataset(Dataset):
    '''
    X: Sequences, encodings or embeddings
    y: Labels
    name: Name of the dataset for future file naming.
    Keys: Header information for the sequences.
    '''
    def __init__(self, X, y=None, name='', keys=list()):
        self.X = X
        self.y = y
        self.name = name
        self.keys = keys
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx], self.keys[idx]
        else:
            return self.X[idx], self.keys[idx]


def prepare_main_ec_classes(df, explode=True):
    df_copy = df.copy()
    df_copy['EC number'] = get_main_ec_classes_from_column(df_copy['EC number'])
    df_copy.rename(columns={'EC number': 'EC main class'}, inplace=True)
    if explode:
        return explode_multilabel_ec_classes(df_copy)
    else:
        return df_copy

def get_main_ec_classes_from_column(column):
    column_list = column.apply(lambda x : x.split(';') if type(x) == str else 0)
    column_list_main_class = column_list.apply(lambda ec_list : list(set(int(ec.split('.')[0]) for ec in ec_list)) if type(ec_list) == list else [0])
    return column_list_main_class

def explode_multilabel_ec_classes(df):
    return df.explode(column='EC main class')

def filter_lengths(df:pd.DataFrame, i:int, j:int):
    '''Removes sequences from DataFrame with length under the i-th and over the j-th percentile.
    
    # Parameters: 
        df (pandas.core.frame.DataFrame): DataFrame to be altered.
        i (int): Lower boundary percentile.
        j (int): Upper boundary percentile.
        
    # Returns:
        pandas.core.frame.DataFrame: DataFrame with filtered lengths.
    '''
    if (i > j):
        raise Exception("Lower boundary percentile can't be larger than upper boundary.")
    len_list = [len(s) for s in df['Sequence']]
    P1 = np.percentile(len_list, i)
    P2 = np.percentile(len_list, j)
    print(f'Removing sequences with length < {P1} and > {P2}')
    return df[(df['Sequence'].str.len() >= P1) & (df['Sequence'].str.len() <= P2)]

def undersample(df):
    if 'EC main class' in df.columns:
        class_partition_dict = dict()
        min_size = 1000000
        for i in range(1, 8):
            temp_df = df[df['EC main class'] == i]
            class_partition_dict[i] = temp_df
            min_size = min(len(temp_df), min_size)
        for i in range(1, 8):
            class_partition_dict[i] = class_partition_dict[i].sample(n=min_size)
        return pd.concat(class_partition_dict.values()).reset_index(drop=True)
    raise Exception("DataFrame doesn't contain column 'Enzyme'.")

def undersample_cap(df, cap):
    if 'EC main class' in df.columns:
        class_partition_dict = dict()
        for i in range(1, 8):
            temp_df = df[df['EC main class'] == i]
            class_partition_dict[i] = temp_df
        for i in range(1, 8):
            if len(class_partition_dict[i]) >= cap:
                class_partition_dict[i] = class_partition_dict[i].sample(n=cap)
        return pd.concat(class_partition_dict.values()).reset_index(drop=True)
    raise Exception("DataFrame doesn't contain column 'Enzyme'.")

def remove_unnatural_seqs(df:pd.DataFrame):
    '''Filters out "unnatural" aminoacid sequences from given DataFrame.

    # Parameters:
        df (pd.core.frame.DataFrame): DataFrame to be altered.

    # Returns:
        pd.core.frame.DataFrame: Dataframe with duplicate and unnatural sequences removed.
    '''
    seq_list = list(df['Sequence'])
    natural = preprocessing.remove_unnatural(seq_list)
    return df[df['Sequence'].isin(natural)]

def cluster_df(df:pd.DataFrame, cluster_tsv_path: str | Path):
    '''Applies mmseqs2 cluster map to given DataFrame, resulting in a clusterd DataFrame.

    # Parameters:
        df (pandas.core.frame.DataFrame): DataFrame to cluster.
        cluster_tsv_path (str | pathlib.Path): Path to the cluster map tsv.

    # Returns:
        pandas.core.frame.DataFrame: DataFrame to which the clustering has been applied.
    '''
    clusters = pd.read_csv(cluster_tsv_path, sep='\t', header=None, names=['Representative', 'Member'])
    df_rep = truncate(clusters, ['Representative']).drop_duplicates().reset_index(drop=True)
    if 'Entry' in df.columns:
        temp = df.merge(df_rep, left_on='Entry', right_on='Representative')
        return truncate(df=temp, columns=['Entry', 'Entry Name', 'Sequence', 'EC main class'])
    raise Exception("DataFrame doesn't contain column 'Entry'.")

def truncate(df:pd.DataFrame, columns:list):
    '''Truncate given DataFrame to specified columns.
    
    # Parameters:
        df (pandas.core.frame.DataFrame): DataFrame to be truncated.
        
    # Returns:
        pandas.core.frame.DataFrame: Truncated DataFrame.
    '''
    if set(columns).issubset(set(df.columns)):
        return df.loc[:, columns]
    raise Exception("Dataframe doesn't contain given column names!")

def remove_duplicate_seqs(df:pd.DataFrame):
    '''Remove duplicate sequences from DataFrame without keeping representatives.

    # Parameters:
        df (pandas.core.frame.DataFrame): DataFrame to be altered.

    # Returns:
        pandas.core.frame.DataFrame: DataFrame with duplicate sequences removed.
    '''
    return df.drop_duplicates(subset='Sequence', keep=False)

def split_dataset_fasta(fasta_path:str | Path, train_size:float=0.8, stratify=False):
    '''Split the dataset from the given fasta into train and test sets and save them to task_1/data/split/.
    
    # Parameters:
        fasta_path (str): Path to the fasta to be split.
        train_size (float): Relative size of the train set.
        
    # Returns:
    '''
    fasta_path = Path(fasta_path)
    rec_list = list(SeqIO.parse(fasta_path, 'fasta'))
    if stratify:
        label_list = [int(rec.description.split(' ')[1].split('=')[1]) for rec in rec_list]
        train_recs, test_recs = train_test_split(rec_list, test_size=1-train_size, random_state=53, stratify=label_list)
    else:
        train_recs, test_recs = train_test_split(rec_list, test_size=1-train_size, random_state=53)
    split_dir_path = Path('../data/split/temp')
    train_split_path = split_dir_path.with_name(f'{fasta_path.stem}_train{int(train_size*100)}.fasta')
    test_split_path = split_dir_path.with_name(f'{fasta_path.stem}_test{int(train_size*100)}.fasta')

    SeqIO.write(sequences=train_recs, handle=train_split_path, format='fasta')
    SeqIO.write(sequences=test_recs, handle=test_split_path, format='fasta')
    return train_split_path, test_split_path

def split_and_create_h5(fasta_path:str | Path, train_size:float=0.8, stratify=False):
    '''Combines methods split_dataset_fasta() and filter_h5_from_fasta to create a new folder in embeddings with the main embedding file and the embeddings of its train and test sets.
    
    # Parameters:
        fasta_path (str | pathlib.Path): Path to the .fasta file to be read.
        train_size (float): Relative size of the train set.
        
    # Returns:
    '''
    train_split_path, test_split_path = split_dataset_fasta(fasta_path=fasta_path, train_size=train_size, stratify=stratify)
    folder_path = Path(f'../data/embeddings/embeddings_uniprot_{fasta_path.stem}')
    folder_path.mkdir(exist_ok=True)
    base_emb_path = folder_path / f'{folder_path.stem}.h5'
    train_emb_path = folder_path / f'embeddings_uniprot_{train_split_path.stem}.h5'
    test_emb_path = folder_path / f'embeddings_uniprot_{test_split_path.stem}.h5'
    filter_h5_from_fasta('../data/embeddings/embeddings_uniprot_exp_enzymes.h5', fasta_path=fasta_path, out_path=base_emb_path) 
    filter_h5_from_fasta(base_emb_path, fasta_path=train_split_path, out_path=train_emb_path)  
    filter_h5_from_fasta(base_emb_path, fasta_path=test_split_path, out_path=test_emb_path)  

    return train_emb_path, test_emb_path
 

def filter_h5_from_fasta(h5_path:str | Path, fasta_path:str | Path, out_path:str | Path=''):
    h5_path = Path(h5_path)
    fasta_path = Path(fasta_path)
    it = SeqIO.parse(fasta_path, format='fasta')
    description_list = list(item.description for item in it)
    fasta_key_list = list(item.split('|')[1] for item in description_list)
    fasta_label_list = list(int(item.split(' ')[1].split('=')[1]) for item in description_list)

    if out_path == '':
        out_path = Path(f'../data/embeddings/embeddings_uniprot_{fasta_path.stem}.h5')
    else:
        out_path = Path(out_path)
        with h5py.File(h5_path, 'r') as h5, h5py.File(out_path, 'w') as h5_out:
            for key in h5.keys():
                temp_key = key.split(' ')[0]
                if temp_key in fasta_key_list:
                    key_index = fasta_key_list.index(temp_key)
                    fasta_label = fasta_label_list[key_index]
                    h5_out.create_dataset(name=f"{key.split('|')[0]}| EC={fasta_label}", data=h5.get(key)[:])


def create_dataset_from_h5(h5_path:str | Path, name:str='', unsqueeze:bool=True, multiclass=False):
    '''Create a Task1Dataset from a h5 embedding file.
    
    # Parameters:
        h5_path (str): Path to the .h5 embedding file to be read.
        name (str): Desired name of the dataset.
        unsqueeze (bool): Unsqueezes the labels if True.
        
    # Returns:
        dadtaset (Task1Dataset): Dataset including embeddings'''
    if name == '':
        name = Path(h5_path).stem
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        embeddings = list()

        for key in keys:
            embeddings.append(f[key][:])

    embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)

    has_labels = 'EC=' in keys[0]
    if has_labels:
        labels = [int(k.split('=')[1]) for k in keys]
        if unsqueeze:
            labels = torch.tensor(np.array(labels), dtype=torch.long).unsqueeze(dim=1)
        else:
            labels = torch.tensor(np.array(labels), dtype=torch.long)
        dataset = Task2Dataset(X=embeddings, y=labels, name=name, keys=keys)
        if multiclass:
            convert_dataset_to_multiclass(dataset)
        return dataset
    else:
        return Task2Dataset(X=embeddings, name=name, keys=keys)


def dim_reduction_train_test(train_set:Task2Dataset, test_set:Task2Dataset, n_dims:int, out_path:str | Path=None):
    dtype = train_set.X.dtype
    X_train_raw = train_set.X 
    X_test_raw = test_set.X
    if X_train_raw.shape[1] < n_dims:
        raise Exception('n_dims is higher than the feature dimensions!')
    pca = PCA(n_components=n_dims)
    X_train_pca = torch.tensor(pca.fit_transform(X_train_raw), dtype=dtype)
    X_test_pca = torch.tensor(pca.transform(X_test_raw), dtype=dtype)

    with open(out_path / f'{train_set.name}_pca.pkl', 'wb') as f:
        pickle.dump(pca, f)

    train_set.X = X_train_pca
    test_set.X = X_test_pca
    train_set.name = f'{train_set.name}_dim{n_dims}'
    test_set.name = f'{test_set.name}_dim{n_dims}'
    if out_path is not None:
        out_path = Path(out_path)
        dataset_to_h5(train_set, out_path / f'{train_set.name}.h5')
        dataset_to_h5(test_set, out_path / f'{test_set.name}.h5')

def dim_reduction(dataset:Task2Dataset, n_dims:int, pca_path=''):
    if pca_path and Path(pca_path).suffix == ".npz":
        from pca_transform import NpzPCA

        pca = NpzPCA(pca_path)
        if pca.components_.shape[0] != n_dims:
            raise ValueError("PCA output dimension mismatch.")

        X = dataset.X
        dataset.X = torch.as_tensor(
            pca.transform(X.detach().cpu().numpy()),
            dtype=X.dtype,
            device=X.device
        )
        return

    if pca_path == '':
        pca = PCA(n_components=n_dims)
    else:
        pca_path = Path(pca_path)
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)

    X_raw = dataset.X 
    dtype = dataset.X.dtype
    
    if pca_path == '':
        X_red = torch.tensor(pca.fit_transform(X_raw), dtype=dtype)
    else:
        X_red = torch.tensor(pca.transform(X_raw), dtype=dtype)
    dataset.X = X_red

def dataset_to_h5(dataset:Task2Dataset, path:str | Path):
    with h5py.File(path, 'w') as f:
        for item in dataset:
            emb = item[0]
            key = item[2]
            f.create_dataset(name=key, data=emb)

def create_label_map(df_raw):
    df = prepare_main_ec_classes(df_raw, False)
    df_trunc = truncate(df, ['Entry', 'EC main class'])
    d = dict(zip(df['Entry'], df['EC main class']))
    with open('../data/exp_enzymes_label_map.pkl', 'wb') as outfile:
        pickle.dump(d, outfile)

def convert_raw_h5(h5_path, name):
    h5_path = Path(h5_path)
    with open('../data/exp_enzymes_label_map.pkl', 'rb') as file:
        label_map = pickle.load(file)
        
    with h5py.File(h5_path, 'r') as infile,  h5py.File(h5_path.with_name(name), 'w') as outfile:
        key_list = list(infile.keys())

        for key in key_list:
            new_key = f'{key} | EC main class={label_map[key]}'
            embedding = infile.get(key)[:]
            outfile.create_dataset(name=new_key, data=embedding)

def get_label_map():
    with open('../data/exp_enzymes_label_map.pkl', 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict

def convert_dataset_to_multiclass(dataset):
    multiclass_list = list()
    for i in dataset.y:
        temp = [0] * 7
        temp[int(i.item()) - 1] = 1
        multiclass_list.append(temp)
    dataset.y = torch.tensor(multiclass_list, dtype=dataset.y.dtype)

def convert_dataset_to_singleclass(dataset):
    singleclass_list = list()
    for i in dataset.y:
        singleclass_list.append(i.argmax() + 1)
    dataset.y = torch.tensor(singleclass_list, dtype=dataset.y.dtype)

def split_dataset(dataset:Dataset, train_size:float=0.8, stratify:bool=False):
    '''Split the given dataset into train and test datasets.
    
    # Parameters:
        dataset (torch.utils.data.Dataset): Dataset to split.
        train_size (float): Relative size the training dataset should have.
        stratify (bool): Stratifies the datasets regarding the label classes.
    
    # Returns:
        train_set (torch.utils.data.Dataset): Train dataset.
        test_set (torch.utils.data.Dataset): Test dataset.
    '''
    if stratify:
        X_train, X_test, y_train, y_test, keys_train, keys_test = train_test_split(dataset.X, dataset.y, dataset.keys, test_size=1-train_size, shuffle=True, stratify=dataset.y, random_state=43)
    else:
        X_train, X_test, y_train, y_test, keys_train, keys_test = train_test_split(dataset.X, dataset.y, dataset.keys, test_size=1-train_size, shuffle=True, random_state=43)
    
    train_set = Task2Dataset(X_train, y_train, name=dataset.name + '_train', keys=keys_train)
    test_set = Task2Dataset(X_test, y_test, name=dataset.name + '_test', keys=keys_test)        

    return train_set, test_set

def create_dataset_from_fasta(fasta_path:str | Path, name:str='', features:str='', unsqueeze:bool=False, known_seqs=True):
    '''Creates a Task1Dataset from a given fasta file.
    Depending on format of fasta, labels are included in the dataset or not.
    If wanted, the sequences in the dataset can also be saved as feature vectors.
    
    # Parameters:
        fasta_path (str | pathlib.Path): Path to the .fasta file to be read.
        name (str): Desired name for the dataset.
        features (str): Desired feature extraction method to be applied to the sequences in the dataset.
        unsqueeze (bool): Only if features='prott5'. Unsqueezes the labels if True.
        
    # Returns:
        dataset (Task1Dataset): Dataset including sequences, encodings, or embeddings, labels and the respective headers for identification.
    '''
    fasta_path = Path(fasta_path)
    dataset_name = name
    if name == '':
        dataset_name = Path(fasta_path).stem
    if features == '':
       # X = sequences   
       pass
    elif features == 'ohe':
        #X = batch_encode_and_flatten(seq_list=sequences)
        pass
    elif features == 'aaindex':
        #X = aaindex1(X=sequences, standardize='zscore')[0]
        pass
    elif features == 'aac':
        #X = aac(X=sequences, relative=True)[0]
        pass
    elif features == 'paac':
        #X = paac(X=sequences)
        pass
    elif features == 'prott5':
        if known_seqs:
            temp_h5_path = Path('../temp/temph5.h5')
            filter_h5_from_fasta(h5_path=f'../data/embeddings/embeddings_uniprot_exp_enzymes.h5', fasta_path=fasta_path, out_path=temp_h5_path)
            dataset = create_dataset_from_h5(temp_h5_path, name=dataset_name, unsqueeze=unsqueeze, multiclass=False)
        else:
            X, keys = generate_prott5_embeddings(fasta_path)
            labels = [int(key.split('=')[1]) for key in keys]
            dataset = Task2Dataset(X=X, y=labels, name=dataset_name, keys=keys)

    return dataset

def generate_prott5_embeddings(path):
    model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"  

    if model_name.startswith("EvolutionaryScale/"):
        from esm.models.esm3 import ESM3
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, SamplingConfig, LogitsConfig

    embedding_type = "per_prot"  

    max_sequence_length = 2000  

    MODEL_NAME_TO_SHORT_KEY_MAP = {
        "Rostlab/prot_t5_xl_half_uniref50-enc": "prot_t5",
        "Rostlab/ProstT5_fp16": "prost_t5",
        "ElnaggarLab/ankh-base": "ankh_base",
        "ElnaggarLab/ankh-large": "ankh_large",
        "facebook/esm2_t6_8M_UR50D": "esm2_8m",
        "facebook/esm2_t12_35M_UR50D": "esm2_35m",
        "facebook/esm2_t30_150M_UR50D": "esm2_150m",
        "facebook/esm2_t33_650M_UR50D": "esm2_650m",
        "EvolutionaryScale/esm3-sm-open-v1": "esm3_open",
        "EvolutionaryScale/esmc-300m-2024-12": "esmc_300m",
        "EvolutionaryScale/esmc-600m-2024-12": "esmc_600m",
    }


    def seq_preprocess(df, model_type="esm"):
        # Store raw sequence before preprocessing
        df["raw_sequence"] = df["sequence"]

        # Replace special amino acids with X
        df["sequence"] = df["sequence"].str.replace("[BJOUZ]", "X", regex=True)

        # Add spaces between amino acids for specific models
        if model_type in ["prost_t5", "prot_t5"]:
            df["sequence"] = df["sequence"].apply(lambda seq: " ".join(list(seq)))

        # Add prefix for ProstT5 model
        if model_type == "prost_t5":
            df["sequence"] = df["sequence"].apply(lambda seq: "<AA2fold> " + seq)

        return df

    def setup_model(checkpoint):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = None
        tokenizer = None
        mod_type = "unknown"

        print(f"Setting up model: {checkpoint} on {device}")

        if "esm2" in checkpoint:
            mod_type = "esm_transformer"
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = EsmModel.from_pretrained(checkpoint)
        elif "esm3-sm-open-v1" in checkpoint:
            mod_type = "native_esm3"
            model = ESM3.from_pretrained("esm3-open")
        elif checkpoint == "EvolutionaryScale/esmc-300m-2024-12":
            mod_type = "native_esmc"
            model = ESMC.from_pretrained("esmc_300m")
        elif checkpoint == "EvolutionaryScale/esmc-600m-2024-12":
            mod_type = "native_esmc"
            model = ESMC.from_pretrained("esmc_600m")
        elif "ankh" in checkpoint:
            mod_type = "ankh"
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = T5EncoderModel.from_pretrained(checkpoint)
        elif "prot_t5" in checkpoint or "ProstT5" in checkpoint:
            is_prostt5 = "prostt5" in checkpoint.lower()
            mod_type = "prost_t5" if is_prostt5 else "prot_t5"
            tokenizer = T5Tokenizer.from_pretrained(
                checkpoint, do_lower_case=not is_prostt5
            )
            model = T5EncoderModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
            if device.type == "cuda":
                model = model.half()
        else:
            raise ValueError(f"Unknown model checkpoint type: {checkpoint}")

        return model.to(device), tokenizer, mod_type

    def create_embedding(
        checkpoint,
        df,
        emb_type="per_prot",
        output_file="protein_embeddings.h5",
        max_len=2000,
    ):
        print("Setting up model...")
        model_instance, tokenizer_instance, mod_type = setup_model(checkpoint)
        model_instance.eval()
        df_processed = seq_preprocess(df.copy(), mod_type)
        device = model_instance.device

        def compute_embedding(
            sequence, current_emb_type, current_mod_type, model, tokenizer
        ):
            if current_mod_type.startswith("native_esm"):
                protein = ESMProtein(sequence=sequence)
                tokens = model.encode(protein).to(device)
                raw_per_residue_embeddings = None
                if current_mod_type == "native_esm3":
                    out = model.forward_and_sample(
                        tokens, SamplingConfig(return_per_residue_embeddings=True)
                    )
                    raw_per_residue_embeddings = (
                        out.per_residue_embedding.squeeze(0).cpu().numpy()
                    )
                elif current_mod_type == "native_esmc":
                    out = model.logits(
                        tokens, LogitsConfig(sequence=True, return_embeddings=True)
                    )
                    raw_per_residue_embeddings = out.embeddings.squeeze(0).cpu().numpy()
                else:
                    raise ValueError(f"Unknown native ESM model type: {current_mod_type}")

                # Directly slice to remove BOS/EOS tokens
                cleaned_per_residue_embeddings = raw_per_residue_embeddings[1:-1, :]

                if current_emb_type == "per_prot":
                    return cleaned_per_residue_embeddings.mean(axis=0)
                elif current_emb_type == "per_res":
                    return cleaned_per_residue_embeddings
            else:  # Transformers-based models
                if tokenizer is None:
                    raise ValueError(
                        f"Tokenizer not available for model type: {current_mod_type}"
                    )

                tokenization_input = sequence
                tokenizer_call_kwargs = {
                    "return_tensors": "pt",
                    "max_length": max_len + 2,
                    "truncation": True,
                    "padding": True,
                    "add_special_tokens": True,  # Adds EOS for T5 models
                }

                inputs = tokenizer(tokenization_input, **tokenizer_call_kwargs).to(device)

                with torch.no_grad():
                    embeddings_tensor = model(**inputs).last_hidden_state.cpu()
                if embeddings_tensor.ndim == 3 and embeddings_tensor.shape[0] == 1:
                    embeddings_tensor = embeddings_tensor.squeeze(0)
                embeddings_np_arr = embeddings_tensor.numpy()

                # Slicing logic based on model type
                token_embeddings_for_avg = None
                per_residue_slice = None

                if current_mod_type == "esm_transformer":
                    # Skip <cls> and <eos>
                    token_embeddings_for_avg = embeddings_np_arr[1:-1, :]
                    per_residue_slice = embeddings_np_arr[1:-1, :]
                elif current_mod_type == "prost_t5":
                    # Skip <AA2fold> (and potential BOS) and <eos>
                    token_embeddings_for_avg = embeddings_np_arr[1:-1, :]
                    per_residue_slice = embeddings_np_arr[1:-1, :]
                elif current_mod_type in ["prot_t5", "ankh"]:  # Regular ProtT5 and Ankh
                    # Skip <eos>
                    token_embeddings_for_avg = embeddings_np_arr[:-1, :]
                    per_residue_slice = embeddings_np_arr[:-1, :]
                else:  # Should not happen if mod_type is correctly set
                    token_embeddings_for_avg = embeddings_np_arr
                    per_residue_slice = embeddings_np_arr

                if current_emb_type == "per_prot":
                    if token_embeddings_for_avg.shape[0] == 0:
                        print(
                            f"Warning: No token embeddings to average for sequence after slicing for model {current_mod_type}. Original shape: {embeddings_np_arr.shape}"
                        )
                        return np.array([])  # Or handle as error appropriately
                    return token_embeddings_for_avg.mean(axis=0)
                elif current_emb_type == "per_res":
                    return per_residue_slice

            raise ValueError(
                f"Invalid embedding type '{current_emb_type}' or model type '{current_mod_type}' combination if not returned by above"
            )

        invalid_headers = df_processed[df_processed["header"].str.contains("/")][
            "header"
        ].tolist()
        if invalid_headers:
            error_msg = (
                "ERROR: The following sequence headers contain '/' which is not allowed in HDF5 dataset names (HDF5 prior to groups):\n"
                + "\n".join(invalid_headers)
                + "\nPlease remove or replace '/' characters in these headers and try again."
            )
            raise ValueError(error_msg)

        print("Generating embeddings...")
        with h5py.File(output_file, "a") as hdf:
            for _, row_data in tqdm(df_processed.iterrows(), total=len(df_processed)):
                sequence_val = row_data["sequence"]
                header_val = row_data["header"]

                # Skip sequences that exceed the maximum length
                if len(row_data["raw_sequence"]) > max_len:
                    tqdm.write(
                        f"Skipping sequence {header_val} (length {len(row_data['raw_sequence'])}) as it exceeds max length {max_len}."
                    )
                    continue

                if header_val in hdf:
                    tqdm.write(f"Skipping existing embedding for {header_val}")
                    continue
                embedding_result = compute_embedding(
                    sequence_val, emb_type, mod_type, model_instance, tokenizer_instance
                )
                hdf.create_dataset(
                    name=header_val,
                    data=embedding_result.astype(np.float32)
                    if isinstance(embedding_result, np.ndarray)
                    else embedding_result,
                )

        del model_instance, tokenizer_instance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fasta_path = Path(path)

    # Get the short key for the selected model_name, fallback to sanitized full name
    short_model_key = MODEL_NAME_TO_SHORT_KEY_MAP.get(
        model_name, model_name.replace("/", "_")
    )

    output_file = Path('../temp/test').with_name(f'infer_temp_embeddings_{fasta_path.stem}_{short_model_key}.h5')

    headers, sequences = read_fasta(fasta_path)
    df = pd.DataFrame({"header": headers, "sequence": sequences})
    print(f"Processing {len(df)} sequences for model {model_name}...")

    create_embedding(
        model_name,
        df,
        emb_type=embedding_type,
        output_file=output_file,
        max_len=max_sequence_length,
    )

    with h5py.File(output_file, 'r') as f:
        key_list = list(f.keys())
        embeddings_list = list()
    
        for key in key_list:
            embeddings_list.append(f[key][:])
    output_embeddings = torch.tensor(np.array(embeddings_list), dtype=torch.float32)
    os.remove(output_file)
    return output_embeddings, key_list

def read_fasta(fasta_path: str | Path):
    '''Reads the given fasta. Method adapted from ProtT5 Embedding Generator ClickThrough.
    
    # Parameters:
        fasta_path (str | pathlib.Path): Path to fasta to be read.
        
    # Returns:
        headers (list[str]): List of all headers in the fasta.
        sequences (list[str]): List of all sequences in the fasta.
    '''
    headers = []
    sequences = []
    with Fasta(str(fasta_path)) as fasta_data:
        use_long_name = 1 if 'EC=' in fasta_data[0].long_name else 0
        for seq_record in fasta_data:
            if use_long_name:
                headers.append(seq_record.long_name)
            else:
                headers.append(seq_record.name)
            sequences.append(str(seq_record))
    return headers, sequences

def preprocess_df(df, flens=[2, 98], usamp=False, cap=0, create_h5=True, train_size=0.85, stratify=True, new_dims=128, c_thresh=0.3):
    from .utils import df_to_fasta, df_to_tsv, class_distribution

    print('Class distribution in the beginning:')
    class_distribution(df=df, show=True)
    plt.show()

    if len(flens) == 0:
        flens_tag = ''
    else:
        p1 = flens[0]
        p2 = flens[1]
        df = filter_lengths(df, p1, p2)
        flens_tag = f'_flens{p1}-{p2}'

    print('Class distribution after filtering lengths:')
    class_distribution(df=df, show=True)
    plt.show()

    if usamp:
        if cap == 0:
            usamp_tag = f'_usamp'
            df = undersample(df)
        else:
            usamp_tag = f'_usamp{cap}'
            df = undersample_cap(df, cap=cap)
    else:
        usamp_tag = ''

    print('Class distribution after undersampling:')
    class_distribution(df=df, show=True)
    plt.show()

    out_file = Path(f'../data/preprocessed/exp_enzymes_c{int(c_thresh*100)}{flens_tag}{usamp_tag}_exp')
    df_to_fasta(df, f'{out_file}.fasta')
    df_to_tsv(df, f'{out_file}.tsv')
    
    if create_h5:
        fasta = Path(f'{out_file}.fasta')
        train_emb_path, test_emb_path = split_and_create_h5(fasta, train_size, stratify=stratify)

    if create_h5 and new_dims > 0:
        train_set = create_dataset_from_h5(train_emb_path)
        test_set = create_dataset_from_h5(test_emb_path)

        dim_reduction_train_test(train_set=train_set, test_set=test_set, n_dims=new_dims, out_path=train_emb_path.parent)

    return df
