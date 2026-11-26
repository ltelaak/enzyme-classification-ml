# Historical research code: curated from source blob 2495440e6f4e5dd8892a86c2f9e69b95337d2354.
# Run from repository root with original data supplied; not revalidated end to end.
from src.crossval import cross_validate
from src.preprocessing import create_dataset_from_h5, dim_reduction
from src.models import Task1Model, build_model
import torch.nn as nn
import torch.optim as optim
import torch
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #embedding_path = 'task_1/data/embeddings/embeddings_uniprot_exp_proteins_c30_flens2-98_usamp_enz/embeddings_uniprot_exp_proteins_c30_flens2-98_usamp_enz_train85.h5'
    #embedding_path = 'task_1/data/embeddings/embeddings_uniprot_exp_proteins_c30_flens2-98_enz/embeddings_uniprot_exp_proteins_c30_flens2-98_enz_train.h5'
    embedding_path = 'task_1/data/embeddings/embeddings_uniprot_exp_proteins_c30_flens2-98_enz/embeddings_uniprot_exp_proteins_c30_flens2-98_enz_train_dim128.h5'
    dataset = create_dataset_from_h5(embedding_path)

    model = Task1Model(dims=[128, 128,  64, 1], dropouts=[0.5, 0.3], activation=nn.LeakyReLU(), normalize=False)
    criterion = nn.BCELoss()
    cross_validate(model_in=model, dataset=dataset, optimizer_fn=optim.Adam, weight_decay=0.00025, loss_fn=criterion, n_splits=5, batch_size=32, lr=0.001, epochs=50)