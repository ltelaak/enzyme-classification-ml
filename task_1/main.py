# Historical research code: curated from source blob ac58fe11dc24579812f5c847f1eb824044e0a92b.
# Run from repository root with original data supplied; not revalidated end to end.
# Main script
# Used to create trained models and eval

from src.crossval import cross_validate
from src.preprocessing import create_dataset_from_h5, split_dataset, dim_reduction, dim_reduction_train_test, dataset_to_h5
from src.models import Task1Model, build_model_from_pretrained
from src.train import train_model
import torch.nn as nn
import torch.optim as optim
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from src.train import train_model
from src.models import build_model
from src.infer import infer

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_path = Path('task_1/data/embeddings/embeddings_uniprot_exp_proteins_c30_flens2-98_enz/embeddings_uniprot_exp_proteins_c30_flens2-98_enz_train_dim128.h5')
    test_path = Path('task_1/data/embeddings/embeddings_uniprot_exp_proteins_c30_flens2-98_enz/embeddings_uniprot_exp_proteins_c30_flens2-98_enz_test_dim128.h5')

    train_set = create_dataset_from_h5(train_path, train_path.stem)
    test_set = create_dataset_from_h5(test_path, test_path.stem)

    #model = build_model_from_pretrained('task1_model3_embeddings_uniprot_exp_proteins_c30_flens2-98_enz_train_dim128')
    # model = build_model('results/task1/model_selection/run_131/run_131_config.yaml')
    # criterion = nn.BCELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00025)
    # model = train_model(model=model, model_name=f'task1_model3_{train_set.name}', dataset=train_set, optimizer=optimizer, criterion=criterion, epochs=100, patience=20)

    #cross_validate(model_in=model, dataset=train_set, optimizer_fn=optim.Adam, weight_decay=0.00025, loss_fn=criterion, n_splits=5, batch_size=32, lr=0.001, epochs=50)
    # Portfolio adaptation: load the same historical checkpoint from its new location.
    model = build_model(Path('configs/task1.yaml'))
    if model is None:
        raise ValueError('Invalid Task 1 model configuration')
    model.load_state_dict(torch.load('checkpoints/task1.pth', map_location=device, weights_only=True))
    model.to('cpu')
    infer(model=model, dataset=test_set, metrics=True, dim_reduce=False)
    # train_set, test_set = split_dataset(dataset, train_size=0.8, stratify=True)

    # model = Task1Model(dims=[1024, 683, 683, 1], dropouts=[0.5, 0.3], activation=nn.LeakyReLU(), normalize=False)
    # criterion = nn.BCELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # trained_model = train_model(model, train_set, optimizer, criterion, 200)

    # torch.save(trained_model.state_dict(), r"task_1/models/test_model.pth")
    # model = build_model('results/task1/model_selection/crossval_runs/run_93/run_93_config.yaml')
    # model = Task1Model(dims=[1024, 683, 683, 1], dropouts=[0.5, 0.3], activation=nn.LeakyReLU(), normalize=False)
#     model.load_state_dict(torch.load(r'task_1/models/test_model.pth'))
#     model.to(device=device)
#     model.eval()

#     loader = DataLoader(dataset=test_set, batch_size=32, shuffle=False)

#     total_acc = 0
#     test = 0
#     for feats, labels in loader:
#         feats, labels = feats.to(device=device), labels.to(device=device)
#         preds = model(feats)
#         with torch.no_grad():
#             preds_rounded = torch.round(preds)
#             if test == 0:
#                 print(preds_rounded)
#                 print(labels)
#         total_acc += accuracy_score(labels.cpu(), preds_rounded.cpu())
#         test += 1
#     print(round(total_acc / len(loader), 4))

# test_list = [dataset.get_keys(i) for i in test_set.indices]
# print(test_list[:32])
    # dim_reduction(dataset=dataset, n_dims=256)
    # train_model(model=model, model_name=f'task1_second_model_{dataset.name}', dataset=dataset, optimizer=optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00015), criterion=nn.BCELoss(), epochs=200, patience=50)