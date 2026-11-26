import torch
from pathlib import Path
import argparse
from torch.utils.data import DataLoader
from matplotlib import colormaps
from .preprocessing import dim_reduction, create_dataset_from_fasta
from .models import build_model_from_pretrained
import torchmetrics


ROOT = Path(__file__).resolve().parents[2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
def infer(model, dataset, batch_size=16, metrics=True, fasta=False, dim_reduce=True, pca_path=None):

    if dim_reduce:
        dim_reduction(dataset=dataset, n_dims=128, pca_path=pca_path or ROOT / "checkpoints/pca/task2_pca_original.npz")

    print(f'Total sequences to infer: {len(dataset)}')

    if isinstance(model, str):
        pretrained = True
    else:
        pretrained = False

    if pretrained:
        infer_path = ROOT / "outputs" / "task2" / model / f"infer_{dataset.name}"
        infer_path.mkdir(parents=True, exist_ok=True)

        if dataset.y is None:
            metrics = False

        if fasta:
            output_file = infer_path / f'infer_fasta_{dataset.name}'
        else:
            output_file = infer_path / f'infer_{dataset.name}'

        model = build_model_from_pretrained(model)
    
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    if pretrained:
        model.eval()
        with torch.no_grad():
            param = next(model.parameters())
            preds_one_hot = model(dataset.X.to(device=param.device, dtype=param.dtype))
            preds = torch.argmax(preds_one_hot, dim=1) + 1
        with open(output_file, 'w') as file:
            for i in range(0, len(dataset)):
                file.write(dataset.keys[i] + ';' + f'{preds[i].item():.0f}' + '\n')
        print('Saved inference results to: ' + str(output_file.resolve()))

    if metrics and dataset.y is not None:
        metrics = get_cv_metrics(model=model, loader=loader)
        weighted_f1 = get_weighted_f1(model=model, loader=loader)
        print('Inference metrics:')
        print(f'Balanced Acc: {metrics[0]} | Macro F1: {metrics[1]} | MCC: {metrics[2]} | Weighted F1: {weighted_f1}')
        print(f'Confusion Matrix:')
        conf_str = str(metrics[3].compute())
        print(conf_str)
        metrics[3].plot(cmap=colormaps.get('Blues'))

        if pretrained:
            with open(infer_path / f'infer_metrics_{dataset.name}.txt', 'w') as f:
                f.write(f'Dataset: {dataset.name}\n\n')
                f.write(f'Inference metrics:\n')
                f.write(f'Balanced Accuracy: {metrics[0]} | Macro F1: {metrics[1]} | MCC: {metrics[2]} | Weighted F1: {weighted_f1}\n')
                f.write(f'Confusion Matrix:\n')
                f.write(conf_str)

            fig, ax = metrics[3].plot(cmap=colormaps.get('Blues'))
            fig.savefig(infer_path / f'infer_{dataset.name}_cm.png', format='png')


def infer_fasta(model_name, fasta_path, metrics=False, known_seqs=False):
    dataset = create_dataset_from_fasta(fasta_path=fasta_path, features='prott5', unsqueeze=False, known_seqs=known_seqs)
    infer(model=model_name, dataset=dataset, metrics=metrics, fasta=True)

def get_cv_metrics(model, loader):
    device = next(model.parameters()).device
    balanced_accuracy = torchmetrics.classification.Accuracy(task='multiclass', num_classes=7, average='macro').to(device=device)
    macro_f1 = torchmetrics.classification.F1Score(task='multiclass', num_classes=7, average='macro').to(device=device)
    mcc = torchmetrics.classification.MatthewsCorrCoef(task='multiclass', num_classes=7).to(device=device)
    conf = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=7).to(device=device)

    model.eval()
    with torch.no_grad():
        for feats, labels, _ in loader:
            feats, labels = feats.to(device=device), labels.to(device=device)
            preds = model(feats)

            labels_as_idx = labels.reshape(-1).long() - 1
            balanced_accuracy.update(preds, labels_as_idx)
            macro_f1.update(preds, labels_as_idx)
            mcc.update(preds, labels_as_idx)
            conf.update(preds, labels_as_idx)
        return [balanced_accuracy.compute().cpu().item(), macro_f1.compute().cpu().item(), mcc.compute().cpu().item(), conf]
    
def get_weighted_f1(model, loader):
    device = next(model.parameters()).device
    weighted_f1 = torchmetrics.classification.F1Score(task='multiclass', num_classes=7, average='weighted').to(device=device)
    model.eval()
    with torch.no_grad():
        for feats, labels, _ in loader:
            feats, labels = feats.to(device=device), labels.to(device=device)
            preds = model(feats)

            labels_as_idx = labels.reshape(-1).long() - 1
            weighted_f1.update(preds, labels_as_idx)
    return weighted_f1.compute().cpu().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name')
    parser.add_argument('-p', '--fasta_path')
    parser.add_argument('-met', '--metrics', action='store_true')
    args = parser.parse_args()

    infer_fasta(args.model_name, args.fasta_path, args.metrics)

if __name__ == "__main__":
    main()
