# Historical research code: curated from source blob 1bcb88f9a222e386bda8851ccaf3db54ad6fa261.
# Run from repository root with original data supplied; not revalidated end to end.
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix
import argparse
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
def infer(model, dataset, metrics=True, fasta=False, dim_reduce=False, pca_path=None):
    print(f'Total sequences to infer: {len(dataset)}')

    if dim_reduce:
        dim_reduction(dataset=dataset, n_dims=128, pca_path=pca_path or ROOT / "checkpoints/pca/task1_pca_reconstructed.npz")
    
    if isinstance(model, str):
        pretrained = True
    else:
        pretrained = False

    if pretrained:
        infer_path = ROOT / "outputs" / "task1" / model / f"infer_{dataset.name}"
        infer_path.mkdir(parents=True, exist_ok=True)

        if dataset.y is None:
            metrics = False

        if fasta:
            output_file = infer_path / f'infer_fasta_{dataset.name}'
        else:
            output_file = infer_path / f'infer_{dataset.name}'

        model = build_model_from_pretrained(model)


    model.eval()

    with torch.no_grad():
        param = next(model.parameters())
        preds = model(dataset.X.to(device=param.device, dtype=param.dtype))
        if pretrained:
            with open(output_file, 'w') as file:
                for i in range(0, len(dataset)):
                    file.write(dataset.keys[i] + ';' + f'{preds[i].item():.0f}' + '\n')
            print('Saved inference results to: ' + str(output_file.resolve()))

    if metrics and dataset.y is not None:
        labels = torch.as_tensor(dataset.y).detach().cpu().numpy().reshape(-1)
        preds_rounded = torch.round(preds).cpu().numpy().reshape(-1)
        acc = accuracy_score(labels, preds_rounded)
        mcc = matthews_corrcoef(labels, preds_rounded)
        auc = roc_auc_score(labels, preds_rounded)
        conf_display = ConfusionMatrixDisplay.from_predictions(labels, preds_rounded, cmap=colormaps.get('Greens'), display_labels=["Non-Enzyme", "Enzyme"], normalize='all', values_format='.1%')
        conf_display.im_.colorbar.remove()
        conf_display.ax_.set_ylabel("Actual label")
        for annotation in conf_display.text_.ravel():
            annotation.set_fontsize(20)
        conf = confusion_matrix(labels, preds_rounded)


        print('Inference metrics:')
        print(f'accuracy: {acc} | mcc: {mcc} | auc: {auc}')
        print('Confusion Matrix:')
        print(conf)
        plt.show()

        if pretrained:
            conf_display.figure_.savefig(infer_path / 'infer_confusion_matrix.png', format='png', dpi=300)
            plt.close()
            with open(infer_path / f'infer_metrics_{dataset.name}.txt', 'w') as f:
                f.write(f'Dataset: {dataset.name}\n\n')
                f.write(f'Inference metrics:\n')    
                f.write(f'accuracy: {acc} | mcc: {mcc} | auc: {auc}\n')
                f.write(str(conf))


def infer_fasta(model_name, fasta_path, metrics=False, error_bars=False, n_samples=0):
    dataset = create_dataset_from_fasta(fasta_path=fasta_path, features='prott5', unsqueeze=True)

    infer(model=model_name, dataset=dataset, metrics=metrics, fasta=True, dim_reduce=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name')
    parser.add_argument('-p', '--fasta_path')
    parser.add_argument('-met', '--metrics', action='store_true')
    args = parser.parse_args()

    infer_fasta(args.model_name, args.fasta_path, args.metrics)

if __name__ == '__main__':
    from task_1.src.models import build_model, build_model_from_pretrained
    from task_1.src.preprocessing import dim_reduction, generate_prott5_embeddings, Task1Dataset, create_dataset_from_fasta
    main()

else:
    from .models import build_model, build_model_from_pretrained
    from .preprocessing import dim_reduction, generate_prott5_embeddings, Task1Dataset, create_dataset_from_fasta