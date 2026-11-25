import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from .preprocessing import split_dataset
from .utils import visualize_cv_training
from pathlib import Path
from .utils import save_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, model_name, dataset, optimizer, criterion, epochs=100, early_stopping=True, patience=10, show_training_curves=True, batch_size=32):
    train_set, val_set = split_dataset(dataset, train_size=0.9)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    model.to(device)
    print('==== Training model ====')

    train_losses = list()
    val_losses = list()
    train_accs = list()
    val_accs = list()
    patience_counter = 0
    best_val_loss = 100000

    for e in range(epochs):
        print('Epoch ' + str(e))
        model.train()
        train_total_loss = 0
        train_correct = 0
        for feats, labels, _ in train_loader:
            feats, labels = feats.to(device=device), labels.to(device=device)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_total_loss += loss.cpu().item()
            with torch.no_grad():
                outputs_rounded = torch.round(outputs)
                train_correct += (outputs_rounded == labels).sum().item()
        train_avg_loss = round(train_total_loss / len(train_loader), 4)
        train_acc = train_correct / len(train_set)
        print('Training loss: ' + str(train_avg_loss) + '; Training accuracy:' + str(train_acc))

        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            feats = val_set.X.to(device=device)
            labels = val_set.y.to(device=device)
            outputs = model(feats)
            loss = criterion(outputs, labels)
            print(loss)
            val_loss += loss.cpu().item()
            outputs_rounded = torch.round(outputs)
            val_correct += (outputs_rounded == labels).sum().item()

        val_acc = val_correct / len(val_set)
        print('Validation loss: ' + str(val_loss) + '; Validation accuracy: ' + str(val_acc))

        train_losses.append(train_avg_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping')
                    break
    model_dir_path = (Path(__file__).parent / Path(f'../results/models/{model_name}')).resolve()
    model_dir_path.mkdir(exist_ok=True)
    save_config(path=model_dir_path, model=model, dataset=dataset, optimizer=optimizer, loss_fn=criterion, epochs=epochs)
    torch.save(model.state_dict(), model_dir_path / f'{model_name}.pth')
    if show_training_curves:
        visualize_cv_training(model_dir_path, train_losses, val_losses, train_accs, val_accs, len(train_losses))
    model.load_state_dict(best_model_state)
    return model
