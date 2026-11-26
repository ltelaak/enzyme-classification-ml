import torch.nn as nn
import torch.optim as optim
import torch
import yaml
from pathlib import Path
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Task1Model(nn.Module):
    def __init__(self, dims=[1024, 683, 1], dropouts=[0.5], activation=nn.LeakyReLU(), normalize=False, dtype=torch.float32):
        super().__init__()
        self.activation = activation
        self.dims = dims
        self.dropouts = dropouts
        self.normalize = normalize
        self.dtype = dtype
        self.layers = []

        for i in range(0, len(self.dims)-1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1], dtype=self.dtype))
            if i < (len(self.dims) - 2):
                self.layers.append(self.activation)  
            if self.normalize and i < (len(self.dims) - 2):
                self.layers.append(nn.LayerNorm(self.dims[i+1], dtype=self.dtype))     
            if i < len(self.dropouts) and dropouts[i] != 0:
                self.layers.append(nn.Dropout(self.dropouts[i])) 

        self.layers.append(nn.Sigmoid())
        self.layers_seq = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers_seq(x)
    
def build_model(model_config):
    if type(model_config) == dict:
        try:
            model_type = Task1Model
            dims = model_config['model']['dims']
            dropouts = model_config['model']['dropouts']
            activation = nn.LeakyReLU(model_config['model']['activation']['negative_slope'])
            normalize = model_config['model']['normalize']
            dtype = torch.float32
        except KeyError:
            print("model_config doesn't have the correct structure!")
            return None
        except TypeError:
            print("model_config doesn't have the correct structure!")
            return None
    else:
        with open(model_config, 'r') as f:
            config = yaml.safe_load(f)
            model_type = Task1Model
            dims = config['model']['dims']
            dropouts = config['model']['dropouts']
            activation = nn.LeakyReLU(config['model']['activation']['negative_slope'])
            normalize = config['model']['normalize']
            dtype = torch.float32
    return model_type(dims=dims, dropouts=dropouts, activation=activation, normalize=normalize, dtype=dtype).to(device=device)

def build_model_from_pretrained(model_name="task1"):
    root = Path(__file__).resolve().parents[2]

    if model_name == "task1":
        config_file = root / "configs/task1.yaml"
        model_file = root / "checkpoints/task1.pth"
    else:
        model_path = root / "task_1/results/models" / model_name
        config_file = model_path / f"{model_name}_config.yaml"
        model_file = model_path / f"{model_name}.pth"

    if not config_file.is_file() or not model_file.is_file():
        raise FileNotFoundError(f"Pretrained model not found: {model_name}")

    model = build_model(config_file)
    model.load_state_dict(
        torch.load(model_file, map_location="cpu", weights_only=True)
    )
    return model
