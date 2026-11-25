import torch.nn as nn
import torch.optim as optim
import torch
import yaml
from pathlib import Path
import pathlib
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Task2Model(nn.Module):
    def __init__(self, dims=[512, 128, 7], dropouts=[0.5], activation=nn.LeakyReLU(), normalize=False, dtype=torch.float32):
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
            model_type = Task2Model
            dims = model_config['model']['dims']
            dropouts = model_config['model']['dropouts']
            activation = nn.LeakyReLU()
            normalize = model_config['model']['normalize']
            dtype = torch.float32
        except KeyError:
            print("model_config doesn't have the correct structure!")
            return None
        except TypeError:
            print("model_config doesn't have the correct structure!")
            return None
    elif isinstance(model_config, (str, Path)):
        with open(model_config, 'r') as f:
            config = yaml.safe_load(f)
            model_type = Task2Model
            dims = config['model']['dims']
            dropouts = config['model']['dropouts']
            activation = nn.LeakyReLU()
            normalize = config['model']['normalize']
            dtype = torch.float32
    else:
        try:
            model_type = Task2Model
            dims = model_config.dims
            dropouts = model_config.dropouts
            activation = model_config.activation
            normalize = model_config.normalize
            dtype = torch.float32
        except KeyError:
            print("model_config doesn't have the correct structure!")
            return None
        except TypeError:
            print("model_config doesn't have the correct structure!")
            return None

    return model_type(dims=dims, dropouts=dropouts, activation=activation, normalize=normalize, dtype=dtype).to(device=device)

def build_model_from_pretrained(model_name):
    model_path = (Path(__file__).parent / Path(f'../results/models/{model_name}')).resolve()
    config_file = model_path / f'{model_name}_config.yaml'
    if model_path.exists():
        model = build_model(config_file)
    else:
        raise Exception("A model with that name doesn't exist.")
    model_file = model_path / f'{model_name}.pth'
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    return model
