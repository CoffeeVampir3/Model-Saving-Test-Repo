from safetensors import safe_open
from safetensors.torch import load_model
import subprocess
import json
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        return self.fc2(self.fc1(x))

def get_git_tag():
    return subprocess.check_output(['git', 'describe', '--tags', '--abbrev=0']).decode().strip()

with safe_open("model.safetensors", framework="pt") as f:
    metadata = f.metadata()

config_dict = {k: json.loads(v) if not isinstance(v, str) else v for k, v in metadata.items()}
print(f"Model version: {config_dict['version']}")
print(f"Git tag: {config_dict['git_tag']}")

model = MLP()
load_model(model, "model.safetensors")
