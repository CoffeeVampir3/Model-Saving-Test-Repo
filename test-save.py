import subprocess
import json
from safetensors.torch import save_model
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

model = MLP()
config_dict = {
    'version': '1.0',
    'git_tag': get_git_tag()
}
metadata = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in config_dict.items()}

save_model(model, "model.safetensors", metadata=metadata)
print(f"Model saved with version: {config_dict['version']}")
print(f"Model saved with tag: {config_dict['git_tag']}")
