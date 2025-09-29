from safetensors import safe_open
from safetensors.torch import load_model
import subprocess
import json

def get_git_tag():
    return subprocess.check_output(['git', 'describe', '--tags', '--always']).decode().strip()

with safe_open("model.safetensors", framework="pt") as f:
    metadata = f.metadata()

config_dict = {k: json.loads(v) if not isinstance(v, str) else v for k, v in metadata.items()}
print(f"Model version: {config_dict['version']}")
print(f"Git tag: {config_dict['git_tag']}")

load_model(model, "model.safetensors")
