import subprocess
import json
from safetensors.torch import save_model

def get_git_tag():
    return subprocess.check_output(['git', 'describe', '--tags', '--always']).decode().strip()

config_dict = {
    'version': '1.0',
    'git_tag': get_git_tag()
}
metadata = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in config_dict.items()}

save_model(model, "model.safetensors", metadata=metadata)
