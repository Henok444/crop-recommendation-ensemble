from pathlib import Path
import yaml

project_root = Path(__file__).resolve().parents[2]
config_path = project_root / "config" / "config.yaml"

def load_config():
    with open(config_path , "r" ) as file:
        return yaml.safe_load(file)

