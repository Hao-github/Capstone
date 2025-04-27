import yaml
import os
from typing import Any

def load_config(config_file_path: str = "config.yaml") -> dict[str, Any]:
    try:
        config_file_path = os.path.join(os.path.dirname(__file__), config_file_path)
        with open(config_file_path, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_file_path}")
        return {}