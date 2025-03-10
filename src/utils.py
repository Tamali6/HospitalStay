import os
import yaml

# Loads configuration settings from a YAML file
def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise Exception(f"Error loading YAML file: {exc}")
    return config

