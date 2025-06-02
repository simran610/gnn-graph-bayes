import yaml
import os

def load_config(filename: str) -> dict:
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    # Construct the full path to the YAML file
    file_path = os.path.join(script_dir, filename)

    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config