import os
import json
import random
import string
import logging

def get_unique_id(length=6):
    characters = string.ascii_letters + string.digits
    unique_id = ''.join(random.choices(characters, k=length))
    return unique_id

def save_unique_id_to_config(config_path, unique_id):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['unique_id'] = unique_id
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config_path = 'traffic-guard/config/config.json'
    unique_id = get_unique_id()
    save_unique_id_to_config(config_path, unique_id)