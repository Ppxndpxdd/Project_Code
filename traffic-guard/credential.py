import os
import json
import random
import string
import logging

def get_edge_id(length=4):
    characters = string.ascii_letters + string.digits
    edge_id = ''.join(random.choices(characters, k=length))
    return edge_id

def save_edge_id_to_config(config_path, edge_id):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['edge_id'] = edge_id
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config_path = 'traffic-guard/config/config.json'
    edge_id = get_edge_id()
    save_edge_id_to_config(config_path, edge_id)