import os
import json
import random
import string
import logging

def get_uniqueId(length=4):
    characters = string.ascii_letters + string.digits
    uniqueId = ''.join(random.choices(characters, k=length))
    return uniqueId

def save_uniqueId_to_config(config_path, uniqueId):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['uniqueId'] = uniqueId
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config_path = 'traffic-guard/config/config.json'
    uniqueId = get_uniqueId()
    save_uniqueId_to_config(config_path, uniqueId)