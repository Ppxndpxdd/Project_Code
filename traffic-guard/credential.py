import uuid
import os
import json

def get_unique_id():
    # Generate a unique ID using the MAC address or a UUID
    unique_id = uuid.uuid4().hex
    return unique_id

def save_unique_id_to_config(config_path, unique_id):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['unique_id'] = unique_id
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

# Generate and save the unique ID
config_path = 'traffic-guard/config/config.json'
unique_id = get_unique_id()
save_unique_id_to_config(config_path, unique_id)