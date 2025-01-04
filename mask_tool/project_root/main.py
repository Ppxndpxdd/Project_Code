import os
import json
from src.mask_tool import MaskTool
from src.zone_tracker import ZoneIntersectionTracker

# Usage
if __name__ == "__main__":
    # Load configuration from config.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config', 'config.json')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # 1. Run MaskTool to define zones on the selected frame
    mask_tool = MaskTool(config)
    mask_positions = mask_tool.run()

    # 3. Ask the user whether to display images
    show_result_input = input("Do you want to display the result images? (y/n): ").lower()
    show_result = show_result_input == 'y'

    # 4. Initialize ZoneIntersectionTracker with the configuration and show_result parameter
    tracker = ZoneIntersectionTracker(config, show_result=show_result)
    tracker.track_intersections(config['video_source'], config['frame_to_edit'])