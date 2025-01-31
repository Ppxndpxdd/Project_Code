import json
import logging
from tools.marker_zone import MarkerZone
from tools.zone_intersection_tracker import ZoneIntersectionTracker
from tools.mqtt_subscriber import MqttSubscriber
from tools.mqtt_publisher import MqttPublisher

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)

    # Load configuration from configs.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config', 'config.json')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Initialize MQTT Subscriber
    mqtt_publisher = MqttPublisher(config)
    mqtt_subscriber = MqttSubscriber(config, mqtt_publisher)

    # 1. Run MaskTool to define zones on the selected frame
    mask_tool = MarkerZone(config)
    mask_positions = mask_tool.run()

    # 2. Ask the user whether to display images
    show_result_input = input("Do you want to display the result images? (y/n): ").lower()
    show_result = show_result_input == 'y'

    # 3. Initialize ZoneIntersectionTracker with the configuration and show_result parameter
    tracker = ZoneIntersectionTracker(config, show_result=show_result)
    tracker.track_intersections(config['video_source'], config['frame_to_edit'])