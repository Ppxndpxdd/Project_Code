import json
import logging
import os
from tools.marker_zone import MarkerZone
from tools.zone_intersection_tracker import ZoneIntersectionTracker
from tools.mqtt_subscriber import MqttSubscriber
from tools.mqtt_publisher import MqttPublisher

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load configuration from config.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config', 'config.json')
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        exit()
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in configuration file: {config_path}")
        exit()

    # Ask the user whether to display images and set show_result
    show_result_input = input("Do you want to display the result images? (y/n): ").lower()
    show_result = show_result_input == 'y'

    # Ask the user whether to extract images from the detection log
    extract_image_input = input("Do you want to extract images from the detection log? (y/n): ").lower()
    extract_image = extract_image_input == 'y'

    # Initialize MQTT Subscriber and Publisher
    mqtt_publisher = MqttPublisher(config)
    logging.info(f"Config being passed to MqttPublisher: {config}") # Add this line
    mqtt_subscriber = MqttSubscriber(config, mqtt_publisher)
    zone_tracker = ZoneIntersectionTracker(config, show_result=show_result, extract_image=extract_image)
    mqtt_subscriber.zone_tracker = zone_tracker

    # 1. Run MaskTool to define zones on the selected frame
    mask_tool = MarkerZone(config)
    mask_positions = mask_tool.run()

    # 2. Start tracking intersections
    tracker = ZoneIntersectionTracker(config, show_result=show_result, extract_image=extract_image)
    tracker.track_intersections(config['video_source'], config['frame_to_edit'])