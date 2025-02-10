import json
import logging
import os
from tools.marker_zone import MarkerZone
from tools.zone_intersection_tracker import ZoneIntersectionTracker
from tools.mqtt_subscriber import MqttSubscriber
from tools.mqtt_publisher import MqttPublisher
from tools.detection_log_processor import crop_object_from_frame, get_frame_from_timestamp, process_detection_log, save_cropped_image

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

    # Initialize MQTT Subscriber and Publisher
    mqtt_publisher = MqttPublisher(config)
    mqtt_subscriber = MqttSubscriber(config, mqtt_publisher)
    
    # Ask whether to display result images
    show_result = input("Do you want to display the result images? (y/n): ").lower() == 'y'
    
    # Ask whether to extract images from the detection log
    extract_image = input("Do you want to extract images from the detection log? (y/n): ").lower() == 'y'

    # 1. Run MaskTool to define zones on the selected frame.
    mask_tool = MarkerZone(config)
    mask_positions = mask_tool.run()

    # 2. Start tracking intersections.
    tracker = ZoneIntersectionTracker(config, show_result=show_result, extract_image=extract_image)
    tracker.track_intersections(config['rtsp_url'], config['frame_to_edit'])

    # 3. Process the detection log (which includes events: enter, exit, enter_movement,
    #    exit_movement, no_entry, no_parking, wrong_way) to extract event images.
    if extract_image:
        # Ask user which events to process (leave blank for all)
        event_input = input("Enter comma-separated events to process (or leave blank for all): ").strip()
        events_to_process = [e.strip() for e in event_input.split(',')] if event_input else None

        process_detection_log(
            detection_log_path=config['detection_log'],
            video_path=config['rtsp_url'],
            output_dir=config['output_dir'],
            get_frame_from_timestamp_func=get_frame_from_timestamp,
            crop_object_from_frame_func=crop_object_from_frame,
            save_cropped_image_func=save_cropped_image,
            events_to_process=events_to_process
        )