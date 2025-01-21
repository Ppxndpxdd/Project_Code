from dataclasses import asdict
import json
import logging
import threading
import time
from .mqtt_handler import MqttHandler

class MqttPublisher:
    def __init__(self, config):
        self.mqtt_handler = MqttHandler(config)
        self.heartbeat_interval = config.get('heartbeat_interval', 60)  # in seconds
        self.heartbeat_topic = config.get('heartbeat_topic', 'heartbeat_log')
        self.unique_id = config.get('unique_id')
        if not self.unique_id or self.unique_id == 'default_id':
            logging.error("Invalid unique_id provided in config.")
            raise ValueError("unique_id must be set in the configuration.")
        # Start the heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        self.heartbeat_thread.start()

    def send_incident(self, detection_entry):
        if not isinstance(detection_entry, dict):
            try:
                payload = asdict(detection_entry)
            except Exception as e:
                logging.error(f"Failed to convert detection_entry to dict: {e}")
                payload = {}
        else:
            payload = detection_entry

        if not payload:
            logging.error("Payload is empty. Incident not sent.")
            return
        topic = f"{self.unique_id}/incident"
        logging.debug(f"Sending incident payload: {payload}")
        try:
            self.mqtt_handler.publish(topic, json.dumps(payload))
        except Exception as e:
            logging.error(f"Failed to send incident: {e}")

    def send_heartbeat(self):
        """Sends a heartbeat message to the MQTT broker at regular intervals."""
        while True:
            try:
                heartbeat_message = {"status": "alive"}
                topic = f"{self.unique_id}/heartbeat"
                logging.info(f"Sending heartbeat to topic {self.heartbeat_topic}")
                self.mqtt_handler.publish(topic, json.dumps(heartbeat_message))
                logging.info("Heartbeat sent successfully.")
            except Exception as e:
                logging.error(f"Failed to send heartbeat: {e}")
            time.sleep(30)  # Send heartbeat every one second