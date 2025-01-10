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

        # Start the heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        self.heartbeat_thread.start()

    def send_incident(self, detection_entry):
        """Sends incident info to the MQTT broker."""
        try:
            logging.info(f"Sending incident: {detection_entry}")
            self.mqtt_handler.publish_detection(detection_entry.__dict__)
            logging.info("Incident sent successfully.")
        except Exception as e:
            logging.error(f"Failed to send incident: {e}")

    def send_heartbeat(self):
        """Sends a heartbeat message to the MQTT broker at regular intervals."""
        while True:
            try:
                heartbeat_message = {"status": "alive"}
                logging.info(f"Sending heartbeat to topic {self.heartbeat_topic}")
                self.mqtt_handler.publish(self.heartbeat_topic, json.dumps(heartbeat_message))
                logging.info("Heartbeat sent successfully.")
            except Exception as e:
                logging.error(f"Failed to send heartbeat: {e}")
            time.sleep(30)  # Send heartbeat every one second