from dataclasses import asdict
import json
import logging
import threading
import time
from .mqtt_handler import MqttHandler

class MqttPublisher:
    def __init__(self, config):
        self.lock = threading.Lock()
        self.mqtt_handler = MqttHandler(config)
        self.heartbeat_interval = config.get('heartbeat_interval', 30)  # in seconds
        self.edge_id = config.get('edge_id')
        if not self.edge_id or self.edge_id == 'default_id':
            logging.error("Invalid edge_id provided in config.")
            raise ValueError("edge_id must be set in the configuration.")
        
        # Start the heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        self.heartbeat_thread.start()

    def update_edge_id(self, new_edge_id: str):
        """Updates the edge_id and notifies the subscriber."""
        with self.lock:  # Ensure thread safety
            self.edge_id = new_edge_id
            logging.info(f"Updated edge_id to: {self.edge_id}")

            # Notify the MQTT subscriber to update its edge_id
            if hasattr(self, 'mqtt_subscriber'):
                self.mqtt_subscriber.update_edge_id(new_edge_id)

    def get_incident_topic(self):
        """Generates the incident topic dynamically using the current edge_id."""
        return f"{self.edge_id}/incident"

    def get_heartbeat_topic(self):
        """Generates the heartbeat topic dynamically using the current edge_id."""
        return f"{self.edge_id}/heartbeat"

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
        
        # Generate the incident topic dynamically
        topic = self.get_incident_topic()
        logging.debug(f"Sending incident payload: {payload} to topic: {topic}")
        try:
            self.mqtt_handler.publish(topic, json.dumps(payload))
        except Exception as e:
            logging.error(f"Failed to send incident: {e}")

    def send_heartbeat(self):
        """Sends a heartbeat message to the MQTT broker at regular intervals."""
        while True:
            try:
                heartbeat_message = f"{self.edge_id}"
                # Generate the heartbeat topic dynamically
                topic = self.get_heartbeat_topic()
                logging.info(f"Sending heartbeat to topic: {topic}")
                self.mqtt_handler.publish(topic, json.dumps(heartbeat_message))
                logging.info("Heartbeat sent successfully.")
            except Exception as e:
                logging.error(f"Failed to send heartbeat: {e}")
            time.sleep(30)  # Send heartbeat every 30 seconds