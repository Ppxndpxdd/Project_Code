from dataclasses import asdict
import json
import logging
import threading
import time
from .mqtt_handler import MqttHandler

class MqttPublisher:
    def __init__(self, config):
        self.lock = threading.Lock()
        self.config = config
        self.mqtt_handler = MqttHandler(config)
        # Get heartbeat and incident topics from config
        self.unique_id = config.get('unique_id', 'default_id')
        self.edge_device_id = config.get('edge_device_id', 'default_id') 
        self.incident_topic = f"{self.config.get('detection_topic')}/{self.unique_id}"
        self.heartbeat_topic =  f"{config.get('heartbeat_topic', 'keep_alive')}/{self.unique_id}" # Add unique_id to heartbeat topic
        self.heartbeat_interval = config.get('heartbeat_interval', 30)
        self.log_topic = config.get('log_topic', 'log')
        
        # Start the heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        self.heartbeat_thread.start()

    def get_incident_topic(self):
        """Retrieve the incident topic name from config.json."""
        return self.incident_topic

    def get_heartbeat_topic(self):
        """Retrieve the heartbeat topic name from config.json."""
        return self.heartbeat_topic

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

        # Ensure id_rule_applied gets added if ruleApplied exists and is a non-empty list
        rule_applied = payload.get("rule_applied")
        if isinstance(rule_applied, list) and rule_applied:
            payload["id_rule_applied"] = rule_applied[0].get("id")
        else:
            logging.debug("No valid ruleApplied found; id_rule_applied not added.")

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
                # Reload edge_device_id in real-time
                self.edge_device_id = self.config.get('edge_device_id', 'default_id')
                heartbeat_message = f"{self.edge_device_id}"
                topic = self.get_heartbeat_topic()
                logging.info(f"Sending heartbeat to topic: {topic}")
                self.mqtt_handler.publish(topic, json.dumps(heartbeat_message))
                logging.info("Heartbeat sent successfully.")
            except Exception as e:
                logging.error(f"Failed to send heartbeat: {e}")
            # use the configured heartbeat_interval instead of hard-coded 30 sec
            time.sleep(self.heartbeat_interval)
            
    def publish_log(self, payload):
        """Publishes a message to the MQTT broker."""
        try:
            topic = f"{self.unique_id}/{self.log_topic}"
            self.mqtt_handler.publish(topic, payload)
        except Exception as e:
            logging.error(f"Failed to publish message: {e}")