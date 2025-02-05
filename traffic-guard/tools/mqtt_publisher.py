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
        self.uniqueId = config.get('uniqueId', 'default_id')
        
        # Start the heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        self.heartbeat_thread.start()

    def get_incident_topic(self):
        """Generates the incident topic dynamically using the current edgeDeviceId."""
        return f"detection_log/{self.uniqueId}"

    def get_heartbeat_topic(self):
        """Generates the heartbeat topic dynamically using the current edgeDeviceId."""
        return f"keep_alive/{self.uniqueId}"

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
        rule_applied = payload.get("ruleApplied")
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
                heartbeat_message = f"{self.uniqueId}"
                # Generate the heartbeat topic dynamically
                topic = self.get_heartbeat_topic()
                logging.info(f"Sending heartbeat to topic: {topic}")
                self.mqtt_handler.publish(topic, json.dumps(heartbeat_message))
                logging.info("Heartbeat sent successfully.")
            except Exception as e:
                logging.error(f"Failed to send heartbeat: {e}")
            time.sleep(30)  # Send heartbeat every 30 seconds