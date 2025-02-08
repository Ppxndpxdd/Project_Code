import json
import logging
import threading
import uuid
import ssl
import os
from typing import Dict, Any
from paho.mqtt import client as mqtt_client
from tools.mqtt_publisher import MqttPublisher

class MqttSubscriber:
    def __init__(self, config: Dict[str, Any], mqtt_publisher: MqttPublisher):
        self.lock = threading.Lock()
        self.emqx_host = config.get('mqtt_broker', 'localhost')
        self.emqx_port = config.get('mqtt_port', 8883)
        self.emqx_username = config.get('mqtt_username', '')
        self.emqx_password = config.get('mqtt_password', '')
        self.client_id = 'client-' + str(uuid.uuid4())
        self.mask_positions_file = config.get('mask_json_path', 'mask_positions.json')
        try:
            with open(self.mask_positions_file, 'r') as f:
                self.mask_positions = json.load(f)
        except Exception as e:
            logging.error(f"Error loading markers from {self.mask_positions_file}: {e}")
            self.mask_positions = []
        self.edgeDeviceId = config.get('edgeDeviceId', 'default_id')
        self.uniqueId = config.get('uniqueId', 'default_id')
        self.rule_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'rule.json')
        self.config = config
        self.mqtt_publisher = mqtt_publisher
        self.rule_applied_counter = 0 # Initialize a counter for ruleApplied IDs

        self.mqtt_client = mqtt_client.Client(client_id=self.client_id, protocol=mqtt_client.MQTTv311)
        self.mqtt_client.username_pw_set(self.emqx_username, self.emqx_password)
        self.configure_tls(config.get('ca_cert_path', 'emqxsl-ca.crt'))

        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message

        self.connect()

    def configure_tls(self, ca_cert_path):
        try:
            self.mqtt_client.tls_set(
                ca_certs=ca_cert_path,
                certfile=None,
                keyfile=None,
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLSv1_2,
                ciphers=None
            )
            self.mqtt_client.tls_insecure_set(False)
        except Exception as e:
            logging.error(f"Error configuring TLS: {e}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info("MQTT client connected successfully.")
            self.subscribe_to_topics()
        else:
            logging.error(f"MQTT client failed to connect. Return code: {rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            if msg.topic == f"{self.uniqueId}/marker/create":
                self.create_marker(payload)
            elif msg.topic == f"{self.uniqueId}/marker/update":
                self.update_marker(payload)
            elif msg.topic == f"{self.uniqueId}/marker/delete":
                self.delete_marker(payload)
            elif msg.topic == f"{self.uniqueId}/create":
                self.create_edge_device(payload)
            elif msg.topic == f"{self.uniqueId}/update":
                self.update_edge_device(payload)
            elif msg.topic == f"{self.uniqueId}/delete":
                self.delete_edge_device(payload)
            elif msg.topic == f"{self.uniqueId}/rule/create":
                self.create_rule_applied(payload)
            elif msg.topic == f"{self.uniqueId}/rule/update":
                self.update_rule_applied(payload)
            elif msg.topic == f"{self.uniqueId}/rule/delete":
                self.delete_rule_applied(payload)
        except Exception as e:
            logging.error(f"Error processing message from topic {msg.topic}: {e}")

    def connect(self):
        try:
            logging.info(f"Connecting to EMQX at {self.emqx_host}:{self.emqx_port} over TLS/SSL...")
            self.mqtt_client.connect(self.emqx_host, self.emqx_port)
            self.mqtt_client.loop_start()
        except Exception as e:
            logging.error(f"Could not connect to EMQX: {e}")

    def subscribe_to_topics(self):
        """Subscribes to MQTT topics specified in config.json under 'subscribe_topics'."""
        subscribe_topics = self.config.get('subscribe_topics', [])
        topics = [f"{self.uniqueId}/{topic}" for topic in subscribe_topics]
        for topic in topics:
            self.mqtt_client.subscribe(topic)
            logging.info(f"Subscribed to topic: '{topic}'")

    def save_mask_positions(self):
        """Saves the current in-memory mask_positions to file."""
        try:
            with open(self.mask_positions_file, 'w') as f:
                json.dump(self.mask_positions, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving markers to {self.mask_positions_file}: {e}")

    def notify_marker_update(self):
        """Notifies relevant components to reload markers in realtime."""
        # For example, if ZoneIntersectionTracker instance is stored:
        if hasattr(self, 'zone_tracker'):
            self.zone_tracker.load_zones()
            self.zone_tracker.load_arrows()
            self.zone_tracker.load_rule_config()

    def create_marker(self, data: Dict[str, Any]):
        try:
            # Append the new marker data to the in-memory list
            self.mask_positions.append(data)
            self.save_mask_positions()
            logging.info(f"Created marker position: {data}")
            self.publish_log("create complete")
            self.notify_marker_update()
        except Exception as e:
            logging.error(f"Error creating marker position: {e}")

    def update_marker(self, data: Dict[str, Any]):
        try:
            updated = False
            for i, position in enumerate(self.mask_positions):
                if position.get('marker_id') == data.get('marker_id'):
                    self.mask_positions[i] = data
                    updated = True
                    break
            if not updated:
                warning_msg = f"No marker found with marker_id {data.get('marker_id')}"
                logging.warning(warning_msg)
                self.publish_log(warning_msg)
                return
            self.save_mask_positions()
            logging.info(f"Updated marker position: {data}")
            self.publish_log("update complete")
            self.notify_marker_update()
        except Exception as e:
            logging.error(f"Error updating marker position: {e}")

    def delete_marker(self, data: Dict[str, Any]):
        try:
            # Delete based on a unique key, e.g. marker_id; adjust if needed
            marker_id = data.get('marker_id')
            new_positions = [
                position for position in self.mask_positions
                if position.get('marker_id') != marker_id
            ]
            if len(new_positions) == len(self.mask_positions):
                warning_msg = f"No marker found with marker_id {marker_id}"
                logging.warning(warning_msg)
                self.publish_log(warning_msg)
                return
            self.mask_positions = new_positions
            self.save_mask_positions()
            logging.info(f"Deleted marker position: {data}")
            self.publish_log("delete complete")
            self.notify_marker_update()
        except Exception as e:
            logging.error(f"Error deleting marker position: {e}")

    def update_edge_device(self, data: Dict[str, Any]):
        """Handles the updating of an existing edge device."""
        try:
            # Update the config with the new edgeDeviceId and rtspUrl
            if 'edgeDeviceId' in data:
                new_edgeDeviceId = data['edgeDeviceId']
                self.config['edgeDeviceId'] = new_edgeDeviceId
                # Notify MqttPublisher to update its edgeDeviceId as well
                # Update subscriber's edgeDeviceId and topics
                self.edge_id = new_edgeDeviceId
                if 'rtspUrl' in data:
                    self.config['rtspUrl'] = data['rtspUrl']

            # Save the updated config to the config.json file
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)

            logging.info(f"Updated edge device: {data}")
            self.publish_log("edge update complete")
        except Exception as e:
            logging.error(f"Error updating edge device: {e}")

    def delete_edge_device(self, data: Dict[str, Any]):
        """Handles the deletion of an edge device."""
        try:
            edge_device_id = data.get('edgeDeviceId')
            # Load config.json
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
            except Exception as e:
                logging.error(f"Error loading config.json: {e}")
                return

            # Remove the edge device from the config
            if 'edge_devices' in config_data and isinstance(config_data['edge_devices'], list):
                config_data['edge_devices'] = [
                    device for device in config_data['edge_devices']
                    if device.get('edgeDeviceId') != edge_device_id
                ]

                # Save the updated config.json
                try:
                    with open(self.config_file, 'w') as f:
                        json.dump(config_data, f, indent=4)
                    logging.info(f"Deleted edge device: {data}")
                    self.publish_log("edge delete complete")
                except Exception as e:
                    logging.error(f"Error saving config.json: {e}")
            else:
                warning_msg = f"No edge device found with edgeDeviceId {edge_device_id}"
                logging.warning(warning_msg)
                self.publish_log(warning_msg)

        except Exception as e:
            logging.error(f"Error deleting edge device: {e}")

    def create_rule_applied(self, data: Dict[str, Any]):
        """Handles the creation of a new rule applied to a marker and updates rule.json."""
        try:
            marker_id = data.get('markerId')
            rule_id = data.get('ruleId')
            json_params = data.get('jsonParams', {})

            if marker_id is None or rule_id is None:
                error_msg = "Payload must contain 'markerId' and 'ruleId'."
                logging.error(error_msg)
                self.publish_log(error_msg)
                return

            # Load rule.json
            try:
                with open(self.rule_config_file, 'r') as f:
                    rule_config = json.load(f)
            except Exception as e:
                logging.error(f"Error loading rule.json: {e}")
                return

            # Find the rule and add a new ruleApplied entry
            for rule in rule_config.get('rules', []):
                if rule.get('id') == rule_id:
                    self.rule_applied_counter += 1 # Increment the counter
                    new_rule_applied = {
                        "id": self.rule_applied_counter,  # Use the auto-incremented ID
                        "markerId": marker_id,
                        "jsonParams": json.dumps(json_params)
                    }
                    if 'ruleApplied' in rule:
                        rule['ruleApplied'].append(new_rule_applied)
                    else:
                        rule['ruleApplied'] = [new_rule_applied]

                    # Save the updated rule.json
                    try:
                        with open(self.rule_config_file, 'w') as f:
                            json.dump(rule_config, f, indent=4)
                        logging.info(f"Applied rule {rule_id} to marker {marker_id} and updated rule.json")
                        self.publish_log(f"Applied rule {rule_id} to marker {marker_id}")
                    except Exception as e:
                        logging.error(f"Error saving rule.json: {e}")
                    break
            else:
                warning_msg = f"No rule found with rule_id {rule_id}"
                logging.warning(warning_msg)
                self.publish_log(warning_msg)

            # Find the marker and update it
            for i, position in enumerate(self.mask_positions):
                if position.get('marker_id') == marker_id:
                    position['rule'] = rule_id  # Or rule name, depending on your needs
                    position['jsonParams'] = json_params
                    self.save_mask_positions()
                    self.notify_marker_update()
                    return

            warning_msg = f"No marker found with marker_id {marker_id}"
            logging.warning(warning_msg)
            self.publish_log(warning_msg)

        except Exception as e:
            logging.error(f"Error applying rule: {e}")

    def update_rule_applied(self, data: Dict[str, Any]):
        """Handles the updating of an existing rule applied to a marker and updates rule.json."""
        try:
            marker_id = data.get('markerId')
            rule_id = data.get('ruleId')
            json_params = data.get('jsonParams', {})
            applied_id = data.get('appliedId')

            if marker_id is None or rule_id is None or applied_id is None:
                error_msg = "Payload must contain 'markerId', 'ruleId', and 'appliedId'."
                logging.error(error_msg)
                self.publish_log(error_msg)
                return

            # Load rule.json
            try:
                with open(self.rule_config_file, 'r') as f:
                    rule_config = json.load(f)
            except Exception as e:
                logging.error(f"Error loading rule.json: {e}")
                return

            # Find the rule and update the ruleApplied entry
            for rule in rule_config.get('rules', []):
                if rule.get('id') == rule_id:
                    if 'ruleApplied' in rule:
                        for applied in rule['ruleApplied']:
                            if applied.get('id') == applied_id:
                                applied['jsonParams'] = json.dumps(json_params)

                                # Save the updated rule.json
                                try:
                                    with open(self.rule_config_file, 'w') as f:
                                        json.dump(rule_config, f, indent=4)
                                    logging.info(f"Updated rule {rule_id} for marker {marker_id} and updated rule.json")
                                    self.publish_log(f"Updated rule {rule_id} for marker {marker_id}")
                                except Exception as e:
                                    logging.error(f"Error saving rule.json: {e}")
                                break
                        else:
                            warning_msg = f"No ruleApplied found with appliedId {applied_id}"
                            logging.warning(warning_msg)
                            self.publish_log(warning_msg)
                            break
                    else:
                        warning_msg = f"No ruleApplied found for rule_id {rule_id}"
                        logging.warning(warning_msg)
                        self.publish_log(warning_msg)
                    break
            else:
                warning_msg = f"No rule found with rule_id {rule_id}"
                logging.warning(warning_msg)
                self.publish_log(warning_msg)

            # Find the marker and update it
            for i, position in enumerate(self.mask_positions):
                if position.get('marker_id') == marker_id:
                    position['rule'] = rule_id  # Or rule name, depending on your needs
                    position['jsonParams'] = json_params
                    self.save_mask_positions()
                    self.notify_marker_update()
                    return

            warning_msg = f"No marker found with marker_id {marker_id}"
            logging.warning(warning_msg)
            self.publish_log(warning_msg)

        except Exception as e:
            logging.error(f"Error updating rule: {e}")

    def delete_rule_applied(self, data: Dict[str, Any]):
        """Handles the deletion of a rule applied to a marker and updates rule.json."""
        try:
            marker_id = data.get('markerId')
            rule_id = data.get('ruleId')
            applied_id = data.get('appliedId')

            if marker_id is None or rule_id is None or applied_id is None:
                error_msg = "Payload must contain 'markerId', 'ruleId', and 'appliedId'."
                logging.error(error_msg)
                self.publish_log(error_msg)
                return

            # Load rule.json
            try:
                with open(self.rule_config_file, 'r') as f:
                    rule_config = json.load(f)
            except Exception as e:
                logging.error(f"Error loading rule.json: {e}")
                return

            # Find the rule and delete the ruleApplied entry
            for rule in rule_config.get('rules', []):
                if rule.get('id') == rule_id:
                    if 'ruleApplied' in rule:
                        rule['ruleApplied'] = [
                            applied for applied in rule['ruleApplied']
                            if applied.get('id') != applied_id
                        ]

                        # Save the updated rule.json
                        try:
                            with open(self.rule_config_file, 'w') as f:
                                json.dump(rule_config, f, indent=4)
                            logging.info(f"Deleted rule {rule_id} for marker {marker_id} and updated rule.json")
                            self.publish_log(f"Deleted rule {rule_id} for marker {marker_id}")
                        except Exception as e:
                            logging.error(f"Error saving rule.json: {e}")
                        break
                    else:
                        warning_msg = f"No ruleApplied found for rule_id {rule_id}"
                        logging.warning(warning_msg)
                        self.publish_log(warning_msg)
                    break
            else:
                warning_msg = f"No rule found with rule_id {rule_id}"
                logging.warning(warning_msg)
                self.publish_log(warning_msg)

            # Find the marker and update it
            for i, position in enumerate(self.mask_positions):
                if position.get('marker_id') == marker_id:
                    position.pop('rule', None)
                    position.pop('jsonParams', None)
                    self.save_mask_positions()
                    self.notify_marker_update()
                    return

            warning_msg = f"No marker found with marker_id {marker_id}"
            logging.warning(warning_msg)
            self.publish_log(warning_msg)

        except Exception as e:
            logging.error(f"Error deleting rule: {e}")

    def publish_log(self, message: str):
        """Publishes a log message to the 'marker_positions/log' topic."""
        marker_log_topic = f"{self.config.get('uniqueId')}/{self.config.get('marker_log_topic', 'marker/log')}"
        try:
            self.mqtt_publisher.publish(marker_log_topic, message)
        except Exception as e:
            logging.error(f"Error publishing log message: {e}")

    def disconnect(self):
        """Disconnects the MQTT client."""
        try:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            logging.info("MQTT client disconnected successfully.")
        except Exception as e:
            logging.error(f"Error disconnecting MQTT client: {e}")