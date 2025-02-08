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
        self.zone_tracker = None
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
            marker_id = data.get('marker_id')
            # Remove marker from mask_positions
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
            
            # Remove applied rule entries associated with marker_id from rule.json
            try:
                with open(self.rule_config_file, 'r') as f:
                    rule_config = json.load(f)
                for rule in rule_config.get('rules', []):
                    if 'ruleApplied' in rule:
                        rule['ruleApplied'] = [
                            applied for applied in rule['ruleApplied']
                            if applied.get('markerId') != marker_id
                        ]
                with open(self.rule_config_file, 'w') as f:
                    json.dump(rule_config, f, indent=4)
                logging.info(f"Deleted applied rule entries for marker_id {marker_id} in rule.json")
            except Exception as e:
                logging.error(f"Error updating rule.json when deleting marker {marker_id}: {e}")
            
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

    def validate_rule_json_params(self, rule_id: int, provided_params: dict) -> bool:
        try:
            with open(self.rule_config_file, 'r') as f:
                rule_config = json.load(f)
        except Exception as e:
            logging.error(f"Error loading rule.json: {e}")
            self.publish_log("Error loading rule configuration.")
            return False

        rule = next((r for r in rule_config.get('rules', []) if r.get('id') == rule_id), None)
        if not rule:
            self.publish_log(f"No rule found with rule_id {rule_id}.")
            return False

        try:
            allowed_params = json.loads(rule.get("jsonParams", "{}"))
            if not isinstance(allowed_params, dict):
                logging.error(f"Allowed parameters for rule_id {rule_id} should be a JSON object.")
                return False
        except Exception as e:
            logging.error(f"Error parsing rule jsonParams for rule_id {rule_id}: {e}")
            return False

        for key in provided_params.keys():
            if key not in allowed_params:
                self.publish_log(
                    f"Invalid json parameter '{key}' for rule_id {rule_id}. Allowed keys: {list(allowed_params.keys())}"
                )
                return False

        return True

    def validate_create_payload(self, data: dict) -> bool:
        if 'markerId' not in data or 'ruleId' not in data:
            self.publish_log("Payload missing required fields 'markerId' and/or 'ruleId' for create_rule_applied.")
            return False
        if 'jsonParams' in data and not isinstance(data['jsonParams'], dict):
            self.publish_log("Field 'jsonParams' must be a JSON object for create_rule_applied.")
            return False
        if not self.validate_rule_json_params(data['ruleId'], data.get('jsonParams', {})):
            return False
        return True

    def validate_update_payload(self, data: dict) -> bool:
        if 'markerId' not in data or 'ruleId' not in data or 'appliedId' not in data:
            self.publish_log("Payload missing one or more required fields ('markerId', 'ruleId', 'appliedId') for update_rule_applied.")
            return False
        if 'jsonParams' in data and not isinstance(data['jsonParams'], dict):
            self.publish_log("Field 'jsonParams' must be a JSON object for update_rule_applied.")
            return False
        if not self.validate_rule_json_params(data['ruleId'], data.get('jsonParams', {})):
            return False
        return True

    def validate_delete_payload(self, data: dict) -> bool:
        if 'markerId' not in data or 'ruleId' not in data or 'appliedId' not in data:
            self.publish_log("Payload missing one or more required fields ('markerId', 'ruleId', 'appliedId') for delete_rule_applied.")
            return False
        return True

    def reset_rule_applied_counter(self):
        """Recalculates the rule_applied_counter from rule.json.
           If no applied rule entries exist, resets counter to 0.
        """
        max_applied = 0
        try:
            with open(self.rule_config_file, 'r') as f:
                rule_config = json.load(f)
            for rule in rule_config.get('rules', []):
                for applied in rule.get('ruleApplied', []):
                    applied_id = applied.get('id', 0)
                    if applied_id > max_applied:
                        max_applied = applied_id
        except Exception as e:
            logging.error(f"Error resetting rule_applied_counter: {e}")
        self.rule_applied_counter = max_applied
        logging.info(f"rule_applied_counter reset to {self.rule_applied_counter}")

    def create_rule_applied(self, data: dict):
        """Handles the creation of a new rule applied to a marker and updates rule.json."""
        try:
            if not self.validate_create_payload(data):
                return

            marker_id = data.get('markerId')
            rule_id = data.get('ruleId')
            json_params = data.get('jsonParams', {})

            # Reset counter based on rule.json contents.
            self.reset_rule_applied_counter()

            # Load rule.json
            with open(self.rule_config_file, 'r') as f:
                rule_config = json.load(f)

            matched_rule_name = None
            for rule in rule_config.get('rules', []):
                if rule.get('id') == rule_id:
                    matched_rule_name = rule.get('name')
                    self.rule_applied_counter += 1
                    new_rule_applied = {
                        "id": self.rule_applied_counter,
                        "markerId": marker_id,
                        "jsonParams": json.dumps(json_params)
                    }
                    if 'ruleApplied' in rule:
                        rule['ruleApplied'].append(new_rule_applied)
                    else:
                        rule['ruleApplied'] = [new_rule_applied]
                    with open(self.rule_config_file, 'w') as f:
                        json.dump(rule_config, f, indent=4)
                    logging.info(f"Applied rule {rule_id} to marker {marker_id} and updated rule.json")
                    self.publish_log(f"Applied rule {rule_id} to marker {marker_id}")
                    break
            else:
                warning_msg = f"No rule found with rule_id {rule_id}"
                logging.warning(warning_msg)
                self.publish_log(warning_msg)
                return

            # Update marker similar to create_marker logic.
            marker_found = False
            for i, position in enumerate(self.mask_positions):
                if position.get('marker_id') == marker_id:
                    self.mask_positions[i]['rule'] = matched_rule_name
                    self.mask_positions[i]['jsonParams'] = json_params
                    self.mask_positions[i]['applied_id'] = self.rule_applied_counter
                    marker_found = True
                    break

            if marker_found:
                self.save_mask_positions()
                logging.debug("Marker updated after creating applied rule. Triggering realtime refresh.")
                self.notify_marker_update()  # Refresh visualization (including text overlays)
            else:
                warning_msg = f"No marker found with marker_id {marker_id}"
                logging.warning(warning_msg)
                self.publish_log(warning_msg)
        except Exception as e:
            logging.error(f"Error applying rule: {e}")

    def update_rule_applied(self, data: dict):
        """Handles the updating of an existing rule applied to a marker and updates rule.json."""
        try:
            if not self.validate_update_payload(data):
                return

            marker_id = data.get('markerId')
            rule_id = data.get('ruleId')
            json_params = data.get('jsonParams', {})
            try:
                applied_id = int(data.get('appliedId'))  # Ensure appliedId is an integer.
            except Exception as e:
                logging.error("Invalid appliedId provided: %s", data.get('appliedId'))
                return

            with open(self.rule_config_file, 'r') as f:
                rule_config = json.load(f)

            matched_rule_name = None
            rule_found = False
            for rule in rule_config.get('rules', []):
                if rule.get('id') == rule_id:
                    matched_rule_name = rule.get('name')
                    rule_found = True
                    if 'ruleApplied' in rule:
                        applied_found = False
                        for applied in rule['ruleApplied']:
                            # Convert applied id to int for consistency.
                            try:
                                applied_rule_id = int(applied.get('id'))
                            except Exception:
                                applied_rule_id = applied.get('id')
                            if applied_rule_id == applied_id:
                                applied['jsonParams'] = json.dumps(json_params)
                                applied_found = True
                                break
                        if not applied_found:
                            warning_msg = f"No applied rule found with appliedId {applied_id}"
                            logging.warning(warning_msg)
                            self.publish_log(warning_msg)
                            return
                    else:
                        warning_msg = f"No applied rules exist for rule_id {rule_id}"
                        logging.warning(warning_msg)
                        self.publish_log(warning_msg)
                        return
                    break

            if not rule_found:
                warning_msg = f"No rule found with rule_id {rule_id}"
                logging.warning(warning_msg)
                self.publish_log(warning_msg)
                return

            with open(self.rule_config_file, 'w') as f:
                json.dump(rule_config, f, indent=4)
            logging.info(f"Updated rule {rule_id} for marker {marker_id}")
            self.publish_log(f"Updated rule {rule_id} for marker {marker_id}")

            marker_found = False
            for i, position in enumerate(self.mask_positions):
                if position.get('marker_id') == marker_id:
                    self.mask_positions[i]['rule'] = matched_rule_name
                    self.mask_positions[i]['jsonParams'] = json_params
                    self.mask_positions[i]['applied_id'] = applied_id
                    marker_found = True
                    break

            if marker_found:
                self.save_mask_positions()
                logging.debug("Marker updated after updating applied rule. Triggering realtime refresh.")
                self.notify_marker_update()
            else:
                warning_msg = f"No marker found with marker_id {marker_id}"
                logging.warning(warning_msg)
                self.publish_log(warning_msg)
        except Exception as e:
            logging.error(f"Error updating rule: {e}")

    def delete_rule_applied(self, data: dict):
        """Handles the deletion of a rule applied to a marker and updates rule.json."""
        try:
            if not self.validate_delete_payload(data):
                return

            marker_id = data.get('markerId')
            rule_id = data.get('ruleId')
            applied_id = data.get('appliedId')

            # Load rule.json
            try:
                with open(self.rule_config_file, 'r') as f:
                    rule_config = json.load(f)
            except Exception as e:
                logging.error(f"Error loading rule.json: {e}")
                self.publish_log("Error loading rule configuration.")
                return

            for rule in rule_config.get('rules', []):
                if rule.get('id') == rule_id:
                    if 'ruleApplied' in rule:
                        rule['ruleApplied'] = [
                            applied for applied in rule['ruleApplied']
                            if applied.get('id') != applied_id
                        ]
                        try:
                            with open(self.rule_config_file, 'w') as f:
                                json.dump(rule_config, f, indent=4)
                            logging.info(f"Deleted rule {rule_id} for marker {marker_id}")
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
                return

            # Update marker info in mask_positions by removing rule, jsonParams, and applied_id
            for i, position in enumerate(self.mask_positions):
                if position.get('marker_id') == data.get('markerId'):
                    position.pop('rule', None)
                    position.pop('jsonParams', None)
                    position.pop('applied_id', None)
                    self.save_mask_positions()
                    # DEBUG: log marker update before notifying
                    logging.debug(f"Deleted applied rule; updating mask_positions for marker {data.get('markerId')}.")
                    self.notify_marker_update()  # Trigger downstream refresh
                    return

            warning_msg = f"No marker found with marker_id {marker_id}"
            logging.warning(warning_msg)
            self.publish_log(warning_msg)

        except Exception as e:
            logging.error(f"Error deleting rule: {e}")

    def notify_marker_update(self):
        logging.debug("notify_marker_update() called. Mask positions: %s", self.mask_positions)
        if self.zone_tracker:
            try:
                logging.debug("Re-loading the full rule configuration from rule.json.")
                # Reload rule.json in zone tracker so that latest ruleApplied entries are used.
                if hasattr(self.zone_tracker, "load_rule_config"):
                    self.zone_tracker.rule_config = self.zone_tracker.load_rule_config()
                    logging.debug("Zone tracker rule configuration reloaded.")
                self.zone_tracker.load_zones()
                self.zone_tracker.load_arrows()
                logging.info("Zone tracker realtime update triggered successfully.")
            except Exception as ex:
                logging.error("Error during zone_tracker update: %s", ex)
        else:
            logging.warning("No zone_tracker defined in MqttSubscriber; realtime update may not occur.")

    def publish_log(self, message: str):
        """Publishes a log message to the 'marker_positions/log' topic."""
        marker_log_topic = f"{self.config.get('uniqueId')}/{self.config.get('log_topic', 'log')}"
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