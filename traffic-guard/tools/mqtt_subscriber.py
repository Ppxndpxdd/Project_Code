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
        self.edge_device_id = config.get('edge_device_id', 'default_id')
        self.unique_id = config.get('unique_id', 'default_id')
        self.rule_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'rule.json')
        self.mqtt_publisher = mqtt_publisher
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config.json')
        self.config = config

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
            topic = msg.topic
            unique_id = self.unique_id  # Capture self.unique_id to avoid repeated access
    
            if topic == f"{unique_id}/marker/create":
                self.create_marker(payload)
            elif topic == f"{unique_id}/marker/update":
                self.update_marker(payload)
            elif topic == f"{unique_id}/marker/delete":
                self.delete_marker(payload)
            elif topic == f"{unique_id}/create":
                self.create_edge_device(payload)
            elif topic == f"{unique_id}/update":
                self.update_edge_device(payload)
            elif topic == f"{unique_id}/delete":
                self.delete_edge_device(payload)
            elif topic == f"{unique_id}/rule/create":
                self.create_rule_applied(payload)
            elif topic == f"{unique_id}/rule/update":
                self.update_rule_applied(payload)
            elif topic == f"{unique_id}/rule/delete":
                self.delete_rule_applied(payload)
            else:
                logging.warning(f"Received message on unknown topic: {topic}")
    
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
        topics = [f"{self.unique_id}/{topic}" for topic in subscribe_topics]
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
                    if 'rule_applied' in rule:
                        rule['rule_applied'] = [
                            applied for applied in rule['rule_applied']
                            if applied.get('marker_id') != marker_id
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

    def create_edge_device(self, data: Dict[str, Any]):
        """Handles the creation of a new edge device."""
        try:
            # Update the config with the new edge_device_id and rtsp_url
            new_edge_device_id = data.get('edge_device_id', self.config['edge_device_id'])
            self.config['edge_device_id'] = new_edge_device_id
            self.config['rtsp_url'] = data.get('rtsp_url', self.config.get('rtsp_url', ''))
            
            # Save the updated config to the config.json file
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            logging.info(f"Created edge device: {data}")
            self.publish_log("edge create complete")

        except Exception as e:
            logging.error(f"Error creating edge device: {e}")

    def update_edge_device(self, data: Dict[str, Any]):
        """Handles the updating of an existing edge device."""
        try:
            # Update the config with the new edge_device_id and rtsp_url
            if 'edge_device_id' in data:
                new_edge_device_id = data['edge_device_id']
                self.config['edge_device_id'] = new_edge_device_id
                # Notify MqttPublisher to update its edge_device_id as well
                # Update subscriber's edge_device_id and topics
                self.edge_id = new_edge_device_id            
                if 'rtsp_url' in data:
                    self.config['rtsp_url'] = data['rtsp_url']
            
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
            # Reset the edge_device_id and rtsp_url to default values
            self.config['edge_device_id'] = ''
            self.config['rtsp_url'] = ''
            
            # Save the updated config to the config.json file
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            logging.info(f"Deleted edge device: {data}")
            self.publish_log("edge delete complete")
        except Exception as e:
            logging.error(f"Error deleting edge device: {e}")

    def validate_rule_json_params(self, rule_id: int, provided_params: dict) -> bool:
        """Validates the json parameters against the rule configuration."""
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
    
        # Use rule.get("json_params") to access the json_params
        json_params_str = rule.get("json_params")
        if not json_params_str:
            logging.warning(f"Rule {rule_id} has no 'jsonParams' defined. All parameters will be rejected.")
            return len(provided_params) == 0
    
        try:
            allowed_params = json.loads(json_params_str)
            if not isinstance(allowed_params, dict):
                logging.error(f"Allowed parameters for rule_id {rule_id} should be a JSON object.")
                return False
        except (TypeError, json.JSONDecodeError) as e:
            logging.error(f"Error parsing jsonParams for rule_id {rule_id}: {e}")
            return False
    
        for key in provided_params.keys():
            if key not in allowed_params:
                self.publish_log(f"Invalid json parameter '{key}' for rule_id {rule_id}. Allowed keys: {list(allowed_params.keys())}")
                return False
    
        return True

    def validate_create_payload(self, data: dict) -> bool:
        """Validates the payload for creating a rule applied."""
        if 'marker_id' not in data or 'rule_id' not in data or 'applied_id' not in data:
            self.publish_log("Payload missing required fields 'marker_id', 'rule_id', and/or 'applied_id' for create_rule_applied.")
            return False

        if 'json_params' in data and not isinstance(data['json_params'], dict):
            self.publish_log("Field 'json_params' must be a JSON object for create_rule_applied.")
            return False

        rule_id = data.get('rule_id')
        json_params = data.get('json_params', {})
        if not self.validate_rule_json_params(rule_id, json_params):
            return False

        # Check if applied_id already exists for the given rule_id
        try:
            with open(self.rule_config_file, 'r') as f:
                rule_config = json.load(f)
        except Exception as e:
            logging.error(f"Error loading rule.json: {e}")
            self.publish_log("Error loading rule configuration.")
            return False

        matched_rule = next((r for r in rule_config.get('rules', []) if r.get('id') == rule_id), None)
        if matched_rule and 'rule_applied' in matched_rule:
            if any(entry.get('applied_id') == data.get('applied_id') for entry in matched_rule['rule_applied']):
                self.publish_log(f"applied_id {data.get('applied_id')} already exists for rule_id {rule_id}")
                return False

        return True

    def validate_update_payload(self, data: dict) -> bool:
        if 'marker_id' not in data or 'rule_id' not in data or 'applied_id' not in data:
            self.publish_log("Payload missing one or more required fields ('marker_id', 'rule_id', 'id') for update_rule_applied.")
            return False
        if 'json_params' in data and not isinstance(data['json_params'], dict):
            self.publish_log("Field 'json_params' must be a JSON object for update_rule_applied.")
            return False
        rule_id = data.get('rule_id')
        json_params = data.get('json_params', {})
        if not self.validate_rule_json_params(rule_id, json_params):
            return False
        return True

    def validate_delete_payload(self, data: dict) -> bool:
        if 'marker_id' not in data or 'rule_id' not in data or 'applied_id' not in data:
            self.publish_log("Payload missing one or more required fields ('marker_id', 'rule_id', 'id') for delete_rule_applied.")
            return False
        return True

    def create_rule_applied(self, data: dict):
        """Handles the creation of a new rule applied to a marker and updates rule.json."""
        try:
            if not self.validate_create_payload(data):
                logging.error("Invalid payload for create_rule_applied.")
                return
            marker_id = data.get('marker_id')
            rule_id = data.get('rule_id')
            provided_params = data.get('json_params', {})  # expected as dict
            applied_id = data.get('applied_id')
            if applied_id is None:
                logging.error("Applied rule id is missing.")
                return
    
            # Load rule.json
            with open(self.rule_config_file, 'r') as f:
                rule_config = json.load(f)
    
            matched_rule = None
            for rule in rule_config.get('rules', []):
                if rule.get('id') == rule_id:
                    matched_rule = rule
                    break
            if not matched_rule:
                logging.error(f"No matching rule found for rule_id {rule_id}.")
                return
    
            # Prepare new applied rule entry using snake_case keys
            new_rule_entry = {
                "applied_id": applied_id,
                "marker_id": marker_id,
                "json_params": json.dumps(provided_params)
            }
            if "rule_applied" in matched_rule:
                if any(entry.get("applied_id") == applied_id for entry in matched_rule["rule_applied"]):
                    logging.error(f"Applied rule with id {applied_id} already exists in rule {rule_id}.")
                    return
                matched_rule["rule_applied"].append(new_rule_entry)
            else:
                matched_rule["rule_applied"] = [new_rule_entry]
    
            # Save the updated rule configuration
            with open(self.rule_config_file, 'w') as f:
                json.dump(rule_config, f, indent=4)
            logging.info(f"Created applied rule for marker {marker_id} in rule {rule_id}.")
        except Exception as e:
            logging.error(f"Error applying rule: {e}")
    
    def update_rule_applied(self, data: dict):
        """Handles the updating of an existing rule applied to a marker and updates rule.json."""
        try:
            if not self.validate_update_payload(data):
                logging.error("Invalid payload for update_rule_applied.")
                return
            marker_id = data.get('marker_id')
            rule_id = data.get('rule_id')
            provided_params = data.get('json_params', {})
            applied_id = data.get('applied_id')
            if applied_id is None:
                logging.error("Applied rule id is missing.")
                return
    
            # Load rule.json
            with open(self.rule_config_file, 'r') as f:
                rule_config = json.load(f)
    
            matched_rule = None
            for rule in rule_config.get("rules", []):
                if rule.get("id") == rule_id:
                    matched_rule = rule
                    break
            if not matched_rule:
                logging.error(f"No matching rule found for rule_id {rule_id}.")
                return
    
            # Update the matched applied rule entry using snake_case keys
            updated = False
            for applied in matched_rule.get("rule_applied", []):
                if applied.get("applied_id") == applied_id and applied.get("marker_id") == marker_id:
                    applied["json_params"] = json.dumps(provided_params)
                    updated = True
                    break
            if not updated:
                logging.error(f"Applied rule with id {applied_id} not found in rule {rule_id}.")
                return
    
            # Save the updated rule configuration
            with open(self.rule_config_file, 'w') as f:
                json.dump(rule_config, f, indent=4)
            logging.info(f"Updated applied rule id {applied_id} for marker {marker_id} in rule {rule_id}.")
        except Exception as e:
            logging.error(f"Error updating applied rule: {e}")
    
    def delete_rule_applied(self, data: dict):
        """Deletes an applied rule entry from rule.json only.
        This function does not modify marker_positions.json or interact with mask positions.
        """
        try:
            if not self.validate_delete_payload(data):
                logging.error("Invalid payload for delete_rule_applied.")
                return
    
            applied_id = data.get('applied_id')
            rule_id = data.get('rule_id')
    
            # Load rule.json
            try:
                with open(self.rule_config_file, 'r') as f:
                    rule_config = json.load(f)
            except Exception as e:
                logging.error(f"Error loading rule.json: {e}")
                self.publish_log("Error loading rule configuration.")
                return
    
            # Find the matching rule
            matched_rule = next((rule for rule in rule_config.get('rules', []) if rule.get('id') == rule_id), None)
            if not matched_rule:
                logging.error(f"No matching rule found for rule_id {rule_id}.")
                return
    
            # Remove the applied rule entry
            if 'rule_applied' in matched_rule:
                matched_rule['rule_applied'] = [
                    applied for applied in matched_rule['rule_applied']
                    if applied.get('applied_id') != applied_id
                ]
            else:
                logging.warning(f"No rule_applied found for rule_id {rule_id}")
                self.publish_log(f"No rule_applied found for rule_id {rule_id}")
                return
    
            # Save the updated rule.json
            try:
                with open(self.rule_config_file, 'w') as f:
                    json.dump(rule_config, f, indent=4)
                logging.info(f"Deleted applied rule {applied_id} from rule {rule_id}")
                self.publish_log(f"Deleted applied rule {applied_id} from rule {rule_id}")
            except Exception as e:
                logging.error(f"Error saving rule.json: {e}")
    
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
        marker_log_topic = f"{self.config.get('unique_id')}/{self.config.get('log_topic', 'log')}"
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