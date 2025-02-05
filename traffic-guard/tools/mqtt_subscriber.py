import json
import threading
import logging
from typing import Dict, Any
from paho.mqtt import client as mqtt_client
import ssl
import uuid
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
        self.config_file = 'traffic-guard/config/config.json'
        self.config = config
        self.mqtt_publisher = mqtt_publisher  # Pass the MqttPublisher instance

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
        """Subscribes to the required MQTT topics based on the current edgeDeviceId."""
        topics = [
            f"{self.uniqueId}/marker/create",
            f"{self.uniqueId}/marker/update",
            f"{self.uniqueId}/marker/delete",
            f"{self.uniqueId}/create",
            f"{self.uniqueId}/update",
            f"{self.uniqueId}/delete"
        ]
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

    def create_edge_device(self, data: Dict[str, Any]):
        """Handles the creation of a new edge device."""
        try:
            # Update the config with the new edgeDeviceId and rtspUrl
            new_edgeDeviceId = data.get('edgeDeviceId', self.config['edgeDeviceId'])
            self.config['edgeDeviceId'] = new_edgeDeviceId
            self.config['rtspUrl'] = data.get('rtspUrl', self.config.get('rtspUrl', ''))
            
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
            # Reset the edgeDeviceId and rtspUrl to default values
            self.config['edgeDeviceId'] = ''
            self.config['rtspUrl'] = ''
            
            # Save the updated config to the config.json file
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            logging.info(f"Deleted edge device: {data}")
            self.publish_log("edge delete complete")
        except Exception as e:
            logging.error(f"Error deleting edge device: {e}")

    def publish_log(self, message: str):
        """Publishes a log message to the 'marker_positions/log' topic."""
        try:
            self.mqtt_client.publish(f'{self.uniqueId}/marker/log', message)
        except Exception as e:
            logging.error(f"Error publishing log message: {e}")

    def disconnect(self):
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        logging.info("Disconnected from EMQX.")