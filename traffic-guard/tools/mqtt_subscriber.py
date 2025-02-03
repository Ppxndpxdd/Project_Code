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
        self.edge_id = config.get('edge_id', 'default_id')
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
            if msg.topic == f"{self.edge_id}/marker/create":
                self.create_marker(payload)
            elif msg.topic == f"{self.edge_id}/marker/update":
                self.update_marker(payload)
            elif msg.topic == f"{self.edge_id}/marker/delete":
                self.delete_marker(payload)
            elif msg.topic == f"{self.edge_id}/edge/create":
                self.create_edge_device(payload)
            elif msg.topic == f"{self.edge_id}/edge/update":
                self.update_edge_device(payload)
            elif msg.topic == f"{self.edge_id}/edge/delete":
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
        """Subscribes to the required MQTT topics based on the current edge_id."""
        topics = [
            f"{self.edge_id}/marker/create",
            f"{self.edge_id}/marker/update",
            f"{self.edge_id}/marker/delete",
            f"{self.edge_id}/edge/create",
            f"{self.edge_id}/edge/update",
            f"{self.edge_id}/edge/delete"
        ]
        for topic in topics:
            self.mqtt_client.subscribe(topic)
            logging.info(f"Subscribed to topic: '{topic}'")

    # Replace the update_edge_id method with the following code:
    def update_edge_id(self, new_edge_id: str):
        """Forces an update of the edge_id and resubscribes to the new topics,
        even if the new_edge_id matches the current edge_id.
        This ensures the config.json value is respected on every update."""
        with self.lock:  # Ensure thread safety
            old_edge_id = self.edge_id
            # Always perform the unsubscription to force a refresh
            old_topics = [
                f"{old_edge_id}/marker/create",
                f"{old_edge_id}/marker/update",
                f"{old_edge_id}/marker/delete",
                f"{old_edge_id}/edge/create",
                f"{old_edge_id}/edge/update",
                f"{old_edge_id}/edge/delete"
            ]
            for topic in old_topics:
                self.mqtt_client.unsubscribe(topic)
                logging.info(f"Unsubscribed from topic: {topic}")

            # Update the edge_id from the one in config (ensuring it is fresh)
            self.edge_id = new_edge_id
            logging.info(f"Force-updated edge_id to: {self.edge_id}")

            # Subscribe to new topics using the updated edge_id
            self.subscribe_to_topics()

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
            # Update the config with the new edge_id and rtsp_url
            new_edge_id = data.get('edge_id', self.config['edge_id'])
            self.config['edge_id'] = new_edge_id
            self.config['rtsp_url'] = data.get('rtsp_url', self.config.get('rtsp_url', ''))
            
            # Save the updated config to the config.json file
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            logging.info(f"Created edge device: {data}")
            self.publish_log("edge create complete")
            # Update subscriber edge_id and topics
            self.update_edge_id(new_edge_id)
        except Exception as e:
            logging.error(f"Error creating edge device: {e}")

    def update_edge_device(self, data: Dict[str, Any]):
        """Handles the updating of an existing edge device."""
        try:
            # Update the config with the new edge_id and rtsp_url
            if 'edge_id' in data:
                new_edge_id = data['edge_id']
                self.config['edge_id'] = new_edge_id
                # Notify MqttPublisher to update its edge_id as well
                self.mqtt_publisher.update_edge_id(new_edge_id)
                # Update subscriber's edge_id and topics
                self.update_edge_id(new_edge_id)
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
            # Reset the edge_id and rtsp_url to default values
            self.config['edge_id'] = 'default_id'
            self.config['rtsp_url'] = ''
            
            # Save the updated config to the config.json file
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            logging.info(f"Deleted edge device: {data}")
            self.publish_log("edge delete complete")
            # Update subscriber edge_id and topics to default
            self.update_edge_id('default_id')
        except Exception as e:
            logging.error(f"Error deleting edge device: {e}")

    def publish_log(self, message: str):
        """Publishes a log message to the 'marker_positions/log' topic."""
        try:
            self.mqtt_client.publish(f'{self.edge_id}/marker/log', message)
        except Exception as e:
            logging.error(f"Error publishing log message: {e}")

    def disconnect(self):
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        logging.info("Disconnected from EMQX.")