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

    def update_edge_id(self, new_edge_id: str):
        """Updates the edge_id and resubscribes to the new topics."""
        with self.lock:  # Ensure thread safety
            # Unsubscribe from old topics
            old_topics = [
                f"{self.edge_id}/marker/create",
                f"{self.edge_id}/marker/update",
                f"{self.edge_id}/marker/delete",
                f"{self.edge_id}/edge/create",
                f"{self.edge_id}/edge/update",
                f"{self.edge_id}/edge/delete"
            ]
            for topic in old_topics:
                self.mqtt_client.unsubscribe(topic)
                logging.info(f"Unsubscribed from topic: {topic}")

            # Update the edge_id
            self.edge_id = new_edge_id
            logging.info(f"Updated edge_id to: {self.edge_id}")

            # Subscribe to new topics using the updated edge_id
            self.subscribe_to_topics()

    def create_marker(self, data: Dict[str, Any]):
        try:
            with open(self.mask_positions_file, 'r') as f:
                mask_positions = json.load(f)
            mask_positions.append(data)
            with open(self.mask_positions_file, 'w') as f:
                json.dump(mask_positions, f, indent=4)
            logging.info(f"Created marker position: {data}")
            self.publish_log("create complete")
        except Exception as e:
            logging.error(f"Error creating marker position: {e}")

    def update_marker(self, data: Dict[str, Any]):
        try:
            with open(self.mask_positions_file, 'r') as f:
                mask_positions = json.load(f)
            for i, position in enumerate(mask_positions):
                if position.get('marker_id') == data.get('marker_id'):
                    mask_positions[i] = data
                    break
            with open(self.mask_positions_file, 'w') as f:
                json.dump(mask_positions, f, indent=4)
            logging.info(f"Updated marker position: {data}")
            self.publish_log("update complete")
        except Exception as e:
            logging.error(f"Error updating marker position: {e}")

    def delete_marker(self, data: Dict[str, Any]):
        try:
            with open(self.mask_positions_file, 'r') as f:
                mask_positions = json.load(f)
            mask_positions = [
                position for position in mask_positions
                if position.get('zone_id') != data.get('zone_id') and position.get('movement_id') != data.get('movement_id')
            ]
            with open(self.mask_positions_file, 'w') as f:
                json.dump(mask_positions, f, indent=4)
            logging.info(f"Deleted marker position: {data}")
            self.publish_log("delete complete")
        except Exception as e:
            logging.error(f"Error deleting marker position: {e}")

    def create_edge_device(self, data: Dict[str, Any]):
        """Handles the creation of a new edge device."""
        try:
            # Update the config with the new edge_id and rtsp_url
            self.config['edge_id'] = data.get('edge_id', self.config['edge_id'])
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
            # Update the config with the new edge_id and rtsp_url
            if 'edge_id' in data:
                self.config['edge_id'] = data['edge_id']
                # Notify MqttPublisher to update its edge_id
                self.mqtt_publisher.update_edge_id(data['edge_id'])
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