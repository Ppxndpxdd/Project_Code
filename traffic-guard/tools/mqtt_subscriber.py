import json
import logging
from typing import Dict, Any
from paho.mqtt import client as mqtt_client
import ssl
import uuid

class MqttSubscriber:
    def __init__(self, config: Dict[str, Any]):
        self.emqx_host = config.get('mqtt_broker', 'localhost')
        self.emqx_port = config.get('mqtt_port', 8883)
        self.emqx_username = config.get('mqtt_username', '')
        self.emqx_password = config.get('mqtt_password', '')
        self.client_id = 'client-' + str(uuid.uuid4())
        self.mask_positions_file = config.get('mask_json_path', 'mask_positions.json')

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
            if msg.topic == 'mask_positions/create':
                self.create_mask_position(payload)
            elif msg.topic == 'mask_positions/update':
                self.update_mask_position(payload)
            elif msg.topic == 'mask_positions/delete':
                self.delete_mask_position(payload)
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
        self.mqtt_client.subscribe('mask_positions/create')
        self.mqtt_client.subscribe('mask_positions/update')
        self.mqtt_client.subscribe('mask_positions/delete')
        logging.info("Subscribed to mask_positions topics.")

    def create_mask_position(self, data: Dict[str, Any]):
        try:
            with open(self.mask_positions_file, 'r') as f:
                mask_positions = json.load(f)
            mask_positions.append(data)
            with open(self.mask_positions_file, 'w') as f:
                json.dump(mask_positions, f, indent=4)
            logging.info(f"Created mask position: {data}")
        except Exception as e:
            logging.error(f"Error creating mask position: {e}")

    def update_mask_position(self, data: Dict[str, Any]):
        try:
            with open(self.mask_positions_file, 'r') as f:
                mask_positions = json.load(f)
            for i, position in enumerate(mask_positions):
                if position['frame'] == data['frame'] and position['zone_id'] == data['zone_id']:
                    mask_positions[i] = data
                    break
            with open(self.mask_positions_file, 'w') as f:
                json.dump(mask_positions, f, indent=4)
            logging.info(f"Updated mask position: {data}")
        except Exception as e:
            logging.error(f"Error updating mask position: {e}")

    def delete_mask_position(self, data: Dict[str, Any]):
        try:
            with open(self.mask_positions_file, 'r') as f:
                mask_positions = json.load(f)
            mask_positions = [position for position in mask_positions if not (position['frame'] == data['frame'] and position['zone_id'] == data['zone_id'])]
            with open(self.mask_positions_file, 'w') as f:
                json.dump(mask_positions, f, indent=4)
            logging.info(f"Deleted mask position: {data}")
        except Exception as e:
            logging.error(f"Error deleting mask position: {e}")

    def disconnect(self):
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        logging.info("Disconnected from EMQX.")