import json
import ssl
import logging
import uuid
from typing import Dict, Any
from paho.mqtt import client as mqtt_client
import numpy as np

class MqttHandler:
    def __init__(self, config: Dict[str, Any]):
        self.emqx_host = config.get('mqtt_broker', 'localhost')
        self.emqx_port = config.get('mqtt_port', 8883)
        self.emqx_username = config.get('mqtt_username', '')
        self.emqx_password = config.get('mqtt_password', '')
        self.emqx_topic = config.get('incident_info_topic', 'detection_log')
        self.client_id = 'client-' + str(uuid.uuid4())

        self.mqtt_client = mqtt_client.Client(client_id=self.client_id, protocol=mqtt_client.MQTTv311)
        self.mqtt_client.username_pw_set(self.emqx_username, self.emqx_password)
        self.configure_tls(config.get('ca_cert_path', 'emqxsl-ca.crt'))

        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_publish = self.on_publish
        self.mqtt_client.on_log = self.on_log

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
        else:
            logging.error(f"MQTT client failed to connect. Return code: {rc}")

    def on_publish(self, client, userdata, mid):
        logging.debug(f"Message {mid} published.")

    def on_log(self, client, userdata, level, buf):
        logging.debug(f"MQTT Log: {buf}")

    def connect(self):
        try:
            logging.info(f"Connecting to EMQX at {self.emqx_host}:{self.emqx_port} over TLS/SSL...")
            self.mqtt_client.connect(self.emqx_host, self.emqx_port)
            self.mqtt_client.loop_start()
        except Exception as e:
            logging.error(f"Could not connect to EMQX: {e}")

    def publish(self, topic, payload):
        """Publish a message to the MQTT broker."""
        result = self.mqtt_client.publish(topic, payload)
        status = result[0]
        if status == mqtt_client.MQTT_ERR_SUCCESS:
            logging.info(f"Sent `{payload}` to topic `{topic}`")
        else:
            logging.error(f"Failed to send message to topic {topic}")

    def publish_detection(self, detection_entry: Dict[str, Any]):
        try:
            serializable_entry = {k: (int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else float(v) if isinstance(v, (np.float32, np.float64)) else v) for k, v in detection_entry.items()}
            self.publish(self.emqx_topic, json.dumps(serializable_entry))
        except Exception as e:
            logging.error(f"Error publishing to EMQX: {e}")

    def disconnect(self):
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        logging.info("Disconnected from EMQX.")