import json
import ssl
import paho.mqtt.client as mqtt_client
import numpy as np
import os

class MqttHandler:
    def __init__(self, config):
        self.emqx_host = config.get('emqx_host', 'localhost')
        self.emqx_port = config.get('emqx_port', 8883)
        self.emqx_username = config.get('emqx_username', '')
        self.emqx_password = config.get('emqx_password', '')
        self.emqx_topic = config.get('emqx_topic', 'detection_log')
        self.client_id = '123456'  # Or generate a unique ID

        self.mqtt_client = mqtt_client.Client(
            client_id=self.client_id,
            protocol=mqtt_client.MQTTv311
        )
        self.mqtt_client.username_pw_set(self.emqx_username, self.emqx_password)

        self.configure_tls()

        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_publish = self.on_publish
        self.mqtt_client.on_log = self.on_log

        self.connect()

    def configure_tls(self):
        try:
            ca_cert_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'emqxsl-ca.crt')
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
            print(f"Error configuring TLS: {e}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("MQTT client connected successfully.")
        else:
            print(f"MQTT client failed to connect. Return code: {rc}")

    def on_publish(self, client, userdata, mid):
        print(f"Message {mid} published.")

    def on_log(self, client, userdata, level, buf):
        print(f"MQTT Log: {buf}")

    def connect(self):
        try:
            print(f"Connecting to EMQX at {self.emqx_host}:{self.emqx_port} over TLS/SSL...")
            self.mqtt_client.connect(self.emqx_host, self.emqx_port)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"Could not connect to EMQX: {e}")

    def publish_detection(self, detection_entry):
        try:
            serializable_entry = {}
            for key, value in detection_entry.items():
                if isinstance(value, (np.integer, np.int32, np.int64)):
                    serializable_entry[key] = int(value)
                elif isinstance(value, (np.float32, np.float64)):
                    serializable_entry[key] = float(value)
                else:
                    serializable_entry[key] = value

            result = self.mqtt_client.publish(self.emqx_topic, json.dumps(serializable_entry))
            status = result[0]
            if status == mqtt_client.MQTT_ERR_SUCCESS:
                print(f"Sent `{serializable_entry}` to topic `{self.emqx_topic}`")
            else:
                print(f"Failed to send message to topic {self.emqx_topic}")
        except Exception as e:
            print(f"Error publishing to EMQX: {e}")

    def disconnect(self):
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        print("Disconnected from EMQX.")