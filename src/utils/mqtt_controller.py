"""
src/utils/mqtt_controller.py
============================
Gate control via MQTT, with a hardware-free mock fallback.

    MQTTGateController      -> publishes JSON commands to an MQTT broker
                               (the ESP32 subscribes and drives the relay).
    MockMQTTGateController  -> no hardware/broker; prints + logs commands.

create_gate_controller(config) picks the right one: it tries the real broker
and automatically falls back to the mock if MQTT is disabled/unavailable, so
the whole system runs on a laptop with no ESP32 present.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path


# --------------------------------------------------------------------------- #
# Mock (no hardware needed)
# --------------------------------------------------------------------------- #
class MockMQTTGateController:
    def __init__(self, gate_id: str = "main_gate", log_dir: str = "logs") -> None:
        self.gate_id = gate_id
        self.log_path = Path(log_dir) / "mock_gate_log.txt"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, msg: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(msg)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            print(f"[MockGate] log error: {exc}")

    def open_gate(self, plate_text: str, duration_sec: int = 3) -> None:
        self._log(f"[GATE] OPEN  <- plate: {plate_text} ({duration_sec} sec)")
        self._log(f"[GATE] CLOSE <- auto after {duration_sec}s")

    def close_gate(self) -> None:
        self._log("[GATE] CLOSE")

    def emergency_stop(self) -> None:
        self._log("[GATE] EMERGENCY_STOP")

    def is_connected(self) -> bool:
        return True

    def get_status(self) -> str:
        return "mock"


# --------------------------------------------------------------------------- #
# Real MQTT
# --------------------------------------------------------------------------- #
class MQTTGateController:
    def __init__(self, broker_host: str, broker_port: int = 1883,
                 gate_id: str = "main_gate") -> None:
        import paho.mqtt.client as mqtt  # raises ImportError if not installed

        self.gate_id = gate_id
        self.control_topic = f"alpr/{gate_id}/control"
        self.status_topic = f"alpr/{gate_id}/status"
        self._connected = False

        self.client = mqtt.Client()

        def on_connect(client, userdata, flags, rc):
            self._connected = (rc == 0)

        def on_disconnect(client, userdata, rc):
            self._connected = False

        self.client.on_connect = on_connect
        self.client.on_disconnect = on_disconnect
        # connect() can raise (no broker) — caller handles fallback.
        self.client.connect(broker_host, broker_port, keepalive=60)
        self.client.loop_start()

    def _publish(self, payload: dict) -> None:
        try:
            self.client.publish(self.control_topic, json.dumps(payload))
        except Exception as exc:
            print(f"[MQTTGate] publish error: {exc}")

    def open_gate(self, plate_text: str, duration_sec: int = 3) -> None:
        self._publish({
            "command": "OPEN",
            "plate": plate_text,
            "duration": duration_sec,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    def close_gate(self) -> None:
        self._publish({"command": "CLOSE",
                       "timestamp": datetime.now().isoformat(timespec="seconds")})

    def emergency_stop(self) -> None:
        self._publish({"command": "EMERGENCY_STOP",
                       "timestamp": datetime.now().isoformat(timespec="seconds")})

    def is_connected(self) -> bool:
        return self._connected

    def get_status(self) -> str:
        if self._connected:
            return "connected"
        return "disconnected"

    def stop(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Factory with auto-fallback
# --------------------------------------------------------------------------- #
def create_gate_controller(config: dict):
    """
    config : the parsed system config dict (expects config['mqtt'] and
             config.get('logging',{})).
    Returns a real MQTTGateController if the broker is reachable, else a
    MockMQTTGateController.
    """
    mqtt_cfg = (config or {}).get("mqtt", {}) or {}
    log_dir = ((config or {}).get("logging", {}) or {}).get("log_dir", "logs")
    gate_id = mqtt_cfg.get("gate_id", "main_gate")

    if mqtt_cfg.get("enabled", False):
        try:
            ctrl = MQTTGateController(
                mqtt_cfg.get("broker_host", "localhost"),
                int(mqtt_cfg.get("broker_port", 1883)),
                gate_id,
            )
            # wait up to 2s for the broker to confirm the connection
            deadline = time.time() + 2.0
            while time.time() < deadline and not ctrl.is_connected():
                time.sleep(0.1)
            if ctrl.is_connected():
                print(f"[gate] connected to MQTT broker "
                      f"{mqtt_cfg.get('broker_host')}:{mqtt_cfg.get('broker_port')}")
                return ctrl
            ctrl.stop()
        except Exception as exc:
            print(f"[gate] MQTT connect failed ({exc}).")
        print("MQTT broker not found - using mock gate controller")

    return MockMQTTGateController(gate_id, log_dir)
