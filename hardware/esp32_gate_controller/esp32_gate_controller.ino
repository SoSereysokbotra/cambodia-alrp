// =====================================================================
// ESP32 Gate Controller
// Cambodian ALPR Project
// ---------------------------------------------------------------------
// Subscribes to MQTT gate commands published by the edge PC (the Python
// ALPR system) and drives a relay + status LEDs. Publishes back a status
// message so the PC knows the gate acted.
//
// REQUIRED ARDUINO LIBRARIES (install via Library Manager):
//   - WiFi          (bundled with the ESP32 board package)
//   - PubSubClient  (by Nick O'Leary)   -> MQTT
//   - ArduinoJson   (by Benoit Blanchon)-> JSON parsing (v6+)
//
// BOARD: any ESP32 dev board (e.g. "ESP32 Dev Module").
//
// WIRING (adjust pins below to your board):
//   RELAY_PIN (5)  -> relay IN  (relay drives the gate motor/lock)
//   LED_GREEN (2)  -> green LED (+ resistor) : ENTRY_ALLOWED / gate open
//   LED_RED   (4)  -> red LED   (+ resistor) : ENTRY_DENIED / emergency
// =====================================================================

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ---------------- Pin definitions ---------------- //
#define RELAY_PIN   5    // gate relay
#define LED_GREEN   2    // allowed indicator
#define LED_RED     4    // denied / emergency indicator

// ---------------- WiFi + MQTT config -------------- //
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* MQTT_BROKER   = "192.168.1.XXX";   // edge PC IP running the broker
const int   MQTT_PORT     = 1883;

const char* SUB_TOPIC = "alpr/main_gate/control";   // commands IN
const char* PUB_TOPIC = "alpr/main_gate/status";    // status OUT
const char* CLIENT_ID = "esp32_main_gate";

WiFiClient   wifiClient;
PubSubClient client(wifiClient);

// =====================================================================
// WiFi
// =====================================================================
void connectWiFi() {
  Serial.printf("Connecting to WiFi '%s' ", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\nWiFi connected. IP: %s\n", WiFi.localIP().toString().c_str());
}

// =====================================================================
// Gate actions
// =====================================================================
void openGate(const char* plate, int durationSec) {
  Serial.printf("[GATE] OPEN for plate %s (%d s)\n", plate, durationSec);
  digitalWrite(RELAY_PIN, HIGH);
  digitalWrite(LED_GREEN, HIGH);
  digitalWrite(LED_RED, LOW);

  // publish status: opened
  StaticJsonDocument<128> doc;
  doc["status"] = "opened";
  doc["plate"]  = plate;
  char buf[128];
  serializeJson(doc, buf);
  client.publish(PUB_TOPIC, buf);

  // hold the gate open, then close (blocking is fine for a single gate)
  delay((unsigned long)durationSec * 1000UL);

  digitalWrite(RELAY_PIN, LOW);
  digitalWrite(LED_GREEN, LOW);
  Serial.println("[GATE] auto-closed");

  StaticJsonDocument<128> doc2;
  doc2["status"] = "closed";
  doc2["plate"]  = plate;
  serializeJson(doc2, buf);
  client.publish(PUB_TOPIC, buf);
}

void closeGate() {
  Serial.println("[GATE] CLOSE");
  digitalWrite(RELAY_PIN, LOW);
  digitalWrite(LED_GREEN, LOW);
  client.publish(PUB_TOPIC, "{\"status\":\"closed\"}");
}

void emergencyStop() {
  Serial.println("[GATE] EMERGENCY_STOP");
  digitalWrite(RELAY_PIN, LOW);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_RED, HIGH);
  client.publish(PUB_TOPIC, "{\"status\":\"emergency_stop\"}");
}

// =====================================================================
// MQTT message callback
// =====================================================================
void callback(char* topic, byte* payload, unsigned int length) {
  // Copy payload into a null-terminated buffer.
  char message[256];
  unsigned int n = (length < sizeof(message) - 1) ? length : sizeof(message) - 1;
  memcpy(message, payload, n);
  message[n] = '\0';
  Serial.printf("[MQTT] %s -> %s\n", topic, message);

  // Parse JSON: {"command":"OPEN","plate":"1AB-2345","duration":3,...}
  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, message);
  if (err) {
    Serial.printf("[MQTT] JSON parse error: %s\n", err.c_str());
    return;
  }

  const char* command  = doc["command"] | "";
  const char* plate    = doc["plate"]   | "unknown";
  int         duration = doc["duration"] | 3;

  if (strcmp(command, "OPEN") == 0) {
    openGate(plate, duration);
  } else if (strcmp(command, "CLOSE") == 0) {
    closeGate();
  } else if (strcmp(command, "EMERGENCY_STOP") == 0) {
    emergencyStop();
  } else {
    Serial.printf("[MQTT] unknown command: %s\n", command);
  }
}

// =====================================================================
// MQTT (re)connect
// =====================================================================
void connectMQTT() {
  while (!client.connected()) {
    Serial.printf("Connecting to MQTT %s:%d ... ", MQTT_BROKER, MQTT_PORT);
    if (client.connect(CLIENT_ID)) {
      Serial.println("connected.");
      client.subscribe(SUB_TOPIC);
      Serial.printf("Subscribed to %s\n", SUB_TOPIC);
      client.publish(PUB_TOPIC, "{\"status\":\"online\"}");
    } else {
      Serial.printf("failed (rc=%d), retrying in 2s\n", client.state());
      delay(2000);
    }
  }
}

// =====================================================================
// Setup / loop
// =====================================================================
void setup() {
  Serial.begin(115200);
  delay(200);

  pinMode(RELAY_PIN, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);   // gate closed on boot (fail-safe)
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_RED, LOW);

  connectWiFi();
  client.setServer(MQTT_BROKER, MQTT_PORT);
  client.setCallback(callback);
  connectMQTT();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }
  if (!client.connected()) {
    connectMQTT();
  }
  client.loop();   // process incoming MQTT messages
}
