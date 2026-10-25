#include <WiFi.h>
#include <WebServer.h>
#include "DHT.h"

// -----------------------------------------
// Configuration
// -----------------------------------------
const char* ssid = "ssid";
const char* password = "pw";

// DHT Sensor Setup
#define DHTPIN 4
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

// Soil Moisture Sensor Setup
#define SOIL_PIN A4
#define PUMP_PIN D3
// Web server on port 80
WebServer server(80);

// -----------------------------------------
// WiFi reconnect settings
// -----------------------------------------
unsigned long lastWifiCheck = 0;
const unsigned long wifiCheckInterval = 600000;   // check every 30.000 = 30 seconds; 1800000 = 30min; 600000 = 10min

unsigned long lastReconnectAttempt = 0;
const unsigned long reconnectInterval = 10000;   // try reconnect every 10000 = 10 seconds

bool wifiWasConnected = false;

// -----------------------------------------
// Functions
// -----------------------------------------
void connectToWiFi() {
  Serial.println();
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);   // avoids writing WiFi config repeatedly to flash
  WiFi.begin(ssid, password);
}

void printWiFiStatus() {
  Serial.print("WiFi status: ");
  Serial.println(WiFi.status());

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Connected. IP address: ");
    Serial.println(WiFi.localIP());
  }
}

void checkWiFiConnection() {
  unsigned long now = millis();

  // only check every wifiCheckInterval
  if (now - lastWifiCheck < wifiCheckInterval) {
    return;
  }
  lastWifiCheck = now;

  wl_status_t status = WiFi.status();

  if (status == WL_CONNECTED) {
    if (!wifiWasConnected) {
      Serial.println("WiFi reconnected.");
      Serial.print("IP address: ");
      Serial.println(WiFi.localIP());
      wifiWasConnected = true;
    }
    return;
  }

  // if we get here, WiFi is disconnected
  if (wifiWasConnected) {
    Serial.println("WiFi connection lost.");
    wifiWasConnected = false;
  }

  // avoid reconnect spamming
  if (now - lastReconnectAttempt >= reconnectInterval) {
    lastReconnectAttempt = now;

    Serial.println("Attempting WiFi reconnect...");
    WiFi.disconnect();
    delay(100);
    WiFi.begin(ssid, password);
  }
}

void handleMetrics() {
  float temp = dht.readTemperature();
  float hum = dht.readHumidity();

  int rawMoisture = analogRead(SOIL_PIN);

  // Calibration values
  int airValue = 2199; //2nd:2199 // 1st: 3500
  int waterValue = 1657; //2nd:1657 // 1st: 1200

  float bodenfeuchte_Prozent = map(rawMoisture, airValue, waterValue, 0, 100);

  if (bodenfeuchte_Prozent < 0) bodenfeuchte_Prozent = 0;
  if (bodenfeuchte_Prozent > 100) bodenfeuchte_Prozent = 100;

  String tempStr = isnan(temp) ? "null" : String(temp, 2);
  String humStr = isnan(hum) ? "null" : String(hum, 2);

  String json = "{";
  json += "\"temperatur\": " + tempStr + ", ";
  json += "\"luftfeuchte\": " + humStr + ", ";
  json += "\"bodenfeuchte\": " + String(bodenfeuchte_Prozent, 2) + ", ";
  json += "\"raw_boden_analog\": " + String(rawMoisture);
  json += "}";

  server.send(200, "application/json", json);
}

void setup() {
  Serial.begin(115200);
  pinMode(PUMP_PIN, OUTPUT);
  digitalWrite(PUMP_PIN, LOW);
  delay(1000);

  dht.begin();

  connectToWiFi();

  // Wait some time for first connection, but not forever
  unsigned long startAttemptTime = millis();
  const unsigned long wifiTimeout = 20000; // 20 seconds

  while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < wifiTimeout) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  printWiFiStatus();
  wifiWasConnected = (WiFi.status() == WL_CONNECTED);

  server.on("/metrics", HTTP_GET, handleMetrics);
  server.begin();

  Serial.println("HTTP server started.");
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("API Address: http://");
    Serial.print(WiFi.localIP());
    Serial.println("/metrics");
  }
}

void loop() {
  checkWiFiConnection();
  server.handleClient();
  delay(10);
}