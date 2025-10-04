#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "Airtel_guna";
const char* password = "Guna@8864";

WebServer server(80);

int buzzerPin = 5;
int buzzCount = 0;
unsigned long buzzTimer = 0;
bool buzzerState = false;
const unsigned long buzzInterval = 500; // 500 ms beep

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  Serial.println("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi connected");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  pinMode(buzzerPin, OUTPUT);
  digitalWrite(buzzerPin, LOW);

  server.on("/alert", handleAlert);
  server.begin();
  Serial.println("HTTP server started");
}

void handleAlert() {
  buzzCount = 3;           // number of buzzes
  buzzTimer = millis();    // start timer
  buzzerState = true;      // start buzzing
  Serial.println("Alert received!");
  server.send(200, "text/plain", "Buzzer triggered");
}

void loop() {
  server.handleClient();

  if (buzzCount > 0) {
    if (millis() - buzzTimer >= buzzInterval) {
      // toggle buzzer state
      buzzerState = !buzzerState;
      digitalWrite(buzzerPin, buzzerState ? HIGH : LOW);

      // only count one cycle when turning OFF
      if (!buzzerState) buzzCount--;

      buzzTimer = millis(); // reset timer
    }
  } else {
    digitalWrite(buzzerPin, LOW); // ensure buzzer OFF
  }
}
