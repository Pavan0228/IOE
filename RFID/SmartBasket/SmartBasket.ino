#include <WiFi.h>
#include "RFIDReader.h"
#include "Basket.h"
#include "TelegramBot.h"

// ==== WiFi Credentials ====
const char* ssid = "PRANAV";
const char* password = "Pranav@@##";

// ==== Global Objects ====
RFIDReader rfid;
Basket basket;
TelegramBot telegramBot("YOUR_TELEGRAM_BOT_TOKEN", "YOUR_CHAT_ID");

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");

  rfid.init();
  telegramBot.sendMessage("Smart Basket System Started ✅");
}

void loop() {
  String item = rfid.readTag();
  if (item != "") {
    String itemName = basket.addItem(item);
    telegramBot.sendMessage("🛒 Item added: " + itemName);
  }

  // For demo, send bill if you type 'b' in Serial
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'b') {
      String bill = basket.getBill("Cash");
      telegramBot.sendMessage(bill);
      basket.clear();
    }
  }

  delay(200);
}
