#include "RFIDReader.h"

void RFIDReader::init() {
  SPI.begin(14, 12, 13, 15);
  mfrc522.PCD_Init();
  Serial.println("RFID Reader Ready");
}

String RFIDReader::readTag() {
  if (!mfrc522.PICC_IsNewCardPresent() || !mfrc522.PICC_ReadCardSerial())
    return "";

  String tag = "";
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    tag += String(mfrc522.uid.uidByte[i] < 0x10 ? "0" : "");
    tag += String(mfrc522.uid.uidByte[i], HEX);
  }
  mfrc522.PICC_HaltA();
  tag.toUpperCase();
  return tag;
}
