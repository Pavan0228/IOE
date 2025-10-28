#ifndef RFIDReader_h
#define RFIDReader_h

#include <SPI.h>
#include <MFRC522.h>

class RFIDReader {
  private:
    #define SS_PIN  15
    #define RST_PIN 2
    MFRC522 mfrc522;
  public:
    RFIDReader() : mfrc522(SS_PIN, RST_PIN) {}
    void init();
    String readTag();
};

#endif
