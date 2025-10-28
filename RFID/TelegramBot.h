#ifndef TelegramBot_h
#define TelegramBot_h

#include <WiFiClientSecure.h>
#include <UniversalTelegramBot.h>

class TelegramBot {
  private:
    String botToken;
    String chatId;
    WiFiClientSecure client;
    UniversalTelegramBot* bot;
  public:
    TelegramBot(String token, String chat);
    void sendMessage(String msg);
};

#endif
