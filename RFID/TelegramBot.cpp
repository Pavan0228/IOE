#include "TelegramBot.h"

TelegramBot::TelegramBot(String token, String chat) {
  botToken = token;
  chatId = chat;
  client.setInsecure();
  bot = new UniversalTelegramBot(botToken, client);
}

void TelegramBot::sendMessage(String msg) {
  bot->sendMessage(chatId, msg, "");
}
