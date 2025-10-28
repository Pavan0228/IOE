#include "Basket.h"

Basket::Basket() {
  // Map RFID tag IDs to item names and prices
  items["A1B2C3D4"] = "Milk";
  prices["A1B2C3D4"] = 45.0;

  items["B2C3D4E5"] = "Bread";
  prices["B2C3D4E5"] = 30.0;

  items["C3D4E5F6"] = "Eggs";
  prices["C3D4E5F6"] = 60.0;
}

String Basket::addItem(String tag) {
  if (items.find(tag) != items.end()) {
    total += prices[tag];
    return items[tag] + " (₹" + String(prices[tag]) + ")";
  } else {
    return "Unknown Item (" + tag + ")";
  }
}

String Basket::getBill(String paymentMethod) {
  String bill = "🧾 *Bill Summary* \n\n";
  for (auto &it : items) {
    bill += it.second + " - ₹" + String(prices[it.first]) + "\n";
  }
  bill += "\n💰 Total: ₹" + String(total) + "\n";
  bill += "Payment: " + paymentMethod + "\n";
  bill += "\nThank you for shopping!";
  return bill;
}

void Basket::clear() {
  total = 0;
}
