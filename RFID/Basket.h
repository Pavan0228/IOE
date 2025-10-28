#ifndef Basket_h
#define Basket_h

#include <Arduino.h>
#include <map>

class Basket {
  private:
    std::map<String, String> items;
    std::map<String, float> prices;
    float total = 0;
  public:
    Basket();
    String addItem(String tag);
    String getBill(String paymentMethod);
    void clear();
};

#endif
