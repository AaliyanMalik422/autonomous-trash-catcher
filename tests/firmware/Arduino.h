// Minimal host-side stand-in for Arduino.h so the ESP32 firmware logic can be
// compiled and unit-tested on a PC (no hardware needed).
#pragma once
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>

#define IRAM_ATTR
#define ESP_ARDUINO_VERSION_MAJOR 3
#ifndef PI
#define PI 3.14159265358979f
#endif
#define HIGH 1
#define LOW 0
#define INPUT 0
#define OUTPUT 1
#define CHANGE 1

template <class T> T constrain(T x, T lo, T hi) { return x < lo ? lo : (x > hi ? hi : x); }

extern uint32_t g_micros;
inline uint32_t micros() { return g_micros; }
inline uint32_t millis() { return g_micros / 1000; }

extern int g_pin[64];
extern uint32_t g_duty[64];
inline void pinMode(int, int) {}
inline void digitalWrite(int p, int v) { g_pin[p] = v; }
inline int digitalRead(int) { return 0; }
inline void attachInterrupt(int, void (*)(), int) {}
inline int digitalPinToInterrupt(int p) { return p; }
inline void ledcAttach(int, uint32_t, uint8_t) {}
inline void ledcWrite(int pin, uint32_t duty) { g_duty[pin] = duty; }

struct SerialClass {
  std::deque<char> rx;
  std::string tx;
  void begin(int) {}
  int available() { return (int)rx.size(); }
  int read() { int c = rx.front(); rx.pop_front(); return c; }
  void feed(const std::string &s) { for (char c : s) rx.push_back(c); }
  void println(const char *s) { tx += s; tx += "\n"; }
  int printf(const char *fmt, ...) {
    char b[160]; va_list a; va_start(a, fmt);
    int n = vsnprintf(b, sizeof b, fmt, a); va_end(a); tx += b; return n;
  }
};
extern SerialClass Serial;
