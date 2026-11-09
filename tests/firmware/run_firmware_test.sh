#!/bin/sh
# Compiles the ESP32 firmware on the PC with Arduino stubs and runs a closed-loop test.
cd "$(dirname "$0")" && g++ -std=c++17 -O1 -Wall -Wno-unused-function -I. host_test.cpp -o /tmp/fw_host_test && /tmp/fw_host_test
