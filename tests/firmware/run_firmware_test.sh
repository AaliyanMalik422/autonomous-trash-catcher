#!/bin/sh
# Legacy C++ host harness for the Mecanum firmware.
# The corresponding source is not present in this repository checkout.
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
if [ ! -f "$ROOT/mecanum_locomotion/mecanum_locomotion.ino" ]; then
    echo "SKIP: mecanum_locomotion/mecanum_locomotion.ino is missing; firmware host test cannot run." >&2
    exit 77
fi
cd "$(dirname "$0")"
g++ -std=c++17 -O1 -Wall -Wno-unused-function -I. host_test.cpp -o /tmp/fw_host_test
/tmp/fw_host_test
