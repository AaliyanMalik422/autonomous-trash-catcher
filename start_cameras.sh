#!/usr/bin/env bash
# Autonomous Trash Catcher - Dual Phone USB Camera Bring-up
set -e

echo "=========================================================="
echo " Starting Dual Phone USB Camera Streams via v4l2loopback  "
echo "=========================================================="

# 1. Ensure loopback nodes exist
if [ ! -e /dev/video10 ] || [ ! -e /dev/video11 ]; then
    echo "[1/3] Loading v4l2loopback kernel module for /dev/video10 and /dev/video11..."
    sudo modprobe -r v4l2loopback 2>/dev/null || true
    sudo modprobe v4l2loopback devices=2 video_nr=10,11 card_label="StereoLeft","StereoRight" exclusive_caps=1
else
    echo "[1/3] /dev/video10 and /dev/video11 already exist."
fi

# 2. Check ADB devices
echo "[2/3] Checking connected ADB phones..."
adb devices -l

SCR="$HOME/opt/scrcpy/scrcpy-linux-x86_64-v5.0.1/scrcpy"
if [ ! -f "$SCR" ]; then
    SCR="scrcpy"
fi

# Kill any existing scrcpy v4l2 streams
pkill -f "scrcpy.*v4l2-sink" 2>/dev/null || true
sleep 1

# 3. Launch background streams
echo "[3/3] Launching camera feeds (640x480 @ 30 FPS)..."

# Left Phone (fec6wctc4ta6m7mv) -> /dev/video10
"$SCR" -s fec6wctc4ta6m7mv --video-source=camera --camera-id=0 --camera-size=640x480 --camera-fps=30 --v4l2-sink=/dev/video10 --no-window --no-audio >/tmp/cam_left.log 2>&1 &
PID_L=$!

# Right Phone (dqeu89yh7xso8dq4) -> /dev/video11
"$SCR" -s dqeu89yh7xso8dq4 --video-source=camera --camera-id=0 --camera-size=640x480 --camera-fps=30 --v4l2-sink=/dev/video11 --no-window --no-audio >/tmp/cam_right.log 2>&1 &
PID_R=$!

sleep 2

# Check if processes are running
if ps -p $PID_L > /dev/null && ps -p $PID_R > /dev/null; then
    echo ""
    echo "=========================================================="
    echo " SUCCESS: Dual USB Cameras are ACTIVE!"
    echo "   Left Camera  -> /dev/video10 (PID: $PID_L)"
    echo "   Right Camera -> /dev/video11 (PID: $PID_R)"
    echo "=========================================================="
    echo "To stop cameras later: pkill -f 'scrcpy.*v4l2-sink'"
else
    echo ""
    echo "WARNING: One or both streams failed to start. Logs:"
    echo "--- Left Log (/tmp/cam_left.log) ---"
    cat /tmp/cam_left.log 2>/dev/null || true
    echo "--- Right Log (/tmp/cam_right.log) ---"
    cat /tmp/cam_right.log 2>/dev/null || true
    exit 1
fi
