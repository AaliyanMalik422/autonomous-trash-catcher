#!/usr/bin/env bash
pkill -f "scrcpy.*v4l2-sink" 2>/dev/null || true
echo "Dual camera streams stopped."
