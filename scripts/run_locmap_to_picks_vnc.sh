#!/usr/bin/env bash
set -euo pipefail

# Minimal helper to run locmap_to_picks_gui.py inside a virtual X server (Xvfb) and expose it via VNC.
# Prereqs (install yourself): apt-get install -y xvfb x11vnc
# Usage:
#   ./scripts/run_locmap_to_picks_vnc.sh --locmap pred_locmaps/sample_pred_locmaps.pt --prompt prompt_1 --output picks.tsv [--tomo path/to/volume.mrc]
# Then forward port 5901 from the container and connect with a VNC client to localhost:5901
# Set VNC_PASSWORD to require a password; set QT_QPA_PLATFORM=offscreen if xcb fails.

DISPLAY_ID="${DISPLAY:-:1}"
VNC_PORT="${VNC_PORT:-5901}"
VNC_PASSWORD="${VNC_PASSWORD:-}"
# Default to xcb for interactive VNC; override with QT_QPA_PLATFORM=offscreen if xcb fails.
QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

start_xvfb() {
  if pgrep -f "Xvfb ${DISPLAY_ID}" >/dev/null 2>&1; then
    echo "Xvfb already running on ${DISPLAY_ID}"
  else
    echo "Starting Xvfb on ${DISPLAY_ID}..."
    Xvfb "${DISPLAY_ID}" -screen 0 1920x1080x24 >/tmp/xvfb.log 2>&1 &
    sleep 1
  fi
}

start_vnc() {
  if pgrep -f "x11vnc.*${DISPLAY_ID}" >/dev/null 2>&1; then
    echo "x11vnc already running on ${DISPLAY_ID}"
  else
    echo "Starting x11vnc on ${DISPLAY_ID}, port ${VNC_PORT} (localhost only)..."
    if [ -n "${VNC_PASSWORD}" ]; then
      x11vnc -display "${DISPLAY_ID}" -localhost -shared -forever -passwd "${VNC_PASSWORD}" -rfbport "${VNC_PORT}" >/tmp/x11vnc.log 2>&1 &
    else
      x11vnc -display "${DISPLAY_ID}" -localhost -shared -forever -nopw -rfbport "${VNC_PORT}" >/tmp/x11vnc.log 2>&1 &
    fi
    sleep 1
  fi
}

start_xvfb
start_vnc

echo "Launching locmap_to_picks_gui.py on ${DISPLAY_ID} (VNC port ${VNC_PORT}) with QT_QPA_PLATFORM=${QPA_PLATFORM}..."
QT_API=pyside6 QT_QPA_PLATFORM="${QPA_PLATFORM}" QT_OPENGL=software DISPLAY="${DISPLAY_ID}" python locmap_to_picks_gui.py "$@"
