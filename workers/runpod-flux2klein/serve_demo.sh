#!/usr/bin/env bash
# Serves demo.html locally at http://localhost:8090/demo.html
# Pre-fills the signaling URL via query param.
#
# Usage:
#   ./serve_demo.sh           # uses default pod 3m90zbu8fwyyqk
#   ./serve_demo.sh <pod-id>  # uses a different pod
set -e
POD_ID="${1:-3m90zbu8fwyyqk}"
PORT="${PORT:-8090}"
SIG_URL="https://${POD_ID}-3000.proxy.runpod.net/webrtc/offer"
URL="http://localhost:${PORT}/demo.html?url=$(printf '%s' "$SIG_URL" | sed 's,/,%2F,g; s,:,%3A,g')"

cd "$(dirname "$0")"
echo "Serving at:    $URL"
echo "Signaling →:   $SIG_URL"
echo "Press Ctrl+C to stop"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
