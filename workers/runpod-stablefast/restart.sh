#!/bin/bash
# Restart script for vj-server on RunPod

echo "=== Stopping existing processes ==="
pkill -f inference_server.py
pkill node
sleep 1

echo "=== Starting server ==="
cd /workspace/vj-server
node server.js
