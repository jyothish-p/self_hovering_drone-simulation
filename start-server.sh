#!/bin/bash
# ── RLT Project — Start Local Server ──────────────────────────────────────────
PORT=${1:-7892}
echo "🚀 Starting RLT Project server on http://localhost:$PORT"
echo "   Training UI:  http://localhost:$PORT/rl-training.html"
echo "   Dashboard:    http://localhost:$PORT/visualization/rl-dashboard.html"
echo "   Drone Demo:   http://localhost:$PORT/drone-demo.html"
echo ""
echo "Press Ctrl+C to stop."
cd "$(dirname "$0")"
npx -y http-server -p $PORT -c-1
