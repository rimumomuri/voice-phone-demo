#!/bin/bash
# 로컬 개발 서버 + ngrok 동시 실행

PORT=8080
cd "$(dirname "$0")/backend"

echo "Starting FastAPI server on port $PORT..."
TTS_ENGINE=voxcpm ../.venv/bin/python3.11 -m uvicorn main:app --reload --port $PORT &
SERVER_PID=$!

sleep 2
echo "Starting ngrok tunnel..."
ngrok http $PORT --log=stdout &
NGROK_PID=$!

echo ""
echo "Server PID: $SERVER_PID"
echo "ngrok PID:  $NGROK_PID"
echo ""
echo "Local:  http://localhost:$PORT"
echo "Public: Check ngrok dashboard at http://localhost:4040"
echo ""
echo "Press Ctrl+C to stop all."

trap "kill $SERVER_PID $NGROK_PID 2>/dev/null; exit" INT
wait
