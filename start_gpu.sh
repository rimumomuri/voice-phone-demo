#!/bin/bash
# GPUSOROBAN GPU 서버 실행 스크립트
# 실행: bash start_gpu.sh [포트번호 (기본값: 8080)]

set -e

PORT=${1:-8080}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

# .env 확인
if [ ! -f "$BACKEND_DIR/.env" ]; then
    echo "ERROR: backend/.env 가 없습니다."
    echo "  → bash setup.sh 를 먼저 실행하거나, backend/.env.example 을 복사해 OPENAI_API_KEY 를 입력하세요."
    exit 1
fi

# OPENAI_API_KEY 확인
source "$BACKEND_DIR/.env" 2>/dev/null || true
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "sk-..." ]; then
    echo "ERROR: OPENAI_API_KEY 가 설정되지 않았습니다."
    echo "  → nano backend/.env 에서 실제 키를 입력하세요."
    exit 1
fi

# venv 확인
if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: .venv 가 없습니다. bash setup.sh 를 먼저 실행하세요."
    exit 1
fi

echo "================================================"
echo " Voice Demo 서버 시작 (GPU / VoxCPM2 모드)"
echo " Port : $PORT"
echo " ※ 첫 실행 시 VoxCPM2 모델을 다운로드합니다 (수 GB, 5~10분)"
echo "================================================"
echo ""

# witts-sol.com 도메인 연결 (역방향 SSH 터널)
VPS_HOST="root@153.122.4.97"
VPS_PORT=8889
if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -q "$VPS_HOST" exit 2>/dev/null; then
    echo " → witts-sol.com 터널 시작 (localhost:$PORT → VPS:$VPS_PORT)"
    nohup bash -c "while true; do
        ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
            -N -R ${VPS_PORT}:localhost:${PORT} ${VPS_HOST}
        sleep 5
    done" > /tmp/tunnel.log 2>&1 &
    echo " → 터널 PID: $!"
else
    echo " → VPS 연결 불가 (터널 없이 실행)"
fi

cd "$BACKEND_DIR"
TTS_ENGINE=voxcpm "$VENV_PYTHON" -m uvicorn main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1
