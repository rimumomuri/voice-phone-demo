#!/bin/bash
# GPUSOROBAN GPU 서버 초기 환경 구성 스크립트
# 실행: bash setup.sh

set -e

echo "================================================"
echo " Voice Demo - GPUSOROBAN Setup"
echo "================================================"

# 1. 시스템 패키지
echo ""
echo "[1/4] 시스템 패키지 설치 중..."
sudo apt-get update -qq
sudo apt-get install -y ffmpeg libsndfile1 git curl

# 2. Python venv (--system-site-packages 로 서버 기설치 PyTorch/CUDA 상속)
echo ""
echo "[2/4] Python 가상환경 생성 중..."
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# pip 업그레이드
pip install --upgrade pip -q

# 3. Python 패키지 설치
echo ""
echo "[3/4] Python 패키지 설치 중..."
pip install -r requirements.gpu.txt -q

# torchcodec 은 torchaudio 버전에 따라 필요 여부가 다름 — 설치 실패해도 계속 진행
pip install torchcodec -q 2>/dev/null || echo "  (torchcodec 스킵 — 현재 torchaudio 버전에서 불필요)"

# 4. .env 파일 확인
echo ""
echo "[4/4] 환경변수 파일 확인 중..."
if [ ! -f backend/.env ]; then
    cp backend/.env.example backend/.env
    echo "  → backend/.env 생성 완료."
    echo "  !! OPENAI_API_KEY 를 반드시 입력하세요: nano backend/.env"
else
    echo "  → backend/.env 이미 존재합니다."
fi

# 5. reference_voice 디렉토리 생성
mkdir -p backend/reference_voice

echo ""
echo "================================================"
echo " 설정 완료!"
echo " 다음 단계:"
echo "   1. nano backend/.env  → OPENAI_API_KEY 입력"
echo "   2. bash start_gpu.sh  → 서버 시작"
echo "================================================"
