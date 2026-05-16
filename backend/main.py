import base64
import io
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# 시작 시 필수 환경변수 확인
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key or _api_key == "sk-...":
    print("ERROR: OPENAI_API_KEY 가 설정되지 않았습니다.", file=sys.stderr)
    print("  backend/.env 에 OPENAI_API_KEY 를 입력하세요.", file=sys.stderr)
    sys.exit(1)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydub import AudioSegment

from stt import transcribe
from llm import chat, Message
from tts import synthesize

app = FastAPI()

BACKEND_DIR = Path(__file__).parent
REFERENCE_WAV = BACKEND_DIR / "reference_voice" / "user.wav"
REFERENCE_TXT = BACKEND_DIR / "reference_voice" / "user.txt"
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"

conversation_history: list[Message] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/register-voice")
async def register_voice(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"音声ファイルの解析に失敗しました: {e}")
    audio = audio.set_frame_rate(24000).set_channels(1)
    REFERENCE_WAV.parent.mkdir(exist_ok=True)
    audio.export(str(REFERENCE_WAV), format="wav")
    ref_text = transcribe(audio_bytes, file.filename or "voice.webm")
    REFERENCE_TXT.write_text(ref_text, encoding="utf-8")
    return {"status": "ok"}


@app.post("/chat")
async def chat_endpoint(file: UploadFile = File(...)):
    global conversation_history

    if not REFERENCE_WAV.exists():
        raise HTTPException(status_code=400, detail="声が登録されていません。先に声を登録してください。")

    audio_bytes = await file.read()

    user_text = transcribe(audio_bytes, file.filename or "audio.webm")
    if not user_text.strip():
        raise HTTPException(status_code=422, detail="音声を認識できませんでした。もう一度お試しください。")

    reply_text, conversation_history = chat(conversation_history, user_text)

    ref_text = REFERENCE_TXT.read_text(encoding="utf-8") if REFERENCE_TXT.exists() else None
    try:
        wav_bytes = synthesize(reply_text, str(REFERENCE_WAV), ref_text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    audio_b64 = base64.b64encode(wav_bytes).decode()

    return JSONResponse({
        "user_text": user_text,
        "reply_text": reply_text,
        "audio_base64": audio_b64
    })


@app.get("/")
def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
