import base64
import io
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key or _api_key == "sk-...":
    print("ERROR: OPENAI_API_KEY が設定されていません。", file=sys.stderr)
    sys.exit(1)

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

from stt import transcribe
from llm import chat, Message
from tts import synthesize

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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


# ── 분리된 엔드포인트 ─────────────────────────────────────────────────────────

@app.post("/stt")
async def stt_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    t0 = time.time()
    text = transcribe(audio_bytes, file.filename or "audio.webm")
    elapsed = round(time.time() - t0, 2)
    if not text.strip():
        raise HTTPException(status_code=422, detail="音声を認識できませんでした。")
    return JSONResponse({"text": text, "elapsed": elapsed})


@app.post("/llm")
async def llm_endpoint(body: dict = Body(...)):
    global conversation_history
    user_text = body.get("text", "")
    if not user_text.strip():
        raise HTTPException(status_code=422, detail="テキストが空です。")
    t0 = time.time()
    reply_text, conversation_history = chat(conversation_history, user_text)
    elapsed = round(time.time() - t0, 2)
    return JSONResponse({"reply": reply_text, "elapsed": elapsed})


@app.post("/tts")
async def tts_endpoint(body: dict = Body(...)):
    text = body.get("text", "")
    if not text.strip():
        raise HTTPException(status_code=422, detail="テキストが空です。")
    if not REFERENCE_WAV.exists():
        raise HTTPException(status_code=400, detail="声が登録されていません。")
    ref_text = REFERENCE_TXT.read_text(encoding="utf-8") if REFERENCE_TXT.exists() else None
    t0 = time.time()
    try:
        wav_bytes = synthesize(text, str(REFERENCE_WAV), ref_text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = round(time.time() - t0, 2)
    audio_b64 = base64.b64encode(wav_bytes).decode()
    return JSONResponse({"audio_base64": audio_b64, "elapsed": elapsed})


# ── 기존 /chat (호환성 유지) ──────────────────────────────────────────────────

@app.post("/chat")
async def chat_endpoint(file: UploadFile = File(...)):
    global conversation_history
    if not REFERENCE_WAV.exists():
        raise HTTPException(status_code=400, detail="声が登録されていません。先に声を登録してください。")
    audio_bytes = await file.read()
    t0 = time.time()
    user_text = transcribe(audio_bytes, file.filename or "audio.webm")
    t1 = time.time()
    if not user_text.strip():
        raise HTTPException(status_code=422, detail="音声を認識できませんでした。もう一度お試しください。")
    reply_text, conversation_history = chat(conversation_history, user_text)
    t2 = time.time()
    ref_text = REFERENCE_TXT.read_text(encoding="utf-8") if REFERENCE_TXT.exists() else None
    try:
        wav_bytes = synthesize(reply_text, str(REFERENCE_WAV), ref_text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    t3 = time.time()
    audio_b64 = base64.b64encode(wav_bytes).decode()
    return JSONResponse({
        "user_text": user_text,
        "reply_text": reply_text,
        "audio_base64": audio_b64,
        "timing": {
            "stt": round(t1 - t0, 2),
            "llm": round(t2 - t1, 2),
            "tts": round(t3 - t2, 2),
        }
    })


@app.get("/")
def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
