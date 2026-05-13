import base64
import io
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydub import AudioSegment

from stt import transcribe
from llm import chat, Message
from tts import synthesize

app = FastAPI()

BACKEND_DIR = Path(__file__).parent
REFERENCE_WAV = BACKEND_DIR / "reference_voice" / "user.wav"
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"

conversation_history: list[Message] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/register-voice")
async def register_voice(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_frame_rate(24000).set_channels(1)
    REFERENCE_WAV.parent.mkdir(exist_ok=True)
    audio.export(str(REFERENCE_WAV), format="wav")
    return {"status": "ok"}


@app.post("/chat")
async def chat_endpoint(file: UploadFile = File(...)):
    global conversation_history

    if not REFERENCE_WAV.exists():
        raise HTTPException(status_code=400, detail="Voice not registered yet")

    audio_bytes = await file.read()

    user_text = transcribe(audio_bytes, file.filename or "audio.webm")
    if not user_text.strip():
        raise HTTPException(status_code=422, detail="Could not transcribe audio")

    reply_text, conversation_history = chat(conversation_history, user_text)

    wav_bytes = synthesize(reply_text, str(REFERENCE_WAV))
    audio_b64 = base64.b64encode(wav_bytes).decode()

    return JSONResponse({
        "user_text": user_text,
        "reply_text": reply_text,
        "audio_base64": audio_b64
    })


@app.get("/")
def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
