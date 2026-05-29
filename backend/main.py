import base64
import hashlib
import io
import json
import os
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key or _api_key == "sk-...":
    print("ERROR: OPENAI_API_KEY が設定されていません。", file=sys.stderr)
    sys.exit(1)

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

from stt import transcribe
from llm import chat, Message
from tts import synthesize, list_engines

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BACKEND_DIR = Path(__file__).parent
VOICES_DIR = BACKEND_DIR / "reference_voices"
VOICES_DIR.mkdir(exist_ok=True)
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"

# Migrate old single-voice storage → new multi-voice format (runs once)
_old_wav = BACKEND_DIR / "reference_voice" / "user.wav"
if _old_wav.exists():
    _mig_dir = VOICES_DIR / "default"
    _mig_dir.mkdir(exist_ok=True)
    if not (_mig_dir / "voice.wav").exists():
        shutil.copy2(_old_wav, _mig_dir / "voice.wav")
        _old_txt = _old_wav.parent / "user.txt"
        if _old_txt.exists():
            shutil.copy2(_old_txt, _mig_dir / "voice.txt")
        meta = {"name": "デフォルト", "created_at": datetime.now().isoformat()}
        (_mig_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))

conversation_history: list[Message] = []


def _tts_cache_path(voice_dir: Path, text: str, engine: str = "default") -> Path:
    cache_dir = voice_dir / "cache" / engine
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{text_hash}.wav"


def _get_voice_dir(voice_id: Optional[str]) -> Optional[Path]:
    if voice_id:
        d = VOICES_DIR / voice_id
        if d.is_dir() and (d / "voice.wav").exists():
            return d
    # fallback: most recently modified voice
    dirs = [d for d in VOICES_DIR.iterdir() if d.is_dir() and (d / "voice.wav").exists()]
    if dirs:
        return max(dirs, key=lambda p: p.stat().st_mtime)
    # final fallback: old single-voice location
    old = BACKEND_DIR / "reference_voice" / "user.wav"
    if old.exists():
        return old.parent
    return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tts/engines")
def get_engines():
    return JSONResponse(list_engines())


# ── Voice management ──────────────────────────────────────────────────────────

@app.get("/voices")
def list_voices():
    voices = []
    for d in VOICES_DIR.iterdir():
        if not d.is_dir() or not (d / "voice.wav").exists():
            continue
        meta_path = d / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        voices.append({
            "id": d.name,
            "name": meta.get("name", d.name),
            "created_at": meta.get("created_at", ""),
        })
    voices.sort(key=lambda v: v["created_at"], reverse=True)
    return JSONResponse(voices)


@app.get("/voices/{voice_id}/audio")
def get_voice_audio(voice_id: str):
    wav_path = VOICES_DIR / voice_id / "voice.wav"
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="Voice not found")
    return FileResponse(str(wav_path), media_type="audio/wav")


@app.post("/register-voice")
async def register_voice(file: UploadFile = File(...), name: str = Form("声")):
    audio_bytes = await file.read()
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"音声ファイルの解析に失敗しました: {e}")
    audio = audio.set_frame_rate(24000).set_channels(1)

    voice_id = uuid.uuid4().hex[:8]
    voice_dir = VOICES_DIR / voice_id
    voice_dir.mkdir(exist_ok=True)
    audio.export(str(voice_dir / "voice.wav"), format="wav")

    ref_text = transcribe(audio_bytes, file.filename or "voice.webm")
    (voice_dir / "voice.txt").write_text(ref_text, encoding="utf-8")

    label = name.strip() or "声"
    meta = {"name": label, "created_at": datetime.now().isoformat()}
    (voice_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))

    return JSONResponse({"id": voice_id, "name": meta["name"], "created_at": meta["created_at"]})


@app.delete("/voices/{voice_id}")
def delete_voice(voice_id: str):
    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail="Voice not found")
    shutil.rmtree(voice_dir)
    return JSONResponse({"status": "deleted"})


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
    voice_id = body.get("voice_id")
    engine = body.get("engine") or os.getenv("TTS_ENGINE", "openai")
    if not text.strip():
        raise HTTPException(status_code=422, detail="テキストが空です。")
    voice_dir = _get_voice_dir(voice_id)
    if voice_dir is None:
        raise HTTPException(status_code=400, detail="声が登録されていません。")

    cache_file = _tts_cache_path(voice_dir, text, engine)
    if cache_file.exists():
        audio_b64 = base64.b64encode(cache_file.read_bytes()).decode()
        return JSONResponse({"audio_base64": audio_b64, "elapsed": 0.0, "cached": True, "engine": engine})

    ref_text = (voice_dir / "voice.txt").read_text(encoding="utf-8") if (voice_dir / "voice.txt").exists() else None
    t0 = time.time()
    try:
        wav_bytes = synthesize(text, str(voice_dir / "voice.wav"), ref_text, engine=engine)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = round(time.time() - t0, 2)
    cache_file.write_bytes(wav_bytes)

    audio_b64 = base64.b64encode(wav_bytes).decode()
    return JSONResponse({"audio_base64": audio_b64, "elapsed": elapsed, "cached": False, "engine": engine})


def _do_warm_cache(voice_dir: Path, texts: list, engine: str):
    ref_text = (voice_dir / "voice.txt").read_text(encoding="utf-8") if (voice_dir / "voice.txt").exists() else None
    for text in texts:
        if not text.strip():
            continue
        cache_file = _tts_cache_path(voice_dir, text, engine)
        if cache_file.exists():
            continue
        try:
            wav_bytes = synthesize(text, str(voice_dir / "voice.wav"), ref_text, engine=engine)
            cache_file.write_bytes(wav_bytes)
        except Exception as e:
            print(f"warm-cache [{engine}] failed: {e}")


@app.post("/voices/{voice_id}/warm-cache")
def warm_cache_endpoint(voice_id: str, body: dict = Body(...)):
    texts = body.get("texts", [])
    engine = body.get("engine") or os.getenv("TTS_ENGINE", "openai")
    voice_dir = _get_voice_dir(voice_id)
    if voice_dir is None:
        raise HTTPException(status_code=404, detail="Voice not found")
    threading.Thread(target=_do_warm_cache, args=(voice_dir, texts[:30], engine), daemon=True).start()
    return JSONResponse({"status": "warming", "count": len(texts), "engine": engine})


@app.post("/tts/cache-check")
def tts_cache_check(body: dict = Body(...)):
    voice_id = body.get("voice_id")
    texts = body.get("texts", [])
    engine = body.get("engine") or os.getenv("TTS_ENGINE", "openai")
    voice_dir = _get_voice_dir(voice_id)
    if voice_dir is None:
        return JSONResponse({"ready": []})
    ready = [t for t in texts if _tts_cache_path(voice_dir, t, engine).exists()]
    return JSONResponse({"ready": ready})


# ── 기존 /chat (호환성 유지) ──────────────────────────────────────────────────

@app.post("/chat")
async def chat_endpoint(file: UploadFile = File(...)):
    global conversation_history
    audio_bytes = await file.read()
    voice_dir = _get_voice_dir(None)
    if voice_dir is None:
        raise HTTPException(status_code=400, detail="声が登録されていません。先に声を登録してください。")
    t0 = time.time()
    user_text = transcribe(audio_bytes, file.filename or "audio.webm")
    t1 = time.time()
    if not user_text.strip():
        raise HTTPException(status_code=422, detail="音声を認識できませんでした。もう一度お試しください。")
    reply_text, conversation_history = chat(conversation_history, user_text)
    t2 = time.time()
    ref_text = (voice_dir / "voice.txt").read_text(encoding="utf-8") if (voice_dir / "voice.txt").exists() else None
    try:
        wav_bytes = synthesize(reply_text, str(voice_dir / "voice.wav"), ref_text)
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
