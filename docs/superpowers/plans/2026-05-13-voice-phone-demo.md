# Voice Phone Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ブラウザで話しかけると、最初に録音した声でAIが日本語で返答するデモアプリを構築する。

**Architecture:** Python FastAPIバックエンドがSTT(Whisper API)→LLM(GPT-4o-mini)→TTS(VoxCPM2 MPS)のパイプラインを処理する。シングルページHTMLフロントエンドがMediaRecorder APIで音声を録音してバックエンドに送り、AIの音声レスポンスを再生する。

**Tech Stack:** Python 3.11, FastAPI, VoxCPM2, OpenAI SDK (Whisper + GPT-4o-mini), pydub, soundfile, Vanilla JS

---

## ファイル構成

```
voice-phone-demo/
├── backend/
│   ├── main.py              # FastAPI サーバー・エントリーポイント
│   ├── stt.py               # Whisper API ラッパー (音声→テキスト)
│   ├── llm.py               # GPT-4o-mini ラッパー (テキスト→返答)
│   ├── tts.py               # VoxCPM2 ラッパー (テキスト→音声)
│   └── reference_voice/     # 登録された声サンプル保存先
│       └── .gitkeep
├── frontend/
│   └── index.html           # 単一ページUI
├── tests/
│   ├── conftest.py          # pytest パス設定
│   ├── test_stt.py
│   ├── test_llm.py
│   └── test_tts.py
├── requirements.txt
├── .env.example
└── .env                     # (gitignore対象)
```

---

### Task 1: プロジェクトセットアップ

**Files:**
- Create: `voice-phone-demo/requirements.txt`
- Create: `voice-phone-demo/.env.example`
- Create: `voice-phone-demo/.gitignore`
- Create: `voice-phone-demo/backend/reference_voice/.gitkeep`
- Create: `voice-phone-demo/tests/conftest.py`

- [ ] **Step 1: ディレクトリ作成**

```bash
cd /Users/rymchoi/workspace
mkdir -p voice-phone-demo/backend/reference_voice
mkdir -p voice-phone-demo/frontend
mkdir -p voice-phone-demo/tests
touch voice-phone-demo/backend/reference_voice/.gitkeep
```

- [ ] **Step 2: requirements.txt を作成**

`voice-phone-demo/requirements.txt`:
```
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
openai==1.40.0
voxcpm
soundfile==0.12.1
numpy==1.26.4
pydub==0.25.1
python-dotenv==1.0.0
pytest==8.3.0
httpx==0.27.0
```

- [ ] **Step 3: .env.example を作成**

`voice-phone-demo/.env.example`:
```
OPENAI_API_KEY=sk-...
```

- [ ] **Step 4: .gitignore を作成**

`voice-phone-demo/.gitignore`:
```
.env
__pycache__/
*.pyc
.venv/
backend/reference_voice/user.wav
```

- [ ] **Step 5: conftest.py を作成**

`voice-phone-demo/tests/conftest.py`:
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
```

- [ ] **Step 6: Python バージョン確認 + 仮想環境作成**

```bash
cd /Users/rymchoi/workspace/voice-phone-demo
python3 --version
# 3.10〜3.12 であることを確認 (3.13は voxcpm 非対応)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

VoxCPM2 モデルは初回実行時に自動ダウンロード（約数GB）。

- [ ] **Step 7: ffmpeg インストール確認**

```bash
ffmpeg -version
# インストールされていない場合:
brew install ffmpeg
```

pydub の webm→wav 変換に必要。

- [ ] **Step 8: .env 作成**

```bash
cp .env.example .env
# .env を開いて OPENAI_API_KEY を設定
```

- [ ] **Step 9: コミット**

```bash
cd /Users/rymchoi/workspace/voice-phone-demo
git init
git add requirements.txt .env.example .gitignore backend/reference_voice/.gitkeep tests/conftest.py
git commit -m "feat: project scaffolding"
```

---

### Task 2: STT モジュール (Whisper API)

**Files:**
- Create: `voice-phone-demo/backend/stt.py`
- Create: `voice-phone-demo/tests/test_stt.py`

- [ ] **Step 1: テストを書く**

`voice-phone-demo/tests/test_stt.py`:
```python
from unittest.mock import patch, MagicMock


def test_transcribe_returns_japanese_string():
    with patch('stt.client') as mock_client:
        mock_result = MagicMock()
        mock_result.text = "こんにちは、お電話ありがとうございます。"
        mock_client.audio.transcriptions.create.return_value = mock_result

        from stt import transcribe
        result = transcribe(b"fake-audio-bytes", "audio.webm")

        assert isinstance(result, str)
        assert result == "こんにちは、お電話ありがとうございます。"
        mock_client.audio.transcriptions.create.assert_called_once()


def test_transcribe_passes_language_ja():
    with patch('stt.client') as mock_client:
        mock_result = MagicMock()
        mock_result.text = "テスト"
        mock_client.audio.transcriptions.create.return_value = mock_result

        from stt import transcribe
        transcribe(b"audio", "audio.webm")

        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs.get('language') == 'ja'
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
cd /Users/rymchoi/workspace/voice-phone-demo
source .venv/bin/activate
pytest tests/test_stt.py -v
# Expected: ERROR (stt モジュールが存在しない)
```

- [ ] **Step 3: stt.py を実装**

`voice-phone-demo/backend/stt.py`:
```python
import io
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def transcribe(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="ja"
    )
    return result.text
```

- [ ] **Step 4: テストが通ることを確認**

```bash
pytest tests/test_stt.py -v
# Expected: 2 passed
```

- [ ] **Step 5: コミット**

```bash
git add backend/stt.py tests/test_stt.py
git commit -m "feat: add STT module (Whisper API)"
```

---

### Task 3: LLM モジュール (GPT-4o-mini)

**Files:**
- Create: `voice-phone-demo/backend/llm.py`
- Create: `voice-phone-demo/tests/test_llm.py`

- [ ] **Step 1: テストを書く**

`voice-phone-demo/tests/test_llm.py`:
```python
from unittest.mock import patch, MagicMock


def test_chat_returns_reply_and_updated_history():
    with patch('llm.client') as mock_client:
        mock_choice = MagicMock()
        mock_choice.message.content = "はい、承知しました。"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        from llm import chat
        reply, history = chat([], "こんにちは")

        assert reply == "はい、承知しました。"
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "こんにちは"}
        assert history[1] == {"role": "assistant", "content": "はい、承知しました。"}


def test_chat_appends_to_existing_history():
    with patch('llm.client') as mock_client:
        mock_choice = MagicMock()
        mock_choice.message.content = "かしこまりました。"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        from llm import chat
        existing = [
            {"role": "user", "content": "前の質問"},
            {"role": "assistant", "content": "前の返答"},
        ]
        reply, history = chat(existing, "次の質問")

        assert len(history) == 4
        assert history[2]["role"] == "user"
        assert history[3]["content"] == "かしこまりました。"


def test_chat_includes_system_prompt():
    with patch('llm.client') as mock_client:
        mock_choice = MagicMock()
        mock_choice.message.content = "返答"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        from llm import chat
        chat([], "質問")

        messages = mock_client.chat.completions.create.call_args.kwargs['messages']
        assert messages[0]['role'] == 'system'
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
pytest tests/test_llm.py -v
# Expected: ERROR (llm モジュールが存在しない)
```

- [ ] **Step 3: llm.py を実装**

`voice-phone-demo/backend/llm.py`:
```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """あなたは親切な日本語の受付AIです。
自然な会話で丁寧にお答えください。
回答は簡潔に2〜3文でお願いします。"""

Message = dict  # {"role": str, "content": str}


def chat(history: list[Message], user_text: str) -> tuple[str, list[Message]]:
    updated = history + [{"role": "user", "content": user_text}]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + updated
    )
    reply = response.choices[0].message.content
    updated = updated + [{"role": "assistant", "content": reply}]
    return reply, updated
```

- [ ] **Step 4: テストが通ることを確認**

```bash
pytest tests/test_llm.py -v
# Expected: 3 passed
```

- [ ] **Step 5: コミット**

```bash
git add backend/llm.py tests/test_llm.py
git commit -m "feat: add LLM module (GPT-4o-mini)"
```

---

### Task 4: TTS モジュール (VoxCPM2)

**Files:**
- Create: `voice-phone-demo/backend/tts.py`
- Create: `voice-phone-demo/tests/test_tts.py`

- [ ] **Step 1: テストを書く**

`voice-phone-demo/tests/test_tts.py`:
```python
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock


def _make_reference_wav(path: str):
    """テスト用のダミーWAVファイルを作成する。"""
    import soundfile as sf
    silence = np.zeros(24000, dtype=np.float32)
    sf.write(path, silence, 24000)


def test_synthesize_returns_bytes():
    with patch('tts._get_model') as mock_get_model:
        mock_model = MagicMock()
        mock_model.generate.return_value = np.zeros(24000, dtype=np.float32)
        mock_get_model.return_value = mock_model

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            ref_path = f.name
        try:
            _make_reference_wav(ref_path)
            from tts import synthesize
            result = synthesize("テストです。", ref_path)
            assert isinstance(result, bytes)
            assert len(result) > 44  # WAV header は44バイト
        finally:
            os.unlink(ref_path)


def test_synthesize_calls_model_with_text_and_reference():
    with patch('tts._get_model') as mock_get_model:
        mock_model = MagicMock()
        mock_model.generate.return_value = np.zeros(24000, dtype=np.float32)
        mock_get_model.return_value = mock_model

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            ref_path = f.name
        try:
            _make_reference_wav(ref_path)
            from tts import synthesize
            synthesize("こんにちは。", ref_path)
            mock_model.generate.assert_called_once_with(
                text="こんにちは。",
                reference_wav_path=ref_path
            )
        finally:
            os.unlink(ref_path)
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
pytest tests/test_tts.py -v
# Expected: ERROR (tts モジュールが存在しない)
```

- [ ] **Step 3: tts.py を実装**

`voice-phone-demo/backend/tts.py`:
```python
import os
import tempfile
import numpy as np
import soundfile as sf
from voxcpm import VoxCPM

SAMPLE_RATE = 24000

_model: VoxCPM | None = None


def _get_model() -> VoxCPM:
    global _model
    if _model is None:
        _model = VoxCPM()
    return _model


def synthesize(text: str, reference_wav_path: str) -> bytes:
    """Returns WAV bytes synthesized in the voice of the reference audio."""
    model = _get_model()
    audio = model.generate(text=text, reference_wav_path=reference_wav_path)

    # audio は numpy float32 array と想定
    if isinstance(audio, tuple):
        audio, sr = audio
    else:
        sr = SAMPLE_RATE

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        sf.write(tmp_path, audio, sr)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)
```

- [ ] **Step 4: テストが通ることを確認**

```bash
pytest tests/test_tts.py -v
# Expected: 2 passed
```

- [ ] **Step 5: VoxCPM2 動作確認 (実際のモデルロード)**

モデルの初回ダウンロードを確認するスモークテスト:
```bash
cd /Users/rymchoi/workspace/voice-phone-demo
source .venv/bin/activate
python3 -c "
from dotenv import load_dotenv
load_dotenv()
from backend.tts import _get_model
print('モデルをロード中...')
m = _get_model()
print('OK:', type(m))
"
# モデルが数GBダウンロードされる (初回のみ)
# Expected: OK: <class 'voxcpm.VoxCPM'>
```

- [ ] **Step 6: コミット**

```bash
git add backend/tts.py tests/test_tts.py
git commit -m "feat: add TTS module (VoxCPM2 voice cloning)"
```

---

### Task 5: FastAPI バックエンド

**Files:**
- Create: `voice-phone-demo/backend/main.py`

- [ ] **Step 1: main.py を実装**

`voice-phone-demo/backend/main.py`:
```python
import base64
import io
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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
```

- [ ] **Step 2: サーバーが起動することを確認**

```bash
cd /Users/rymchoi/workspace/voice-phone-demo/backend
source ../.venv/bin/activate
uvicorn main:app --reload --port 8080
# Expected: Uvicorn running on http://127.0.0.1:8080
# Ctrl+C で停止
```

- [ ] **Step 3: /health エンドポイントを確認**

```bash
curl http://localhost:8080/health
# Expected: {"status":"ok"}
```

- [ ] **Step 4: コミット**

```bash
git add backend/main.py
git commit -m "feat: add FastAPI server with /register-voice and /chat endpoints"
```

---

### Task 6: フロントエンド UI

**Files:**
- Create: `voice-phone-demo/frontend/index.html`

- [ ] **Step 1: index.html を実装**

`voice-phone-demo/frontend/index.html`:
```html
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI 電話対応デモ</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, sans-serif; background: #f0f2f5; color: #333; }
    .container { max-width: 600px; margin: 40px auto; padding: 0 20px; }
    h1 { font-size: 22px; margin-bottom: 24px; color: #1a1a2e; }
    .card { background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .card h2 { font-size: 16px; margin-bottom: 12px; color: #555; }
    blockquote { background: #f8f9fa; border-left: 3px solid #3498db; padding: 12px 16px; border-radius: 4px; font-style: italic; margin: 12px 0; line-height: 1.6; }
    button { padding: 12px 24px; font-size: 15px; border: none; border-radius: 8px; cursor: pointer; transition: all 0.2s; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-red { background: #e74c3c; color: white; }
    .btn-red:hover:not(:disabled) { background: #c0392b; }
    .btn-red.recording { animation: pulse 1s infinite; }
    .btn-green { background: #27ae60; color: white; margin-left: 8px; }
    .btn-green:hover:not(:disabled) { background: #219a52; }
    .btn-blue { background: #3498db; color: white; width: 100%; padding: 20px; font-size: 18px; }
    .btn-blue:hover:not(:disabled) { background: #2980b9; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
    #chat-log { min-height: 80px; max-height: 320px; overflow-y: auto; background: #f8f9fa; border-radius: 8px; padding: 12px; margin-bottom: 16px; }
    .msg { margin: 6px 0; padding: 8px 14px; border-radius: 18px; max-width: 85%; font-size: 14px; line-height: 1.5; }
    .msg-user { background: #3498db; color: white; margin-left: auto; border-bottom-right-radius: 4px; }
    .msg-ai { background: white; border: 1px solid #e0e0e0; border-bottom-left-radius: 4px; }
    .msg-wrap { display: flex; flex-direction: column; }
    #status { font-size: 13px; color: #7f8c8d; margin-bottom: 12px; min-height: 20px; }
    .hidden { display: none; }
    .step-label { font-size: 12px; font-weight: bold; color: #3498db; margin-bottom: 6px; }
  </style>
</head>
<body>
<div class="container">
  <h1>🎤 AI 電話対応デモ</h1>

  <!-- Step 1: 声の登録 -->
  <div class="card" id="step-register">
    <div class="step-label">STEP 1</div>
    <h2>声を登録する</h2>
    <p style="font-size:14px;color:#666;margin-bottom:8px;">下のテキストを読み上げながら録音してください（約10秒）:</p>
    <blockquote>「こんにちは。本日はお電話いただきありがとうございます。少々お待ちください。どのようなご用件でしょうか。」</blockquote>
    <div style="margin-top:16px;">
      <button class="btn-red" id="btn-rec" onclick="toggleRegister()">● 録音開始</button>
      <button class="btn-green hidden" id="btn-send-voice" onclick="submitVoice()">✓ 登録する</button>
    </div>
    <p id="reg-status" style="font-size:13px;color:#7f8c8d;margin-top:10px;"></p>
  </div>

  <!-- Step 2: 会話 -->
  <div class="card hidden" id="step-chat">
    <div class="step-label">STEP 2</div>
    <h2>話しかける</h2>
    <div class="msg-wrap" id="chat-log"></div>
    <div id="status">ボタンを押している間、話してください</div>
    <button class="btn-blue" id="btn-speak"
      onmousedown="startChat()" onmouseup="stopChat()"
      ontouchstart="startChat(event)" ontouchend="stopChat(event)">
      🎤 押して話す
    </button>
  </div>
</div>

<script>
  let recorder = null;
  let chunks = [];
  let regBlob = null;
  let chatStream = null;

  // ── Step 1: 声の登録 ──────────────────────────────────

  async function toggleRegister() {
    const btn = document.getElementById('btn-rec');
    if (recorder && recorder.state === 'recording') {
      recorder.stop();
      btn.textContent = '● 録音開始';
      btn.classList.remove('recording');
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = e => chunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      regBlob = new Blob(chunks, { type: 'audio/webm' });
      document.getElementById('btn-send-voice').classList.remove('hidden');
      document.getElementById('reg-status').textContent = '録音完了 ✓ 登録ボタンを押してください';
    };
    recorder.start();
    btn.textContent = '■ 録音停止';
    btn.classList.add('recording');
  }

  async function submitVoice() {
    if (!regBlob) return;
    document.getElementById('reg-status').textContent = '登録中...';
    const form = new FormData();
    form.append('file', regBlob, 'voice.webm');
    const res = await fetch('/register-voice', { method: 'POST', body: form });
    if (res.ok) {
      document.getElementById('step-register').classList.add('hidden');
      document.getElementById('step-chat').classList.remove('hidden');
    } else {
      document.getElementById('reg-status').textContent = 'エラーが発生しました。もう一度お試しください。';
    }
  }

  // ── Step 2: 会話 ──────────────────────────────────────

  async function startChat(e) {
    if (e) e.preventDefault();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chatStream = stream;
    chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = ev => chunks.push(ev.data);
    recorder.start();
    document.getElementById('btn-speak').textContent = '🔴 録音中... (離して送信)';
    document.getElementById('status').textContent = '話してください...';
  }

  async function stopChat(e) {
    if (e) e.preventDefault();
    if (!recorder || recorder.state !== 'recording') return;
    recorder.onstop = async () => {
      if (chatStream) chatStream.getTracks().forEach(t => t.stop());
      const blob = new Blob(chunks, { type: 'audio/webm' });
      await sendChat(blob);
    };
    recorder.stop();
  }

  async function sendChat(blob) {
    const btn = document.getElementById('btn-speak');
    btn.disabled = true;
    document.getElementById('status').textContent = 'AI が考え中...';
    const form = new FormData();
    form.append('file', blob, 'chat.webm');
    try {
      const res = await fetch('/chat', { method: 'POST', body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      addMsg(data.user_text, 'msg-user');
      addMsg(data.reply_text, 'msg-ai');
      document.getElementById('status').textContent = 'AI が返答中...';
      await playAudio(data.audio_base64);
    } catch (err) {
      document.getElementById('status').textContent = 'エラー: ' + err.message;
    } finally {
      btn.disabled = false;
      btn.textContent = '🎤 押して話す';
      document.getElementById('status').textContent = 'ボタンを押している間、話してください';
    }
  }

  function addMsg(text, cls) {
    const log = document.getElementById('chat-log');
    const div = document.createElement('div');
    div.className = 'msg ' + cls;
    div.textContent = text;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  function playAudio(b64) {
    return new Promise(resolve => {
      const bytes = atob(b64);
      const arr = new Uint8Array(bytes.length);
      for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
      const url = URL.createObjectURL(new Blob([arr], { type: 'audio/wav' }));
      const audio = new Audio(url);
      audio.onended = () => { URL.revokeObjectURL(url); resolve(); };
      audio.onerror = resolve;
      audio.play();
    });
  }
</script>
</body>
</html>
```

- [ ] **Step 2: サーバーを起動してブラウザで確認**

```bash
cd /Users/rymchoi/workspace/voice-phone-demo/backend
source ../.venv/bin/activate
uvicorn main:app --reload --port 8080
# ブラウザで http://localhost:8080 を開く
```

- [ ] **Step 3: 動作確認チェックリスト (手動)**

```
□ Step 1 カードが表示される
□ 「録音開始」ボタンで録音できる
□ 「■ 録音停止」で停止し「登録する」ボタンが出る
□ 「登録する」でStep 2に遷移する
□ 「押して話す」を押しながら日本語で話す
□ 離すとスピナー → チャットログに文字起こしが表示
□ AIが録音した声で日本語を返答する
```

- [ ] **Step 4: コミット**

```bash
git add frontend/index.html
git commit -m "feat: add frontend UI with voice registration and chat"
```

---

### Task 7: 全テスト実行 + 最終確認

- [ ] **Step 1: 全ユニットテストを実行**

```bash
cd /Users/rymchoi/workspace/voice-phone-demo
source .venv/bin/activate
pytest tests/ -v
# Expected: 7 passed
```

- [ ] **Step 2: エンドツーエンド動作確認**

```bash
# サーバー起動
cd backend && uvicorn main:app --port 8080

# 別ターミナルで /health 確認
curl http://localhost:8080/health
# Expected: {"status":"ok"}
```

ブラウザで `http://localhost:8080` を開き、声を登録して日本語で話しかける。

- [ ] **Step 3: 最終コミット**

```bash
git add -A
git commit -m "feat: voice phone demo complete"
```

---

## トラブルシューティング

| 問題 | 対処 |
|------|------|
| `voxcpm` インストールエラー | Python 3.13 は非対応。`python3.11 -m venv .venv` で作り直す |
| pydub 変換エラー | `brew install ffmpeg` を実行 |
| マイク権限エラー | ブラウザのマイクアクセスを許可。`http://localhost` は許可されるが `http://127.0.0.1` は不可の場合あり |
| VoxCPM2 モデルDL失敗 | HuggingFace に接続できる環境で実行。VPN があれば切る |
| 声クローニング品質が低い | 録音時の背景音を減らす。より明確に読み上げる |
