import os
import tempfile

# TTS_ENGINE=voxcpm → VoxCPM2 voice cloning (requires GPU)
# TTS_ENGINE=openai  → OpenAI TTS API (server default, no GPU needed)
TTS_ENGINE = os.getenv("TTS_ENGINE", "openai")


def synthesize(text: str, reference_wav_path: str) -> bytes:
    if TTS_ENGINE == "voxcpm":
        return _synthesize_voxcpm(text, reference_wav_path)
    return _synthesize_openai(text)


# ── VoxCPM2 (local, GPU required) ────────────────────────────────────────────

def _synthesize_voxcpm(text: str, reference_wav_path: str) -> bytes:
    import numpy as np
    import soundfile as sf
    from voxcpm import VoxCPM

    model = _get_model()
    audio = model.generate(text=text, reference_wav_path=reference_wav_path)

    if isinstance(audio, tuple):
        audio, sr = audio
    else:
        sr = 24000

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        sf.write(tmp_path, audio, sr)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


_voxcpm_model = None

def _get_model():
    global _voxcpm_model
    if _voxcpm_model is None:
        from voxcpm import VoxCPM
        _voxcpm_model = VoxCPM()
    return _voxcpm_model


# ── OpenAI TTS (server, no GPU) ───────────────────────────────────────────────

def _synthesize_openai(text: str) -> bytes:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "test-api-key")
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
        response_format="wav"
    )
    return response.content
