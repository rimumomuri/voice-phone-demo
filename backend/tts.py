import os
import tempfile

TTS_ENGINE = os.getenv("TTS_ENGINE", "openai")

_ENGINES = {
    "voxcpm": {"name": "VoxCPM2",   "speed": "slow",   "cloning": True},
    "f5tts":  {"name": "F5-TTS",    "speed": "fast",   "cloning": True},
    "xtts":   {"name": "XTTS v2",   "speed": "medium", "cloning": True},
    "openai": {"name": "OpenAI TTS","speed": "fast",   "cloning": False},
}


def list_engines() -> list:
    result = []
    for eid, meta in _ENGINES.items():
        available = _check_engine(eid)
        result.append({**meta, "id": eid, "available": available})
    return result


def _check_engine(engine_id: str) -> bool:
    try:
        if engine_id == "voxcpm":
            import voxcpm  # noqa
        elif engine_id == "f5tts":
            import f5_tts  # noqa
        elif engine_id == "xtts":
            from TTS.api import TTS  # noqa
        elif engine_id == "openai":
            pass  # always available
        return True
    except ImportError:
        return False


def synthesize(text: str, reference_wav_path: str, reference_text: str = None,
               engine: str = None) -> bytes:
    eng = engine or TTS_ENGINE
    if eng == "voxcpm":
        return _synthesize_voxcpm(text, reference_wav_path, reference_text)
    if eng == "f5tts":
        return _synthesize_f5tts(text, reference_wav_path, reference_text)
    if eng == "xtts":
        return _synthesize_xtts(text, reference_wav_path)
    return _synthesize_openai(text)


# ── VoxCPM2 ──────────────────────────────────────────────────────────────────

_voxcpm_model = None


def _get_voxcpm():
    global _voxcpm_model
    if _voxcpm_model is None:
        try:
            from voxcpm import VoxCPM
            _voxcpm_model = VoxCPM.from_pretrained()
        except Exception as e:
            raise RuntimeError(f"VoxCPM2 モデルロード失敗: {e}") from e
    return _voxcpm_model


def _synthesize_voxcpm(text: str, reference_wav_path: str, reference_text: str = None) -> bytes:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError(f"必須パッケージなし: {e}") from e

    model = _get_voxcpm()
    timesteps = int(os.getenv("VOXCPM_TIMESTEPS", "10"))
    try:
        if reference_text:
            audio = model.generate(
                text=text,
                prompt_wav_path=reference_wav_path,
                prompt_text=reference_text,
                inference_timesteps=timesteps,
                denoise=False,
            )
        else:
            audio = model.generate(text=text, inference_timesteps=timesteps, denoise=False)
    except Exception as e:
        raise RuntimeError(f"VoxCPM 生成失敗: {e}") from e

    if isinstance(audio, tuple):
        audio, sr = audio
    else:
        sr = model.tts_model.sample_rate

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        sf.write(tmp, audio, sr)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp)


# ── F5-TTS ───────────────────────────────────────────────────────────────────

_f5tts_model = None


def _get_f5tts():
    global _f5tts_model
    if _f5tts_model is None:
        try:
            from f5_tts.api import F5TTS
            _f5tts_model = F5TTS()
        except Exception as e:
            raise RuntimeError(f"F5-TTS モデルロード失敗: {e}") from e
    return _f5tts_model


def _synthesize_f5tts(text: str, reference_wav_path: str, reference_text: str = None) -> bytes:
    try:
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError(f"soundfile なし: {e}") from e

    model = _get_f5tts()
    try:
        wav, sr, _ = model.infer(
            ref_file=reference_wav_path,
            ref_text=reference_text or "",
            gen_text=text,
            seed=-1,
        )
    except Exception as e:
        raise RuntimeError(f"F5-TTS 生成失敗: {e}") from e

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        sf.write(tmp, wav, sr)
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp)


# ── XTTS v2 ──────────────────────────────────────────────────────────────────

_xtts_model = None


def _get_xtts():
    global _xtts_model
    if _xtts_model is None:
        try:
            import torch
            from TTS.api import TTS
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        except Exception as e:
            raise RuntimeError(f"XTTS v2 モデルロード失敗: {e}") from e
    return _xtts_model


def _synthesize_xtts(text: str, reference_wav_path: str) -> bytes:
    model = _get_xtts()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        model.tts_to_file(
            text=text,
            speaker_wav=reference_wav_path,
            language="ja",
            file_path=tmp,
        )
        with open(tmp, "rb") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"XTTS v2 生成失敗: {e}") from e
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ── OpenAI TTS ────────────────────────────────────────────────────────────────

def _synthesize_openai(text: str) -> bytes:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(f"openai パッケージなし: {e}") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "sk-...":
        raise RuntimeError("OPENAI_API_KEY が設定されていません。")

    client = OpenAI(api_key=api_key)
    response = client.audio.speech.create(
        model="tts-1", voice="nova", input=text, response_format="wav"
    )
    return response.content
