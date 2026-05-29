import os
import tempfile

# TTS_ENGINE=voxcpm → VoxCPM2 음성 클로닝 (CUDA GPU 필수)
# TTS_ENGINE=openai  → OpenAI TTS API (GPU 불필요, 서버 기본값)
TTS_ENGINE = os.getenv("TTS_ENGINE", "openai")


def synthesize(text: str, reference_wav_path: str, reference_text: str = None) -> bytes:
    if TTS_ENGINE == "voxcpm":
        return _synthesize_voxcpm(text, reference_wav_path, reference_text)
    return _synthesize_openai(text)


# ── VoxCPM2 (로컬 GPU 필수) ───────────────────────────────────────────────────

def _synthesize_voxcpm(text: str, reference_wav_path: str, reference_text: str = None) -> bytes:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError(f"필수 패키지 없음: {e}. pip install numpy soundfile") from e

    try:
        from voxcpm import VoxCPM
    except ImportError as e:
        raise RuntimeError(f"voxcpm 패키지 없음: {e}. pip install voxcpm") from e

    model = _get_model()

    try:
        if reference_text:
            audio = model.generate(text=text, prompt_wav_path=reference_wav_path, prompt_text=reference_text)
        else:
            audio = model.generate(text=text)
    except Exception as e:
        raise RuntimeError(f"VoxCPM 생성 실패: {e}") from e

    # generate()는 tuple이 아닌 array만 반환 → SR은 model에서 가져옴
    if isinstance(audio, tuple):
        audio, sr = audio
    else:
        sr = model.tts_model.sample_rate

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
        try:
            from voxcpm import VoxCPM
            _voxcpm_model = VoxCPM.from_pretrained()
        except Exception as e:
            raise RuntimeError(
                f"VoxCPM2 모델 로드 실패: {e}\n"
                "CUDA GPU 환경인지, voxcpm 패키지가 설치되어 있는지 확인하세요."
            ) from e
    return _voxcpm_model


# ── OpenAI TTS (GPU 불필요) ───────────────────────────────────────────────────

def _synthesize_openai(text: str) -> bytes:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(f"openai 패키지 없음: {e}. pip install openai") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "sk-...":
        raise RuntimeError("OPENAI_API_KEY 가 설정되지 않았습니다. backend/.env 를 확인하세요.")

    client = OpenAI(api_key=api_key)
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
        response_format="wav"
    )
    return response.content
