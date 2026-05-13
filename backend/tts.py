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
