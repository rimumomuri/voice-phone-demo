import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock


def _make_reference_wav(path: str):
    import soundfile as sf
    silence = np.zeros(24000, dtype=np.float32)
    sf.write(path, silence, 24000)


def test_synthesize_openai_returns_bytes(monkeypatch):
    monkeypatch.setenv("TTS_ENGINE", "openai")
    import importlib, tts as tts_mod
    importlib.reload(tts_mod)

    mock_client = MagicMock()
    mock_client.audio.speech.create.return_value = MagicMock(content=b"RIFF\x00\x00\x00\x00WAVEfmt ")

    with patch("openai.OpenAI", return_value=mock_client):
        result = tts_mod._synthesize_openai("テストです。")

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_synthesize_voxcpm_calls_model(monkeypatch):
    monkeypatch.setenv("TTS_ENGINE", "voxcpm")

    with patch("tts._get_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.generate.return_value = np.zeros(24000, dtype=np.float32)
        mock_get_model.return_value = mock_model

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref_path = f.name
        try:
            _make_reference_wav(ref_path)
            from tts import _synthesize_voxcpm
            result = _synthesize_voxcpm("こんにちは。", ref_path)
            assert isinstance(result, bytes)
            assert len(result) > 44
        finally:
            os.unlink(ref_path)
