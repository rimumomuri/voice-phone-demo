import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock


def _make_reference_wav(path: str):
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
