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
