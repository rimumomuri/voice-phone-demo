import io
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "test-api-key")


def transcribe(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="ja"
    )
    return result.text
