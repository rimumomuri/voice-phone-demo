import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "test-api-key")

SYSTEM_PROMPT = """あなたは親切な日本語の受付AIです。
必ず1〜2文以内で簡潔にお答えください。長い説明は避けてください。"""

Message = dict  # {"role": str, "content": str}


def chat(history: list[Message], user_text: str) -> tuple[str, list[Message]]:
    updated = history + [{"role": "user", "content": user_text}]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + updated,
        max_tokens=80,
    )
    reply = response.choices[0].message.content
    updated = updated + [{"role": "assistant", "content": reply}]
    return reply, updated
