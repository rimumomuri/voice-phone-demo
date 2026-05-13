import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "test-api-key")

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
