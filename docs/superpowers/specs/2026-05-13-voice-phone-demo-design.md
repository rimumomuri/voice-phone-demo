# 設計書: 日本語 AI 電話対応ブラウザデモ（自分声クローニング）

作成日: 2026-05-13

---

## 概要

ブラウザ上で日本語の会話型AIと話し、AIがユーザー自身の声で返答するデモアプリ。
VoxCPM2のControllable Cloningを使い、10秒程度の声サンプルから音声を複製する。

---

## アーキテクチャ

```
[ブラウザ]                         [ローカル Python サーバー (FastAPI)]
  1. 声サンプル録音 (10秒)
       ↓ WAV送信
                         → reference_voice.wav として保存
                                      (初回のみ)

  2. ユーザーが話す (マイク録音)
       ↓ WAV送信
                         → OpenAI Whisper API  → 日本語テキスト
                         → OpenAI GPT-4o-mini  → 日本語返答テキスト
                         → VoxCPM2 ANE (M chip) → ユーザー声で音声生成
       ↑ MP3/WAV受信して再生
```

### 予想レイテンシ
| ステップ | 時間 |
|--------|------|
| STT (Whisper API) | ~0.5–1s |
| LLM (GPT-4o-mini) | ~1–2s |
| TTS (VoxCPM2 ANE) | ~1–2s |
| **合計** | **約2〜5秒** |

---

## 技術スタック

| 役割 | 技術 | 理由 |
|------|------|------|
| フロントエンド | HTML + Vanilla JS | ビルド不要、シンプル |
| バックエンド | Python FastAPI | VoxCPM2がPython専用 |
| STT | OpenAI Whisper API | 日本語精度が高い |
| LLM | GPT-4o-mini | 速度/コストのバランス |
| TTS | VoxCPM2 (VoxCPMANE) | Apple Siliconに最適化 |

---

## プロジェクト構成

```
voice-phone-demo/
├── backend/
│   ├── main.py              # FastAPI エントリーポイント
│   ├── stt.py               # Whisper API ラッパー
│   ├── llm.py               # GPT-4o-mini ラッパー
│   ├── tts.py               # VoxCPM2 ラッパー
│   ├── reference_voice/     # ユーザーの声サンプル保存先
│   │   └── .gitkeep
│   └── requirements.txt
├── frontend/
│   └── index.html           # 単一ページUI
└── .env                     # OPENAI_API_KEY
```

---

## UI 仕様

2ステップのシンプルな画面:

**Step 1: 声登録画面**（初回のみ）
```
┌─────────────────────────────────┐
│  🎤 あなたの声を登録してください  │
│                                 │
│  [ ● 録音開始 (10秒) ]           │
│                                 │
│  「こんにちは、私の名前は〇〇です。│
│   よろしくお願いします。」        │ ← 読み上げ例文表示
│                                 │
│  [ → デモを始める ]              │ ← 録音完了後に有効化
└─────────────────────────────────┘
```

**Step 2: 会話画面**
```
┌─────────────────────────────────┐
│  AI 電話対応デモ (日本語)         │
│                                 │
│  [ 🎤 話す ]  ← 押している間録音  │
│                                 │
│  あなた: 営業時間を教えてください  │
│  AI:    はい、営業時間は〜        │
│  ...                            │
└─────────────────────────────────┘
```

---

## VoxCPM2 音声生成仕様

Controllable Cloning モードを使用:
```python
audio = model.generate(
    text="こんにちは、ご連絡ありがとうございます。",
    reference_wav_path="reference_voice/user.wav"
)
```

- reference_wav: ユーザーが録音した10秒クリップ
- テキスト説明なしの基本クローニング
- 出力: WAVファイル → フロントに返却

---

## AI システムプロンプト

```
あなたは親切な日本語の受付AIです。
自然な会話で丁寧にお答えください。
回答は簡潔に2〜3文でお願いします。
```

---

## API エンドポイント

| メソッド | パス | 説明 |
|---------|------|------|
| POST | `/register-voice` | 声サンプル登録 (WAV受信・保存) |
| POST | `/chat` | 音声入力 → AI音声返答 (WAV返却) |
| GET | `/health` | サーバー死活確認 |

---

## 環境変数 (.env)

```
OPENAI_API_KEY=sk-...
```

---

## 制約・注意事項

- VoxCPMANE は Python 3.10〜3.12 が必要 (3.13 未対応)
- 声クローニングはユーザー本人の声のみ使用すること
- デモ用途のため会話履歴はサーバー再起動でリセット
- Whisper APIへの音声送信: OpenAIのプライバシーポリシー適用
