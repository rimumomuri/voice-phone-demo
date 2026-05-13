from unittest.mock import patch, MagicMock


def test_chat_returns_reply_and_updated_history():
    with patch('llm.client') as mock_client:
        mock_choice = MagicMock()
        mock_choice.message.content = "はい、承知しました。"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        from llm import chat
        reply, history = chat([], "こんにちは")

        assert reply == "はい、承知しました。"
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "こんにちは"}
        assert history[1] == {"role": "assistant", "content": "はい、承知しました。"}


def test_chat_appends_to_existing_history():
    with patch('llm.client') as mock_client:
        mock_choice = MagicMock()
        mock_choice.message.content = "かしこまりました。"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        from llm import chat
        existing = [
            {"role": "user", "content": "前の質問"},
            {"role": "assistant", "content": "前の返答"},
        ]
        reply, history = chat(existing, "次の質問")

        assert len(history) == 4
        assert history[2]["role"] == "user"
        assert history[3]["content"] == "かしこまりました。"


def test_chat_includes_system_prompt():
    with patch('llm.client') as mock_client:
        mock_choice = MagicMock()
        mock_choice.message.content = "返答"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        from llm import chat
        chat([], "質問")

        messages = mock_client.chat.completions.create.call_args.kwargs['messages']
        assert messages[0]['role'] == 'system'
