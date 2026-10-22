from src.session_manager import ConversationSession


def test_conversation_session_export():
    s = ConversationSession()
    s.add_message('Test', 'Hello')
    assert s.get_message_count() == 1
    assert s.last_export_status == ""
    success = s.export()
    assert success is True
    assert s.last_export_status == "Success"
    assert s.last_export_path.startswith("chat_")


if __name__ == "__main__":
    test_conversation_session_export()
