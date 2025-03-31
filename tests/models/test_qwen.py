from src.llms.qwen import QwenClient


def test_generate():
    """Test single-round generation"""
    qwenclient = QwenClient()
    content = qwenclient.generate("What's the highest mountain in the world?")
    assert content


def test_multi_round_generate():
    """Test multi-round generation"""
    qwenclient = QwenClient()
    # First round
    content1 = qwenclient.multi_round_generate(
        "What's the highest mountain in the world?"
    )
    assert content1
    # Second round
    content2 = qwenclient.multi_round_generate("How tall is it?")
    assert content2


def test_view_messages():
    """Test viewing message history"""
    qwenclient = QwenClient()
    qwenclient.generate("Test message", max_tokens=10)
    # Just verify the method runs without error
    qwenclient.view_messages()


def test_save_messages(tmp_path):
    """Test saving messages to file"""
    import os

    qwenclient = QwenClient()
    qwenclient.generate("Test save message", max_tokens=10)
    test_file = tmp_path / "test_messages.jsonl"
    qwenclient.save_messages(str(test_file))
    assert os.path.exists(test_file)
    assert os.path.getsize(test_file) > 0
