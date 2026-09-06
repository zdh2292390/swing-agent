import os

from tradebot.config import update_env


def test_update_env_replaces_keeps_comment_and_appends(tmp_path):
    env = tmp_path / ".env"
    env.write_text("MODE=local                # 模式\nALPACA_API_KEY=\nOTHER=1\n", encoding="utf-8")
    update_env({"MODE": "alpaca", "ALPACA_API_KEY": "abc123", "NEWKEY": "x"}, path=env)
    txt = env.read_text(encoding="utf-8")
    assert "MODE=alpaca  # 模式" in txt and "ALPACA_API_KEY=abc123" in txt and "OTHER=1" in txt and txt.strip().endswith("NEWKEY=x")
    assert os.environ["MODE"] == "alpaca" and os.environ["ALPACA_API_KEY"] == "abc123"
    os.environ["MODE"] = "local"; os.environ.pop("ALPACA_API_KEY", None); os.environ.pop("NEWKEY", None)
