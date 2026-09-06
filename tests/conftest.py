import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TRADEBOT_OFFLINE", "1")  # 测试不联网拉大盘基准
