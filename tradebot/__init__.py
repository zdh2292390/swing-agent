"""tradebot: 美股小时频交易 agent 骨架。

分层：data -> strategies -> (agent) -> risk -> execution，风控和执行是确定性代码，
LLM 只能否决或降低仓位，永远不能直接下单。
"""
