from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple
from .prompt import (
    compression_prompt,
    keyword_continuity_score_prompt,
    first_downgrade_prompt,
    second_downgrade_prompt,
)
from pywen.llm.llm_basics import LLMMessage
from pywen.llm.llm_events import LLM_Events
from pywen.config.manager import ConfigManager

class MemoryMonitor:
    def __init__(self, config_mgr: ConfigManager):
        self.cfg_mgr = config_mgr
        mm = self.cfg_mgr.get_app_config().memory_monitor
        self.check_interval: int = (mm.check_interval if mm and mm.check_interval else 5)
        self.max_tokens: int = (mm.maximum_capacity if mm and mm.maximum_capacity else 4096)
        self.rules: List[Tuple[float, int]] = (mm.rules if mm and getattr(mm, "rules", None) else [])
        if not self.rules:
            self.rules = [(0.92, 1), (0.80, 2), (0.60, 3), (0.0, 5)]
        self._last_checked_turn = 0

    async def run_monitored(self, llm_client, history: List[LLMMessage], tokens_used: int, turn: int) ->Tuple[int, str]:
        if (turn - self._last_checked_turn) % self.check_interval != 0:
            return 0, ""

        ratio = self._safe_ratio(tokens_used, self.max_tokens)
        alert = self._make_alert(ratio)
        self.check_interval = alert["check_interval"]
        self._last_checked_turn = turn

        if alert["level"] != "compress":
            return 0, ""

        original_text = self._flatten_history(history)
        tokens_used, summary_text = await self._ask_user_prompt(llm_client, compression_prompt.format(original_text))
        if not summary_text:
            return 0, self._retain_latest_messages(history)

        return await self._validate_and_maybe_downgrade(llm_client, summary_text, original_text, tokens_used)

    def _safe_ratio(self, used: int, cap: int) -> float:
        if cap <= 0:
            return 0.0
        return max(0.0, min(1.0, used / cap))

    def _pick_interval(self, ratio: float) -> int:
        for r, interval in self.rules:
            if ratio >= r:
                return interval
        return self.rules[-1][1]

    def _make_alert(self, ratio: float) -> Dict[str, Any]:
        if ratio >= 0.92:
            level = "compress"
        elif ratio >= 0.80:
            level = "high"
        elif ratio >= 0.60:
            level = "moderate"
        else:
            level = ""

        interval = self._pick_interval(ratio)
        if level == "compress":
            suggestion = "Memory usage reached threshold. Executing compression."
        elif level == "high":
            suggestion = f"Memory usage high; next check every {interval} turn(s)."
        elif level == "moderate":
            suggestion = f"Memory usage moderate; next check every {interval} turn(s)."
        else:
            suggestion = f"Memory usage low; next check every {interval} turn(s)."

        return {"level": level, "check_interval": interval, "suggestion": suggestion}

    def _flatten_history(self, history: List[LLMMessage]) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in history if getattr(m, "content", None))

    async def _ask_user_prompt(self, llm_client, prompt_text: str) -> Tuple[int, str]:
        messages = [{"role": "user", "content": prompt_text}]
        model = self.cfg_mgr.get_active_model_name()
        buf: List[str] = []
        tokens_used : int = 0
        try:
            async for event in llm_client.astream_response(
                messages=messages, api="chat", model=model, temperature=0, top_p=0.7
            ):
                if event.type == LLM_Events.ASSISTANT_DELTA and event.data:
                    buf.append(str(event.data))
                elif event.type in (LLM_Events.ERROR, LLM_Events.RESPONSE_FINISHED, LLM_Events.ASSISTANT_FINISHED):
                    break
                elif event.type == LLM_Events.TOKEN_USAGE and event.data:
                    tokens_used += int(event.data.get("total_tokens", 0))
        except Exception as e:
            return 0, ""

        return tokens_used, "".join(buf).strip()

    def _section_score(self, txt: str) -> float:
        required = [
            "Primary Request and Intent",
            "Key Technical Concepts",
            "Files and Code Sections",
            "Errors and fixes",
            "Problem Solving",
            "All user messages",
            "Pending Tasks",
            "Current Work",
        ]
        found = sum(1 for s in required if re.search(rf"\b{re.escape(s)}\b", txt, re.I))
        return found / len(required)

    async def _keyword_continuity(self, llm_client, summary: str, original: str) -> Tuple[float, float]:
        """
        期望模型返回: "Result: <keyword_score> <continuity_score>"
        失败则返回 (0.0, 0.0)。
        """
        _, reply = await self._ask_user_prompt(llm_client, keyword_continuity_score_prompt.format(summary, original))
        if not reply.startswith("Result"):
            return (0.0, 0.0)
        try:
            _, tail = reply.split("Result", 1)
            nums = tail.replace(":", " ").split()
            return (float(nums[0]), float(nums[1]))
        except Exception:
            return (0.0, 0.0)

    def _estimate_tokens(self, text: str) -> int:
        # 粗估：约 4 字符 ≈ 1 token
        return max(1, len(text) // 4)

    async def _quality(self, llm_client, summary_text: str, original: str, tokens_used: int) -> Dict[str, Any]:
        section_ratio = self._section_score(summary_text)
        keyword_ratio, continuity_ratio = await self._keyword_continuity(llm_client, summary_text, original)
        est_summary_tokens = self._estimate_tokens(summary_text)
        ratio_score = (est_summary_tokens / tokens_used) if tokens_used > 0 else 1.0

        fidelity = int(
            section_ratio * 100 * 0.3
            + keyword_ratio * 100 * 0.4
            + continuity_ratio * 100 * 0.2
            + (100 if ratio_score <= 0.15 else 50) * 0.1
        )
        return {
            "valid": fidelity >= 80,
            "fidelity": fidelity,
            "section_ratio": section_ratio,
            "keyword_ratio": keyword_ratio,
            "continuity_ratio": continuity_ratio,
            "ratio_score": ratio_score,
        }

    async def _validate_and_maybe_downgrade(self, llm_client, summary_text: str, original_text: str, tokens_used: int,)-> Tuple[int, str]:
        # 第一次质量评估
        q = await self._quality(llm_client, summary_text, original_text, tokens_used)
        if q["valid"]:
            return tokens_used, summary_text

        # 降级 1
        tokens_used, d1_text = await self._ask_user_prompt(llm_client, first_downgrade_prompt.format(summary_text, original_text))
        if d1_text:
            q1 = await self._quality(llm_client, d1_text, original_text, tokens_used)
            if q1["fidelity"] >= 75:
                return tokens_used, d1_text

        # 降级 2
        base = d1_text if d1_text else summary_text
        tokens_used, d2_text = await self._ask_user_prompt(llm_client, second_downgrade_prompt.format(base, original_text))
        if d2_text:
            q2 = await self._quality(llm_client, d2_text, original_text, tokens_used)
            if q2["fidelity"] >= 70:
                return True, d2_text

        return 0, ""

    def _retain_latest_messages(self, history: List[LLMMessage]) -> str:
        """
        兜底：返回最近 30%，并从最近的 user 起拼接。
        """
        if not history:
            return ""

        total = len(history)
        keep = max(1, int(total * 0.3))
        candidates = history[-keep:]

        first_user_idx = next((i for i, m in enumerate(candidates) if m.role == "user"), None)
        if first_user_idx is None:
            start = None
            for i in range(total - keep - 1, -1, -1):
                if history[i].role == "user":
                    start = i
                    break
            retained = history[start:] if start is not None else history[-1:]
        else:
            retained = candidates[first_user_idx:]

        return "\n".join(f"{m.role}: {m.content}" for m in retained if getattr(m, "content", None))
