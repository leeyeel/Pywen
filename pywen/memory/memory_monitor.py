from __future__ import annotations
import re
import os
from openai import AsyncOpenAI
from .prompt import compression_prompt, keyword_continuity_score_prompt, first_downgrade_prompt, second_downgrade_prompt
from pywen.utils.llm_basics import LLMMessage
from typing import Dict, Any


class MemoryMonitor:

    def __init__(self, config,console, verbose=True):
        self.config = config
        self.check_interval = self.config.memory_monitor.check_interval
        self.max_tokens = self.config.memory_monitor.maximum_capacity
        self.rules = self.config.memory_monitor.rules
        self.model = self.config.memory_monitor.model
        self.last_checkd_turn = 0
        self.console = console
        self.verbose = verbose


    async def call_llm(self, prompt) -> str:
        client = AsyncOpenAI(
            api_key=self.config.active_model.api_key,
            base_url=self.config.active_model.base_url
        )

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                top_p=0.7,
                temperature=0
            )
            return response

        except Exception as e:
            if self.verbose:
                self.console.print(f"Error calling LLM for memory compression: {e}", "red", True)


    async def run_monitored(self, turn, conversation_history, usage):
        if self.verbose:
            self.console.print(f"\nMonitoring on turn {turn} 🚀", "magenta", True)

        if (turn - self.last_checkd_turn) % self.check_interval == 0:
            alert = self.maybe_compress(usage)
            self.check_interval = alert["check_interval"]
            self.last_checkd_turn = turn
        else:
            return None

        if alert is not None and alert["level"] == "compress":
            if self.verbose:
                self.console.print(alert["suggestion"], "red", True)
            summary, original = await self.do_compress(conversation_history)
            quality = await self.quality_validation(summary, original, usage)
            if quality["valid"]:
                if self.verbose:
                    self.console.print("🚀 Memory compression success!", "green", True)
                summary_content = summary.choices[0].message.content
                return summary_content
            else:
                if self.verbose:
                    self.console.print("⚠️ Memory compression fail, downgrade strategy will be executed!", "green", True)
                summary = await self.downgrade_compression(summary, original)
                if summary is not None:
                    summary_content = summary.choices[0].message.content
                    return summary_content
                else:
                    if self.verbose:
                        self.console.print("⚠️ All downgrade attempts failed, using 30% latest messages strategy...", "yellow", False)
                    summary_content = self.retain_latest_messages(conversation_history)
                    return summary_content
            
        elif alert is not None and alert["level"] != "compress":
            if self.verbose:
                self.console.print(alert["suggestion"], "yellow", False)


    def maybe_compress(self, token_usage) -> Dict[str, Any] | None:
        ratio = token_usage / self.max_tokens
        if self.verbose:
            self.console.print(f"Token usage: {token_usage}, ratio: {ratio:.2%}", "cyan", True)

        if ratio >= 0.92:
            return self.warning("compress", ratio)
        elif ratio >= 0.80:
            return self.warning("high", ratio)
        elif ratio >= 0.60:
            return self.warning("moderate", ratio)
        else:
            return self.warning("", ratio)


    def pick_interval(self, ratio: float) -> int:
        for r, interval in self.rules:
            if ratio >= r:
                return interval


    def warning(self, level: str, ratio: float) -> Dict[str, Any]:
        check_interval = self.pick_interval(ratio)

        match ratio:
            case r if r >= 0.92:
                suggestion = "Memory usage – threshold reached! Executing compression!"
            case r if r >= 0.80:
                suggestion = f"Memory usage – high, checking every {check_interval} turn(s). You can restart a new conversation!"
            case r if r >= 0.60:
                suggestion = f"Memory usage – moderate, checking every {check_interval} turn(s)."
            case _:
                suggestion = f"Memory usage – low, checking every {check_interval} turn(s)."
                
        return {
            "level": level,
            "suggestion": suggestion,
            "check_interval": check_interval
        }


    async def do_compress(self, conversation_history: list[LLMMessage]) -> tuple[str, str]:
        original = "\n".join(f"{message.role}: {message.content}" for message in conversation_history)
        prompt = compression_prompt.format(original)
        summary = await self.call_llm(prompt)

        return summary, original
        

    def ratio_score(self, summary_tokens: int, usage: int) -> float:
        return summary_tokens / usage


    def section_score(self, summary: str) -> float:
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

        found = [s for s in required if re.search(rf"\b{re.escape(s)}\b", summary, re.I)]

        return len(found) / len(required)


    async def keyword_continuity_score(self, summary: str, original: str):
        prompt = keyword_continuity_score_prompt.format(summary, original)
        response = await self.call_llm(prompt)
        response = response.choices[0].message.content.strip()

        if not response.startswith("Result:"):
            raise ValueError("Missing 'Result:' prefix")
        _, scores = response.split("Result:", 1)
        parts = scores.strip().split()

        if len(parts) != 2:
            raise ValueError("Malformed score line")
        return float(parts[0]), float(parts[1]) 

    
    async def quality_validation(self, summary: str, original: str, usage: int) -> Dict[str, Any]:
        summary_tokens = summary.usage.completion_tokens
        summary_content = summary.choices[0].message.content
        ratio_score = self.ratio_score(summary_tokens, usage)
        section_ratio = self.section_score(summary_content)
        keyword_ratio, continuity_ratio = await self.keyword_continuity_score(summary_content, original)

        fidelity = int(
            section_ratio * 100 * 0.3 +
            keyword_ratio * 100 * 0.4 +
            continuity_ratio * 100 * 0.2 +
            (100 if ratio_score <= 0.15 else 50) * 0.1
        )

        is_valid = fidelity >= 80
        suggestions = []

        if section_ratio < 0.875:
            suggestions.append(f"⚠️  {section_ratio:.2%} Missing required sections; please include all 8.")
        if keyword_ratio < 0.8:
            suggestions.append(f"⚠️  {keyword_ratio:.2%} Key information loss detected; compress less aggressively.")
        if ratio_score > 0.15:
            suggestions.append(f"⚠️  {ratio_score:.2%} Compression ratio too low; consider deeper summarization.")
        if continuity_ratio < 0.6:
            suggestions.append(f"⚠️  {continuity_ratio:.2%} Context flow broken; add transition phrases.")

        return {
            "fidelity": fidelity,
            "valid": is_valid,
            "suggestions": suggestions,
        }

    
    async def downgrade_compression(self, summary: str, original: str) -> list[LLMMessage]:
        summary_content = summary.choices[0].message.content
        attempts = [
            dict(
                label="First attempt", 
                prompt=first_downgrade_prompt, 
                threshold=75, 
                emoji="🔄"
            ),
            dict(
                label="Second attempt",
                prompt=second_downgrade_prompt,
                threshold=70,
                emoji="📦"
            ),
        ]

        for attempt in attempts:
            if self.verbose:
                self.console.print(f"{attempt['emoji']} {attempt['label']}: recompress the conversation history...", "cyan", False)
            prompt = attempt["prompt"].format(summary_content, original)
            downgrade_summary = await self.call_llm(prompt)
            quality = await self.quality_validation(downgrade_summary, original)

            if quality["fidelity"] >= attempt["threshold"]:
                if self.verbose:
                    self.console.print(f"✅ {attempt['label']} successful, fidelity: {quality['fidelity']}%", "green", False)
                return downgrade_summary
            else:
                if self.verbose:
                    self.console.print(f"❌ {attempt['label']} fail.", "red", False)
                return None
        

    def retain_latest_messages(self, conversation_history: list[LLMMessage]) -> list[LLMMessage]:
        if not conversation_history:
            return ""

        total = len(conversation_history)
        keep = max(1, int(total * 0.3))
        candidates = conversation_history[-keep:]

        first_user_idx = next((i for i, m in enumerate(candidates) if m.role == "user"), None)

        if first_user_idx is None:
        # 向前补到最近的 user
            for i in range(total - keep - 1, -1, -1):
                if conversation_history[i].role == "user":
                    retained = conversation_history[i:]
                    break
            else:
                retained = conversation_history[-1:]  # 兜底最后一条
        else:
            retained = candidates[first_user_idx:]

        # 拼接成文本
        return "\n".join(f"{msg.role}: {msg.content}" for msg in retained)



        # if not conversation_history:
        #     return None

        # total_tokens = 0
        # message_tokens = []
        
        # for message in conversation_history:
        #     content = message.content.strip()
        #     tokens = self.tokenizer.encode(content)
        #     token_count = len(tokens)
        #     message_tokens.append((message, token_count))
        #     total_tokens += token_count
        
        # retain_tokens = max(1, int(total_tokens * 0.3))
        
        # retained_messages = []
        # accumulated_tokens = 0
        
        # for message, token_count in reversed(message_tokens):
        #     if accumulated_tokens + token_count <= retain_tokens:
        #         retained_messages.insert(0, message)
        #         accumulated_tokens += token_count
        #     else:
        #         if not retained_messages:
        #             retained_messages.insert(0, message)
        #         break
        
        # first_user_index = None
        # for i, msg in enumerate(retained_messages):
        #     if msg.role == "user":
        #         first_user_index = i
        #         break
        
        # if first_user_index is not None and first_user_index > 0:
        #     adjusted_messages = retained_messages[first_user_index:]
        # else:
        #     adjusted_messages = retained_messages
        
        # actual_retained_tokens = 0
        # for msg in adjusted_messages:
        #     content = msg.content.strip()
        #     tokens = self.tokenizer.encode(content)
        #     actual_retained_tokens += len(tokens)
        
        # self.console.print(f"[green]✅ Retained {actual_retained_tokens} out of {total_tokens} tokens "
        #       f"({actual_retained_tokens/total_tokens:.1%}), {len(adjusted_messages)} messages")
        
        # summary = "\n".join(f"{message.role}: {message.content}" for message in adjusted_messages)
        # return summary
        


        



        


