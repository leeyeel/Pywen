from __future__ import annotations
import re
import os
from pathlib import Path
from openai import AsyncOpenAI
from .prompt import compression_prompt, keyword_continuity_score_prompt, first_downgrade_prompt, second_downgrade_prompt
from dataclasses import dataclass
from pywen.utils.llm_basics import LLMMessage
from typing import Dict, Any
from rich import print
from .file_restorer import IntelligentFileRestorer


@dataclass
class AdaptiveThreshold:

    check_interval: int = 3
    max_tokens: int = 200000
    rules: tuple[tuple[float, int], ...] = (
        (0.92, 1),
        (0.80, 1),   # ≥80 % 每 1 轮
        (0.60, 2),   # ≥60 % 每 2 轮
        (0.00, 3),   # 默认每 3 轮
    )


class MemoryMonitor:

    def __init__(self, threshold: AdaptiveThreshold | None = None):
        self.check_interval = threshold.check_interval
        self.max_tokens = threshold.max_tokens
        self.rules = threshold.rules
        self.model = "Qwen/Qwen3-235B-A22B-Instruct-2507"
        self.file_restorer = IntelligentFileRestorer()


    async def call_llm(self, prompt) -> str:
        client = AsyncOpenAI(
            api_key=os.environ["MODELSCOPE_API_KEY"],
            base_url="https://api-inference.modelscope.cn/v1/"
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
            print(f"[bold red]Error calling LLM for memory compression: {e}[/]")


    async def run_monitored(self, turn, conversation_history, usage):
        print(f"\n[bold magenta]Monitoring[/] on turn [underline cyan]{turn}[/] :rocket:")

        if turn % self.check_interval == 0:
            alert = self.maybe_compress(usage)
            self.check_interval = alert["check_interval"]
        else:
            return None

        if alert is not None and alert["level"] == "compress":
            print(alert["suggestion"])
            summary, original = await self.do_compress(conversation_history)
            quality = await self.quality_validation(summary, original, usage)
            if quality["valid"]:
                print("[bold green]🚀 Memory compression success![/]")
                conversation_summary = [LLMMessage(role="user", content=summary)]
                return conversation_summary
            else:
                print("[bold green]⚠️ Memory compression fail, downgrade strategy will be executed.![/]")
                summary = await self.downgrade_compression(summary, original)
                if summary is not None:
                    conversation_summary = [LLMMessage(role="user", content=summary)]
                    return conversation_summary
                else:
                    print("[yellow]⚠️ All downgrade attempts failed, using 30% latest messages strategy...[/]")
                    summary = self.retain_latest_messages(conversation_history)
                    conversation_summary = [LLMMessage(role="user", content=summary)]
                    return conversation_summary
            
        elif alert is not None and alert["level"] != "compress":
            print(alert["suggestion"])


    def maybe_compress(self, token_usage) -> Dict[str, Any] | None:
        ratio = token_usage / self.max_tokens
        print(f"Token usage: [bold cyan]{token_usage}[/], ratio: [bold magenta]{ratio:.2%}[/]")

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
                suggestion = f"[bold red]Memory usage – threshold reached! [/][red]Executing compression![/]"
            case r if r >= 0.80:
                suggestion = f"[orange1]Memory usage – high, checking every {check_interval} turn(s).[/] [yellow]You can restart a new conversation![/]"
            case r if r >= 0.60:
                suggestion = f"[bright_green]Memory usage – moderate, checking every {check_interval} turn(s).[/]"
            case _:
                suggestion = f"[dim bright_blue]Memory usage – low, checking every {check_interval} turn(s).[/]"
                
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
        

    def ratio_score(self, summary: str, usage: int) -> float:
        return len(summary) / len(usage)


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
        summary_tokens = summary.usage.output_tokens
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
            suggestions.append(f"[red]⚠️  {section_ratio:.2%}[/red] Missing required sections; please include all 8.")
        if keyword_ratio < 0.8:
            suggestions.append(f"[orange1]⚠️  {keyword_ratio:.2%}[/orange1] Key information loss detected; compress less aggressively.")
        if ratio_score > 0.15:
            suggestions.append(f"[bright_magenta]⚠️  {ratio_score:.2%}[/bright_magenta] Compression ratio too low; consider deeper summarization.")
        if continuity_ratio < 0.6:
            suggestions.append(f"[yellow]⚠️  {continuity_ratio:.2%}[/yellow] Context flow broken; add transition phrases.")

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
            print(f"[cyan]{attempt['emoji']} {attempt['label']}: recompress the conversation history...[/]")
            prompt = attempt["prompt"].format(summary_content, original)
            downgrade_summary = (await self.call_llm(prompt)).strip()
            quality = await self.quality_validation(downgrade_summary, original)

            if quality["fidelity"] >= attempt["threshold"]:
                print(f"[green]✅ {attempt['label']} successful, fidelity: {quality['fidelity']}%[/]")
                return downgrade_summary
            else:
                print(f"[red]❌ {attempt['label']} fail.[/]")
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
        
        # print(f"[green]✅ Retained {actual_retained_tokens} out of {total_tokens} tokens "
        #       f"({actual_retained_tokens/total_tokens:.1%}), {len(adjusted_messages)} messages")
        
        # summary = "\n".join(f"{message.role}: {message.content}" for message in adjusted_messages)
        # return summary
        

    def file_recover(self):
        directory = Path.cwd()
        metadatas = self.file_restorer.get_directory_metadata(directory)
        ranked_files = []
        for md in metadatas:
            md_copy = md.copy()
            score = self.file_restorer.calculate_importance_score(md_copy)
            md_copy["score"] = score
            ranked_files.append(md_copy)
        selected = self.file_restorer.select_optimal_file_set(ranked_files)
        # Sort selected files by score descending
        sorted_selected = sorted(selected["files"], key=lambda f: f["score"], reverse=True)
        # Read contents
        dir_path = Path(directory).resolve()
        contents = []
        for file in sorted_selected:
            full_path = dir_path / file["path"]
            try:
                content = full_path.read_text(encoding="utf-8")
                contents.append(f"File: {file['path']}\nScore: {file['score']}\nContent:\n{content}\n\n")
            except Exception as e:
                contents.append(f"File: {file['path']}\nError reading: {str(e)}\n\n")
        return "".join(contents)


        



        


