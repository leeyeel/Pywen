import asyncio
from typing import Any
from llm.llm_events import TurnResult, CoreStreamEvent
from hooks.middlewares import MiddlewareChain

class ConversationRunner:
    def __init__(self, agent: Any, renderer: Any, middlewares: MiddlewareChain) -> None:
        self.agent = agent
        self.renderer = renderer
        self.middlewares = middlewares
        self.in_turn = False
        self.cancel_event = asyncio.Event()
        self.current_task: asyncio.Task | None = None

    def start(self):
        self.in_turn = True
        self.cancel_event.clear()

    def attach(self, task: asyncio.Task):
        self.current_task = task

    def cancel(self):
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
        self.in_turn = False
        self.cancel_event.set()

    def reset(self):
        self.in_turn = False
        self.current_task = None
        self.cancel_event.clear()

    async def run_turn(self, user_input: str) -> TurnResult:
        ok, msg, extra = await self.middlewares.before_prompt_submit(user_input)
        if not ok:
            self.renderer.print(f"⛔ {msg or 'Prompt blocked by hook'}", "yellow")
            return TurnResult.COMPLETED
        if extra.get("additionalContext"):
            from pywen.llm.llm_basics import LLMMessage
            self.agent.conversation_history.append(LLMMessage(role="user", content=extra["additionalContext"]))

        try:
            async for event in self.agent.run(user_input):
                if self.cancel_event.is_set():
                    self.renderer.print("\n⚠️ Operation cancelled by user", "yellow")
                    return TurnResult.CANCELLED

                if not await self.middlewares.on_event(event, self.agent):
                    return TurnResult.CANCELLED

                result = await self.renderer.handle_streaming_event(event, self.agent)

                if result == "tool_result":
                    await self.middlewares.on_tool_result(event, self.agent)

                if result in {"task_complete","max_turns_reached","waiting_for_user"}:
                    await self.middlewares.on_turn_stop(result, event, self.agent, user_input)
                    if result == "waiting_for_user":
                        return TurnResult.WAITING_FOR_USER
                    if result == "max_turns_reached":
                        return TurnResult.MAX_TURNS
                    return TurnResult.COMPLETED

                if event.get("type") == "error":
                    return TurnResult.ERROR

            return TurnResult.COMPLETED

        except asyncio.CancelledError:
            self.renderer.print("\n⚠️ Task was cancelled", "yellow")
            return TurnResult.CANCELLED
        except UnicodeError as e:
            self.renderer.print(f"\nUnicode 错误: {e}", "red")
            return TurnResult.ERROR
        except KeyboardInterrupt:
            self.renderer.print("\n⚠️ Operation interrupted by user", "yellow")
            return TurnResult.CANCELLED
        except Exception as e:
            self.renderer.print(f"\nError: {e}", "red")
            return TurnResult.ERROR
        finally:
            self.reset()
