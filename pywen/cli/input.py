from typing import AsyncIterator

async def interactive_inputs(prompt_session) -> AsyncIterator[str]:
    while True:
        try:
            text = await prompt_session.prompt_async()
        except EOFError:
            break
        s = (text or "").strip()
        if not s: continue
        if s.lower() in {"q","quit","exit"}: break
        yield s

async def single_input(text: str) -> AsyncIterator[str]:
    if text and text.strip():
        yield text

