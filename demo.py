import os
from openai import OpenAI


def call_llm(prompt) -> str:
        client = OpenAI(
            api_key=os.environ["MODELSCOPE_API_KEY"],
            base_url="https://api-inference.modelscope.cn/v1/"
        )

        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen3-235B-A22B-Instruct-2507",
                messages=[{"role": "user", "content": prompt}],
                top_p=0.7,
                temperature=0
            )
            return response

        except Exception as e:
            print(f"[bold red]Error calling LLM for memory compression: {e}[/]")

    
res = call_llm("你好")
print(res)