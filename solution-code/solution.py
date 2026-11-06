import os
import time
from typing import Any, Callable

OPENAI_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"

# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (Đã xong)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "mock-key"))

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time

    response_text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }

    return response_text, latency, usage

def call_gemini(*args, **kwargs):
    raise NotImplementedError

def call_anthropic(*args, **kwargs):
    raise NotImplementedError

def compare_models(*args, **kwargs):
    raise NotImplementedError

def streaming_chatbot(*args, **kwargs):
    raise NotImplementedError

def retry_with_backoff(*args, **kwargs):
    raise NotImplementedError

def batch_compare(*args, **kwargs):
    raise NotImplementedError

def format_comparison_table(*args, **kwargs):
    raise NotImplementedError