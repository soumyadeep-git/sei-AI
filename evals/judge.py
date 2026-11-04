"""Judge LLM for `eval:` criteria, on Groq's free tier.

A different model family from the bot's, so the bot doesn't grade itself.
"""

import os

from pipecat.services.groq.llm import GroqLLMService


def groq(config: dict) -> GroqLLMService:
    return GroqLLMService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqLLMService.Settings(
            model=config.get("model", "qwen/qwen3.8-27b"),
            reasoning_effort="none",
        ),
    )
