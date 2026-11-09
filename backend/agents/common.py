
"""Shared configuration and helpers for AgentFlow AI agents."""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


GLOBAL_AGENT_PRINCIPLES = """
GLOBAL RESPONSE PRINCIPLES (apply to every agent):
- Understand the user's actual goal; provide an accurate, high-quality, and directly relevant answer.
- Work on information needed to fulfil the request thoroughly and concisely. Avoid irrelevant filler.
- The requested content is the deliverable; external sources are supporting evidence.
- A URL/citation is NOT required for every question. Include links when explicitly requested
  or materially useful for checking current, consequential, or researched information.
- Match effort, depth, and detail to the request: simple questions get direct answers; requests for
  explanations, comparisons, plans, or research get thorough, well-structured deliverables.
- Synthesize evidence honestly. Distinguish established facts, reasoned analysis, and unknowns.
""".strip()


BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BACKEND_DIR / ".env"

load_dotenv(ENV_PATH, override=False)


def format_llm_error(exc: Exception) -> str:
    """Provide a user-friendly diagnostic message for Gemini API errors."""

    msg = str(exc)
    lower_msg = msg.lower()

    if (
        "401" in msg
        or "unauthorized" in lower_msg
        or "invalid api key" in lower_msg
    ):
        return (
            "**Gemini API Authentication Error (401)**\n\n"
            "Your GEMINI_API_KEY may be invalid. "
            "Check your key in https://aistudio.google.com/apikey."
        )

    if "403" in msg or "permission denied" in lower_msg:
        return (
            "**Gemini API Permission Error (403)**\n\n"
            "Check whether your API key and Google AI Studio project "
            "have permission to access the requested model."
        )

    if (
        "429" in msg
        or "resource_exhausted" in lower_msg
        or "rate limit" in lower_msg
    ):
        return (
            "**Gemini API Rate Limit / Quota Error (429)**\n\n"
            "Your Gemini API request limit or quota may have been reached. "
            "Check your usage and limits in Google AI Studio."
        )

    if "503" in msg or "unavailable" in lower_msg:
        return (
            "**Gemini API Temporarily Unavailable (503)**\n\n"
            "Google's Gemini service may be experiencing high demand. "
            "Please wait briefly and try again."
        )

    if "404" in msg or "not found" in lower_msg:
        return (
            "**Gemini Model Error (404)**\n\n"
            "The configured Gemini model may not be available. "
            "Check the model name in backend/.env."
        )

    if "timeout" in lower_msg:
        return (
            "**Gemini API Timeout**\n\n"
            "The request took too long. Please try again."
        )

    return f"Gemini API execution error: {msg}"


def get_llm(
    *,
    temperature: float = 0,
    max_tokens: int | None = None
) -> ChatGoogleGenerativeAI:
    """Create the configured Gemini model using Google's native integration."""

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to backend/.env."
        )

    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=(
            max_tokens
            if max_tokens is not None
            else int(os.getenv("LLM_MAX_TOKENS", "1500"))
        ),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
    )


def run_async(coro: Any) -> Any:
    """Run a coroutine from a synchronous LangGraph node, even if a loop exists."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def extract_text(result: Any) -> str:
    """Normalize common MCP/LangChain tool result shapes to plain text."""

    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        if "text" in result:
            return str(result["text"])
        return str(result)

    if isinstance(result, (list, tuple)):
        parts = []

        for item in result:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(item.text))

        return "\n".join(parts)

    if hasattr(result, "content"):
        return extract_text(result.content)

    if hasattr(result, "text"):
        return str(result.text)

    return str(result)


def normalise_text(value: str) -> str:
    import re

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        (value or "").lower()
    ).strip()


def text_contains(value: str, source: str) -> bool:
    needle = normalise_text(value)
    haystack = normalise_text(source)

    return bool(
        needle
        and len(needle) >= 2
        and needle in haystack
    )


def is_hiring_request(request: str) -> bool:
    text = (request or "").lower()

    terms = (
        "hiring",
        "fresher",
        "freshers",
        "job opening",
        "job openings",
        "vacancy",
        "vacancies",
        "recruitment",
        "job opportunities",
        "actively hiring",
        "off-campus",
        "off campus",
        "jobs in india",
    )

    return any(term in text for term in terms)
