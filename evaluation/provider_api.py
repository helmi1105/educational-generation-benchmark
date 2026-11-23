"""Provider adapter for OpenAI Responses and OpenRouter Chat Completions.

No API keys are stored here. Keys are read from OPENAI_API_KEY or
OPENROUTER_API_KEY according to the explicitly selected provider.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def provider_env_var(provider: str) -> str:
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    raise ValueError(f"Unsupported provider: {provider}")


def create_client(
    provider: str,
    *,
    timeout: float,
    openrouter_site_url: str | None = None,
    openrouter_app_name: str | None = None,
) -> AsyncOpenAI:
    env_var = provider_env_var(provider)
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(f"{env_var} is not set")
    if provider == "openai":
        return AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=0)
    headers: dict[str, str] = {}
    if openrouter_site_url:
        headers["HTTP-Referer"] = openrouter_site_url
    if openrouter_app_name:
        headers["X-Title"] = openrouter_app_name
    return AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=headers or None,
        timeout=timeout,
        max_retries=0,
    )


def _parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    if not cleaned:
        raise ValueError("The API response contained no JSON text")
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("The model output was not a JSON object")
    return value


def _chat_content(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("OpenRouter returned no choices")
    content = getattr(choices[0].message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(getattr(item, "text", None), str):
                parts.append(item.text)
        return "".join(parts)
    raise ValueError("OpenRouter returned no textual message content")


def _portable_api_schema(value: Any) -> Any:
    """Remove assertions that several routed structured-output backends reject.

    The original schema remains in the prompt and runners perform their own
    semantic validation after parsing.
    """
    unsupported = {"$schema", "minLength", "minItems", "maxItems", "uniqueItems"}
    if isinstance(value, dict):
        return {
            key: _portable_api_schema(item)
            for key, item in value.items()
            if key not in unsupported
        }
    if isinstance(value, list):
        return [_portable_api_schema(item) for item in value]
    return value


async def request_json_once(
    client: AsyncOpenAI,
    *,
    provider: str,
    model: str,
    system: str,
    prompt: str,
    schema: dict[str, Any],
    schema_name: str,
    max_output_tokens: int,
    openai_output_mode: str,
    openrouter_output_mode: str,
    reasoning_effort: str = "omit",
    verbosity: str = "low",
    send_openrouter_reasoning: bool = False,
) -> tuple[Any, dict[str, Any]]:
    if provider == "openai":
        if openai_output_mode == "json_schema":
            output_format: dict[str, Any] = {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        else:
            output_format = {"type": "json_object"}
        kwargs: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "text": {"format": output_format, "verbosity": verbosity},
            "max_output_tokens": max_output_tokens,
        }
        if reasoning_effort != "omit":
            kwargs["reasoning"] = {"effort": reasoning_effort}
        response = await client.responses.create(**kwargs)
        return response, _parse_json_text(getattr(response, "output_text", "") or "")

    # Include the schema in the prompt even when response_format is enabled;
    # this keeps behavior explicit for open-source models and providers.
    router_prompt = prompt + "\n\nRequired JSON Schema:\n" + json.dumps(
        schema, ensure_ascii=False, separators=(",", ":")
    )
    chat_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": router_prompt},
        ],
        "max_tokens": max_output_tokens,
    }
    if openrouter_output_mode == "json_schema":
        portable_schema = _portable_api_schema(schema)
        chat_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": portable_schema},
        }
    elif openrouter_output_mode == "json_object":
        chat_kwargs["response_format"] = {"type": "json_object"}
    elif openrouter_output_mode != "prompt_only":
        raise ValueError(f"Unsupported OpenRouter output mode: {openrouter_output_mode}")
    if send_openrouter_reasoning and reasoning_effort != "omit":
        chat_kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
    response = await client.chat.completions.create(**chat_kwargs)
    return response, _parse_json_text(_chat_content(response))


def response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    input_tokens = int(
        getattr(usage, "input_tokens", 0)
        or getattr(usage, "prompt_tokens", 0)
        or 0
    )
    output_tokens = int(
        getattr(usage, "output_tokens", 0)
        or getattr(usage, "completion_tokens", 0)
        or 0
    )
    total_tokens = int(getattr(usage, "total_tokens", 0) or input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
