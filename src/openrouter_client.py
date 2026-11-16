"""Minimal, transparent OpenRouter Chat Completions client with tool-calling, retry on transient 429/5xx, and exact token/cost accounting."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

# Standard OpenRouter pricing for openai/gpt-4o-mini ($0.15 / 1M input, $0.60 / 1M output)
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "google/gemini-2.5-flash": (0.15, 0.60),
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int, reported_cost: Any = None) -> float:
    if isinstance(reported_cost, (int, float)) and reported_cost > 0:
        return float(reported_cost)
    in_rate, out_rate = MODEL_PRICING_PER_1M.get(model, (0.15, 0.60))
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0


class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 45.0,
        max_retries: int = 4,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError("Thiếu OPENROUTER_API_KEY trong môi trường/.env")
        self.model = model or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/daongochan/K5-L3B-DAY07-2A202603779-DaoNgocHan-RetrievalContext",
            "X-Title": "Day7-Agentic-RAG-Lab",
        }

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if "error" in raw:
                    raise RuntimeError(f"OpenRouter API error: {raw['error']}")
                choice = raw["choices"][0]["message"]
                usage = raw.get("usage") or {}
                p_tok = int(usage.get("prompt_tokens", 0))
                c_tok = int(usage.get("completion_tokens", 0))
                cost = estimate_cost_usd(self.model, p_tok, c_tok, usage.get("cost"))
                return {
                    "message": choice,
                    "model": raw.get("model", self.model),
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "total_tokens": int(usage.get("total_tokens", p_tok + c_tok)),
                    "cost_usd": cost,
                    "elapsed_ms": elapsed_ms,
                }
            except urllib.error.HTTPError as exc:
                err_text = exc.read().decode("utf-8", errors="replace")
                last_err = RuntimeError(f"HTTP {exc.code}: {err_text[:400]}")
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise last_err from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_err = RuntimeError(f"Network/timeout error: {exc}")
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise last_err from exc

        assert last_err is not None
        raise last_err
