"""Token usage tracking and LLM cost estimation.

Encapsulates all token counting, cost estimation, and model invocation
with usage tracking into a composable TokenTracker class.
"""

import os
from time import sleep
from typing import Any, Dict, Optional, Tuple

from langchain.callbacks import get_openai_callback

from config import LLM_INVOKE_MAX_ATTEMPTS, OPENAI_RATE_CARD


class TokenTracker:
    """Tracks cumulative token usage and estimated costs across LLM calls."""

    def __init__(self):
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }

    def reset(self):
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }

    @staticmethod
    def extract_usage_from_result(result: Any) -> Dict[str, int]:
        def _to_int(value: Any) -> int:
            try:
                return int(value)
            except Exception:
                return 0

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        usage_meta = getattr(result, "usage_metadata", None)
        if isinstance(usage_meta, dict):
            prompt_tokens = _to_int(usage_meta.get("input_tokens") or usage_meta.get("prompt_tokens"))
            completion_tokens = _to_int(usage_meta.get("output_tokens") or usage_meta.get("completion_tokens"))
            total_tokens = _to_int(usage_meta.get("total_tokens"))

        response_meta = getattr(result, "response_metadata", None)
        if isinstance(response_meta, dict):
            token_usage = response_meta.get("token_usage", {})
            if isinstance(token_usage, dict):
                if prompt_tokens <= 0:
                    prompt_tokens = _to_int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens"))
                if completion_tokens <= 0:
                    completion_tokens = _to_int(token_usage.get("completion_tokens") or token_usage.get("output_tokens"))
                if total_tokens <= 0:
                    total_tokens = _to_int(token_usage.get("total_tokens"))

        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens

        return {
            "prompt_tokens": max(0, prompt_tokens),
            "completion_tokens": max(0, completion_tokens),
            "total_tokens": max(0, total_tokens),
        }

    @staticmethod
    def resolve_model_name(result: Any, base_model: Any = None) -> str:
        response_meta = getattr(result, "response_metadata", None)
        if isinstance(response_meta, dict):
            model_name = str(response_meta.get("model_name") or response_meta.get("model") or "").strip()
            if model_name:
                return model_name
        if base_model is not None:
            model_name = str(getattr(base_model, "model_name", "") or getattr(base_model, "model", "")).strip()
            return model_name
        return ""

    @staticmethod
    def resolve_openai_rates_per_1m(model_name: str) -> Tuple[Optional[float], Optional[float]]:
        input_override = os.getenv("OPENAI_INPUT_COST_PER_1M")
        output_override = os.getenv("OPENAI_OUTPUT_COST_PER_1M")
        if input_override and output_override:
            try:
                return float(input_override), float(output_override)
            except (ValueError, TypeError) as e:
                print(f"[TOKEN] Invalid cost override env vars: {e}")

        key = str(model_name or "").strip().lower()
        for prefix, rates in OPENAI_RATE_CARD.items():
            if key.startswith(prefix):
                return rates

        return None, None

    @staticmethod
    def estimate_openai_cost(prompt_tokens: int, completion_tokens: int, model_name: str) -> Optional[float]:
        in_rate, out_rate = TokenTracker.resolve_openai_rates_per_1m(model_name)
        if in_rate is None or out_rate is None:
            return None
        return (float(prompt_tokens) * in_rate + float(completion_tokens) * out_rate) / 1_000_000.0

    def invoke_model_with_usage(self, model, message, tag, base_model=None):
        """Invokes an LLM model with retry logic and accumulates usage tracking."""
        max_attempts = LLM_INVOKE_MAX_ATTEMPTS
        for attempt in range(1, max_attempts + 1):
            try:
                with get_openai_callback() as cb:
                    result = model.invoke(message)
                break
            except Exception as e:
                if attempt >= max_attempts:
                    raise
                wait_s = 2 * attempt
                print(f"[!] {tag} invoke failed (attempt {attempt}/{max_attempts}): {e}. Retrying in {wait_s}s")
                sleep(wait_s)

        prompt_tokens = int(cb.prompt_tokens or 0)
        completion_tokens = int(cb.completion_tokens or 0)
        total_tokens = int(cb.total_tokens or 0)
        estimated_cost = float(cb.total_cost or 0.0)

        if prompt_tokens <= 0 and completion_tokens <= 0:
            usage = self.extract_usage_from_result(result)
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            total_tokens = int(usage.get("total_tokens", 0))

        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens

        model_name = self.resolve_model_name(result, base_model)
        if estimated_cost <= 0.0 and (prompt_tokens > 0 or completion_tokens > 0):
            estimated = self.estimate_openai_cost(prompt_tokens, completion_tokens, model_name)
            if estimated is not None:
                estimated_cost = estimated

        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.usage["total_tokens"] += total_tokens
        self.usage["total_cost"] += estimated_cost

        print(f"TOKEN USAGE ({tag}):")
        print(f"   Model: {model_name or 'unknown'}")
        print(f"   Prompt tokens: {prompt_tokens:,}")
        print(f"   Completion tokens: {completion_tokens:,}")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Estimated cost: ${estimated_cost:.6f}")

        return result

    def print_total_usage(self):
        t = self.usage
        print("TOTAL TOKEN USAGE:")
        print(f"   Prompt tokens: {t['prompt_tokens']:,}")
        print(f"   Completion tokens: {t['completion_tokens']:,}")
        print(f"   Total tokens: {t['total_tokens']:,}")
        print(f"   Estimated cost: ${t['total_cost']:.6f}")
