import logging
import re
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Optional, Dict, Any
from pydantic import SecretStr


logger = logging.getLogger(__name__)

# Default rate limiter (used when no provider-specific config is supplied)
DEFAULT_RATE_LIMIT_PARAMS = {
    "requests_per_second": 7,  # Aligns with OpenAI's 500 req/min default
    "check_every_n_seconds": 0.1,
    "max_bucket_size": 10,
}
DEFAULT_RATE_LIMITER = InMemoryRateLimiter(**DEFAULT_RATE_LIMIT_PARAMS)
# Backwards-compatible alias for existing imports/tests
rate_limiter = DEFAULT_RATE_LIMITER

_PROVIDER_LIMIT_PARAMS: Dict[str, Dict[str, Any]] = {
    "anthropic": {
        "requests_per_second": 50 / 60,  # 50 RPM default across Anthropic plans
        "check_every_n_seconds": 0.5,
        "max_bucket_size": 5,
        "model_caps": {
            "default": 10000,
            "claude-3-5-sonnet": 8000,
            "claude-3-sonnet": 8000,
            "claude-4.1-sonnet": 8000,
            "claude-4.1-opus": 8000,
            "claude-3-5-opus": 8000,
        },
    }
}
_PROVIDER_RATE_LIMITERS: dict[str, InMemoryRateLimiter] = {}


def set_provider_limits(
    provider: str,
    *,
    requests_per_minute: Optional[float] = None,
    requests_per_second: Optional[float] = None,
    check_every_n_seconds: Optional[float] = None,
    max_bucket_size: Optional[int] = None,
    max_tokens: Optional[int] = None,
    model_caps: Optional[Dict[str, int]] = None,
) -> None:
    """
    Configure rate limiting and token caps for a given provider.

    Args:
        provider: Provider name (case-insensitive).
        requests_per_minute: Optional requests/minute override. Wins over requests_per_second if both provided.
        requests_per_second: Optional requests/second override.
        check_every_n_seconds: Frequency that the limiter wakes to replenish tokens.
        max_bucket_size: Burst capacity for the limiter.
        max_tokens: Per-request token ceiling to apply when invoking the provider.
    """

    provider_key = provider.lower()
    params = _PROVIDER_LIMIT_PARAMS.get(provider_key, {}).copy()

    if requests_per_minute is not None:
        requests_per_second = requests_per_minute / 60.0

    if requests_per_second is not None:
        params["requests_per_second"] = max(requests_per_second, 0.01)
    if check_every_n_seconds is not None:
        params["check_every_n_seconds"] = max(check_every_n_seconds, 0.01)
    if max_bucket_size is not None:
        params["max_bucket_size"] = max(int(max_bucket_size), 1)
    if model_caps is not None:
        params["model_caps"] = {k: max(int(v), 1) for k, v in model_caps.items()}
    if max_tokens is not None:
        params.setdefault("model_caps", {})
        params["model_caps"]["default"] = max(int(max_tokens), 1)

    _PROVIDER_LIMIT_PARAMS[provider_key] = params
    # Discard any cached limiter so it reuses the updated params next time.
    _PROVIDER_RATE_LIMITERS.pop(provider_key, None)


def _get_provider_limits(provider: str) -> Dict[str, Any]:
    return _PROVIDER_LIMIT_PARAMS.get(provider.lower(), {})


def _get_rate_limiter(provider: str) -> InMemoryRateLimiter:
    provider_key = provider.lower()
    if provider_key not in _PROVIDER_RATE_LIMITERS:
        params = _get_provider_limits(provider_key)
        if params:
            limiter_kwargs = {
                "requests_per_second": params.get("requests_per_second")
                or DEFAULT_RATE_LIMIT_PARAMS["requests_per_second"],
                "check_every_n_seconds": params.get("check_every_n_seconds")
                or DEFAULT_RATE_LIMIT_PARAMS["check_every_n_seconds"],
                "max_bucket_size": int(
                    params.get("max_bucket_size")
                    or DEFAULT_RATE_LIMIT_PARAMS["max_bucket_size"]
                ),
            }
            _PROVIDER_RATE_LIMITERS[provider_key] = InMemoryRateLimiter(
                **limiter_kwargs
            )
        else:
            _PROVIDER_RATE_LIMITERS[provider_key] = DEFAULT_RATE_LIMITER
    return _PROVIDER_RATE_LIMITERS[provider_key]


def _resolve_model_token_cap(limits: Dict[str, Any], model: str) -> Optional[int]:
    if not limits:
        return None
    model_caps = limits.get("model_caps")
    if isinstance(model_caps, dict):
        # Exact match
        if model in model_caps:
            return int(model_caps[model])
        # Prefix match for family names
        for key, value in model_caps.items():
            if key == "default":
                continue
            if model.startswith(key):
                return int(value)
        default_cap = model_caps.get("default")
        return int(default_cap) if default_cap is not None else None
    return None


class ChatModelManager:
    @classmethod
    def get_model(
        cls,
        provider: str,
        model: str,
        base_url: Optional[str] = "http://localhost:11434",
        api_key: Optional[SecretStr] = None,
        top_p: float = 1.0,
        temperature: float = 0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        rate_limiter: Optional[InMemoryRateLimiter] = None,
        max_output_tokens: Optional[int] = None,
    ) -> BaseChatModel:
        """
        Get a chat model dynamically based on the provider with rate limiting & retries.

        Args:
            provider: The model provider ('llama', 'openai', 'anthropic')
            model: The model name to use
            api_key: API key for the provider (if required)
            temperature: Temperature setting for response randomness
            max_output_tokens: Provider-specific hard limit on generated tokens.
            rate_limiter: Optional rate limiter
            max_retries: Maximum number of retry attempts

        Returns:
            LangChain-compatible chat model instance with retry functionality
        """
        try:
            provider_key = provider.lower()
            limits = _get_provider_limits(provider_key)
            selected_rate_limiter = (
                rate_limiter if rate_limiter is not None else _get_rate_limiter(provider_key)
            )
            max_token_cap = _resolve_model_token_cap(limits, model)
            requested_max_tokens = max_output_tokens

            init_kwargs = {
                "model_provider": provider,
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "rate_limiter": selected_rate_limiter,
            }
            if provider_key == "anthropic":
                init_kwargs["api_key"] = api_key

                del init_kwargs["frequency_penalty"]
                del init_kwargs["presence_penalty"]
                # Anthropic API rejects requests when both temperature and top_p are set.
                if "top_p" in init_kwargs:
                    del init_kwargs["top_p"]
                if requested_max_tokens is not None:
                    capped_tokens = (
                        min(requested_max_tokens, max_token_cap)
                        if max_token_cap is not None
                        else requested_max_tokens
                    )
                    init_kwargs["max_tokens"] = capped_tokens
            elif provider.lower() == "openai":
                init_kwargs["api_key"] = api_key

                is_reasoning_model = bool(re.match(r"o\d-", model.lower()))
                if is_reasoning_model:
                    del init_kwargs["top_p"]
                    del init_kwargs["frequency_penalty"]
                    del init_kwargs["presence_penalty"]
                    del init_kwargs["temperature"]
                elif requested_max_tokens is not None:
                    init_kwargs["max_tokens"] = requested_max_tokens

            elif provider.lower() == "ollama":
                init_kwargs["base_url"] = base_url

            return init_chat_model(**init_kwargs)

        except Exception as e:
            raise RuntimeError(f"❌ Failed to create model instance: {str(e)}")
