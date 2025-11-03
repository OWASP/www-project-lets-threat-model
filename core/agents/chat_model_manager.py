import re
from copy import deepcopy
from typing import Optional, Dict, Any

from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

DEFAULT_RETRY_PARAMS: Dict[str, Any] = {
    "stop_after_attempt": 5,
    "wait_exponential_jitter": True,
    "exponential_jitter_params": {"initial": 1.0, "max": 60.0},
}

PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "rate_limiter_kwargs": {
            "requests_per_second": 400 / 60,
            "check_every_n_seconds": 0.1,
            "max_bucket_size": 10,
        },
    },
    "anthropic": {
        "rate_limiter_kwargs": {
            "requests_per_second": 50 / 60,
            "check_every_n_seconds": 0.1,
            "max_bucket_size": 5,
        },
    },
}

FALLBACK_DEFAULTS: Dict[str, Any] = {
    "rate_limiter_kwargs": {
        "requests_per_second": 60 / 60,
        "check_every_n_seconds": 0.1,
        "max_bucket_size": 5,
    },
}

DEFAULT_RATE_LIMITER = InMemoryRateLimiter(**FALLBACK_DEFAULTS["rate_limiter_kwargs"])
# Backwards-compatible alias for existing imports/tests
rate_limiter = DEFAULT_RATE_LIMITER

USER_RATE_LIMIT_OVERRIDE: Dict[str, Any] = {}


def _provider_config(provider_key: str) -> Dict[str, Any]:
    return PROVIDER_DEFAULTS.setdefault(
        provider_key,
        {
            "rate_limiter_kwargs": deepcopy(FALLBACK_DEFAULTS["rate_limiter_kwargs"]),
        },
    )


def _apply_limit_updates(
    target: Dict[str, Any],
    *,
    requests_per_minute: Optional[float] = None,
    check_every_n_seconds: Optional[float] = None,
    max_bucket_size: Optional[int] = None,
) -> None:
    if requests_per_minute is not None:
        target["requests_per_second"] = max(float(requests_per_minute) / 60.0, 0.01)
    if check_every_n_seconds is not None:
        target["check_every_n_seconds"] = max(float(check_every_n_seconds), 0.01)
    if max_bucket_size is not None:
        target["max_bucket_size"] = max(int(max_bucket_size), 1)


class RetryableChatModel:
    """
    Lightweight proxy that preserves the chat model interface while guaranteeing
    that every runnable returned from it (or invoked on it) carries the shared
    retry policy.
    """

    _RETRY_METHODS = {"invoke", "ainvoke", "batch", "abatch"}
    _RUNNABLE_BUILDERS = {
        "bind_tools",
        "with_structured_output",
        "bind",
        "with_config",
        "with_fallbacks",
    }

    def __init__(self, model: BaseChatModel, retry_params: Dict[str, Any]):
        self._model = model
        self._retry_params = retry_params
        self._retry_wrapper = None

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"RetryableChatModel(model={self._model!r})"

    @property
    def base_model(self) -> BaseChatModel:
        """Expose the underlying chat model instance."""
        return self._model

    def __getattr__(self, item: str):
        base_attr = getattr(self._model, item)

        if item in self._RETRY_METHODS:
            return getattr(self._retryable(), item)

        if item in self._RUNNABLE_BUILDERS:
            return self._wrap_call(base_attr)

        # Delegate all other attributes to the underlying model (e.g. get_num_tokens).
        return base_attr

    def _retryable(self):
        if self._retry_wrapper is None:
            self._retry_wrapper = self._model.with_retry(**self._retry_params)
        return self._retry_wrapper

    def _wrap_runnable(self, runnable: Any):
        if hasattr(runnable, "with_retry"):
            return runnable.with_retry(**self._retry_params)
        return runnable

    def _wrap_call(self, method):
        def inner(*args, **kwargs):
            runnable = method(*args, **kwargs)
            return self._wrap_runnable(runnable)

        return inner


def _provider_defaults(provider_key: str) -> Dict[str, Any]:
    provider_config = _provider_config(provider_key)
    rl_kwargs = deepcopy(FALLBACK_DEFAULTS["rate_limiter_kwargs"])
    rl_kwargs.update(provider_config.get("rate_limiter_kwargs", {}))
    if USER_RATE_LIMIT_OVERRIDE:
        rl_kwargs.update(USER_RATE_LIMIT_OVERRIDE)
    return {
        "rate_limiter_kwargs": rl_kwargs,
    }


def set_rate_limits(
    *,
    requests_per_minute: Optional[float] = None,
    check_every_n_seconds: Optional[float] = None,
    max_bucket_size: Optional[int] = None,
) -> None:
    """
    Set global rate limit overrides applied to every provider.
    """
    USER_RATE_LIMIT_OVERRIDE.clear()
    temp: Dict[str, Any] = {}
    _apply_limit_updates(
        temp,
        requests_per_minute=requests_per_minute,
        check_every_n_seconds=check_every_n_seconds,
        max_bucket_size=max_bucket_size,
    )
    USER_RATE_LIMIT_OVERRIDE.update(temp)


def clear_rate_limits() -> None:
    """Clear any global rate limit overrides."""
    USER_RATE_LIMIT_OVERRIDE.clear()


def _init_anthropic_model(
    *,
    model: str,
    api_key: Optional[SecretStr],
    rate_limiter: InMemoryRateLimiter,
    temperature: float,
    max_output_tokens: Optional[int],
) -> ChatAnthropic:
    init_kwargs = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
        "rate_limiter": rate_limiter,
    }
    if max_output_tokens is not None:
        init_kwargs["max_tokens"] = int(max_output_tokens)
    return ChatAnthropic(**init_kwargs)


def _init_openai_model(
    *,
    model: str,
    api_key: Optional[SecretStr],
    rate_limiter: InMemoryRateLimiter,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_output_tokens: Optional[int],
) -> ChatOpenAI:
    is_reasoning_model = bool(re.match(r"o\d-", model.lower()))
    init_kwargs = {
        "model": model,
        "api_key": api_key,
        "rate_limiter": rate_limiter,
    }
    if not is_reasoning_model:
        init_kwargs.update(
            {
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }
        )
        if max_output_tokens is not None:
            init_kwargs["max_tokens"] = int(max_output_tokens)
    return ChatOpenAI(**init_kwargs)


def _init_generic_model(
    *,
    provider_key: str,
    provider: str,
    model: str,
    api_key: Optional[SecretStr],
    base_url: Optional[str],
    rate_limiter: InMemoryRateLimiter,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_output_tokens: Optional[int],
) -> BaseChatModel:
    init_kwargs: Dict[str, Any] = {
        "model_provider": provider,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "rate_limiter": rate_limiter,
    }
    if api_key is not None:
        init_kwargs["api_key"] = api_key
    if provider_key == "ollama" and base_url is not None:
        init_kwargs["base_url"] = base_url
    if max_output_tokens is not None:
        init_kwargs["max_tokens"] = int(max_output_tokens)
    return init_chat_model(**init_kwargs)


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
            rate_limiter: Optional rate limiter

        Returns:
            LangChain-compatible chat model instance with retry functionality
        """
        try:
            provider_key = provider.lower()
            defaults = _provider_defaults(provider_key)
            selected_rate_limiter = rate_limiter or InMemoryRateLimiter(
                **defaults["rate_limiter_kwargs"]
            )

            if provider_key == "anthropic":
                model_instance = _init_anthropic_model(
                    model=model,
                    api_key=api_key,
                    rate_limiter=selected_rate_limiter,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            elif provider_key == "openai":
                model_instance = _init_openai_model(
                    model=model,
                    api_key=api_key,
                    rate_limiter=selected_rate_limiter,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_output_tokens=max_output_tokens,
                )
            else:
                model_instance = _init_generic_model(
                    provider_key=provider_key,
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    rate_limiter=selected_rate_limiter,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_output_tokens=max_output_tokens,
                )

            return RetryableChatModel(
                model=model_instance, retry_params=DEFAULT_RETRY_PARAMS
            )

        except Exception as e:  # pragma: no cover - surface helpful message
            raise RuntimeError(f"❌ Failed to create model instance: {str(e)}")
