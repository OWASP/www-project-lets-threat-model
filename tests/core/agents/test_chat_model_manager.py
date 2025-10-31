import pytest
from unittest.mock import patch, MagicMock
from core.agents.chat_model_manager import (
    ChatModelManager,
    rate_limiter,
    set_provider_limits,
)
from pydantic import SecretStr
from langchain_core.language_models.chat_models import BaseChatModel


@pytest.fixture
def mock_init_chat_model():
    """Fixture to mock `init_chat_model`"""
    with patch("core.agents.chat_model_manager.init_chat_model") as mock:
        mock.return_value = MagicMock(spec=BaseChatModel)
        yield mock


@pytest.fixture(autouse=True)
def reset_provider_limits():
    """Ensure Anthropic limits are reset for each test run."""
    set_provider_limits(
        "anthropic",
        requests_per_minute=50,
        check_every_n_seconds=0.5,
        max_bucket_size=5,
        max_tokens=10000,
    )
    yield


@pytest.mark.parametrize(
    "provider, model, extra_kwargs, expected_call",
    [
        (
            "anthropic",
            "test_model",
            {"max_output_tokens": 2048},
            {
                "model_provider": "anthropic",
                "model": "test_model",
                "temperature": 0.0,
                "api_key": SecretStr("test_api_key"),
                "max_tokens": 2048,
                "rate_limiter_attrs": {
                    "requests_per_second": 50 / 60,
                    "check_every_n_seconds": 0.5,
                    "max_bucket_size": 5,
                },
            },
        ),
        (
            "openai",
            "test_model",
            {"max_output_tokens": 2048},
            {
                "model_provider": "openai",
                "model": "test_model",
                "temperature": 0.0,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "api_key": SecretStr("test_api_key"),
                "max_tokens": 2048,
                "rate_limiter_attrs": {
                    "requests_per_second": rate_limiter.requests_per_second,
                    "check_every_n_seconds": rate_limiter.check_every_n_seconds,
                    "max_bucket_size": rate_limiter.max_bucket_size,
                },
            },
        ),
        (
            "openai",
            "o3-test_model",
            {"max_output_tokens": 2048},
            {
                "model_provider": "openai",
                "model": "o3-test_model",
                "api_key": SecretStr("test_api_key"),
                "rate_limiter_attrs": {
                    "requests_per_second": rate_limiter.requests_per_second,
                    "check_every_n_seconds": rate_limiter.check_every_n_seconds,
                    "max_bucket_size": rate_limiter.max_bucket_size,
                },
            },
        ),
        (
            "ollama",
            "test_model",
            {},
            {
                "model_provider": "ollama",
                "model": "test_model",
                "temperature": 0.0,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "base_url": "http://localhost:11434",
                "rate_limiter_attrs": {
                    "requests_per_second": rate_limiter.requests_per_second,
                    "check_every_n_seconds": rate_limiter.check_every_n_seconds,
                    "max_bucket_size": rate_limiter.max_bucket_size,
                },
            },
        ),
    ],
)
def test_get_model(mock_init_chat_model, provider, model, extra_kwargs, expected_call):
    """Test model creation with different providers."""

    created_model = ChatModelManager.get_model(
        provider=provider,
        model=model,
        api_key=SecretStr("test_api_key"),
        **extra_kwargs,
    )

    mock_init_chat_model.assert_called_once()
    _, kwargs = mock_init_chat_model.call_args

    expected_rl_attrs = expected_call.pop("rate_limiter_attrs", None)
    rate_limiter_arg = kwargs.pop("rate_limiter")
    if expected_rl_attrs:
        assert pytest.approx(rate_limiter_arg.requests_per_second) == pytest.approx(
            expected_rl_attrs["requests_per_second"]
        )
        assert pytest.approx(rate_limiter_arg.check_every_n_seconds) == pytest.approx(
            expected_rl_attrs["check_every_n_seconds"]
        )
        assert rate_limiter_arg.max_bucket_size == expected_rl_attrs["max_bucket_size"]

    for key, value in expected_call.items():
        assert kwargs.get(key) == value
    assert isinstance(created_model, BaseChatModel)


def test_get_model_exception(mock_init_chat_model):
    """Test exception handling in `get_model`."""
    mock_init_chat_model.side_effect = Exception("Test exception")

    with pytest.raises(
        RuntimeError, match="❌ Failed to create model instance: Test exception"
    ):
        ChatModelManager.get_model(provider="openai", model="test_model")
