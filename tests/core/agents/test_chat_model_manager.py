import pytest
from unittest.mock import MagicMock, patch

from core.agents.chat_model_manager import ChatModelManager
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr


@pytest.fixture
def mock_constructors():
    with (
        patch("core.agents.chat_model_manager.ChatOpenAI") as mock_openai,
        patch("core.agents.chat_model_manager.ChatAnthropic") as mock_anthropic,
        patch("core.agents.chat_model_manager.init_chat_model") as mock_init,
    ):
        mock_openai.return_value = MagicMock(spec=BaseChatModel)
        mock_anthropic.return_value = MagicMock(spec=BaseChatModel)
        mock_init.return_value = MagicMock(spec=BaseChatModel)
        yield mock_openai, mock_anthropic, mock_init


def _extract_rate_limiter(call_kwargs):
    limiter = call_kwargs.get("rate_limiter")
    assert isinstance(limiter, InMemoryRateLimiter)
    return limiter


def test_get_model_openai_standard(mock_constructors):
    mock_openai, mock_anthropic, mock_init = mock_constructors

    model = ChatModelManager.get_model(
        provider="openai",
        model="gpt-4o",
        api_key=SecretStr("test_api_key"),
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        max_output_tokens=2048,
    )

    mock_openai.assert_called_once()
    mock_anthropic.assert_not_called()
    mock_init.assert_not_called()

    kwargs = mock_openai.call_args.kwargs
    limiter = _extract_rate_limiter(kwargs)
    assert pytest.approx(limiter.requests_per_second) == pytest.approx(400 / 60)
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["api_key"] == SecretStr("test_api_key")
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    assert kwargs["frequency_penalty"] == 0.1
    assert kwargs["presence_penalty"] == 0.2
    assert kwargs["max_tokens"] == 2048

    assert hasattr(model, "base_model")
    assert model.base_model is mock_openai.return_value


def test_get_model_openai_reasoning(mock_constructors):
    mock_openai, _, _ = mock_constructors

    ChatModelManager.get_model(
        provider="openai",
        model="o3-mini",
        api_key=SecretStr("k"),
        temperature=0.5,
        top_p=0.8,
    )

    kwargs = mock_openai.call_args.kwargs
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "frequency_penalty" not in kwargs
    assert "presence_penalty" not in kwargs


def test_get_model_anthropic(mock_constructors):
    _, mock_anthropic, _ = mock_constructors

    model = ChatModelManager.get_model(
        provider="anthropic",
        model="claude-3-5-sonnet",
        api_key=SecretStr("anthropic-key"),
        temperature=0.2,
        max_output_tokens=9000,
    )

    mock_anthropic.assert_called_once()
    kwargs = mock_anthropic.call_args.kwargs
    limiter = _extract_rate_limiter(kwargs)
    assert pytest.approx(limiter.requests_per_second) == pytest.approx(50 / 60)
    assert kwargs["max_tokens"] == 9000
    assert kwargs["temperature"] == 0.2
    assert "top_p" not in kwargs

    assert model.base_model is mock_anthropic.return_value


def test_get_model_fallback_provider(mock_constructors):
    mock_openai, mock_anthropic, mock_init = mock_constructors

    ChatModelManager.get_model(
        provider="ollama",
        model="llama3",
        base_url="http://localhost:9001",
    )

    mock_init.assert_called_once()
    kwargs = mock_init.call_args.kwargs
    limiter = _extract_rate_limiter(kwargs)
    assert pytest.approx(limiter.requests_per_second) == pytest.approx(60 / 60)
    assert kwargs["model_provider"] == "ollama"
    assert kwargs["base_url"] == "http://localhost:9001"

    mock_openai.assert_not_called()
    mock_anthropic.assert_not_called()


def test_get_model_exception(mock_constructors):
    mock_openai, _, _ = mock_constructors
    mock_openai.side_effect = Exception("boom")

    with pytest.raises(RuntimeError, match="Failed to create model instance: boom"):
        ChatModelManager.get_model(
            provider="openai",
            model="gpt-4o",
            api_key=SecretStr("k"),
        )
