import asyncio
from contextlib import asynccontextmanager


_DEFAULT_LIMIT = 2
_current_limit = _DEFAULT_LIMIT
_semaphore = asyncio.Semaphore(_DEFAULT_LIMIT)


def set_concurrency_limit(limit: int) -> None:
    """
    Set the maximum number of concurrent LLM calls allowed.
    Resets the underlying semaphore to the new limit.
    """
    global _semaphore, _current_limit

    limit = max(1, limit)
    _current_limit = limit
    _semaphore = asyncio.Semaphore(limit)


def get_concurrency_limit() -> int:
    """Return the current semaphore capacity."""
    return _current_limit


@asynccontextmanager
async def llm_concurrency_guard():
    """Async context manager that enforces the configured concurrency limit."""
    await _semaphore.acquire()
    try:
        yield
    finally:
        _semaphore.release()
