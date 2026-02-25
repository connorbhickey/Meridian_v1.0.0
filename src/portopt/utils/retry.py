"""Retry logic with exponential backoff for transient network failures."""

from __future__ import annotations

import logging
import random
import time
from functools import wraps
from typing import Callable, TypeVar

import requests

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions that are safe to retry (transient network issues)
TRANSIENT_EXCEPTIONS = (
    requests.ConnectionError,
    requests.Timeout,
    ConnectionError,
    TimeoutError,
    OSError,
)


def is_transient(exc: BaseException) -> bool:
    """Classify whether an exception is transient (worth retrying).

    Returns True for network errors, timeouts, and server errors (5xx).
    Returns False for client errors (4xx), auth failures, and data errors.
    """
    # Check HTTPError first (it inherits from OSError via ConnectionError)
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        # 429 = rate limit, 5xx = server errors → retry
        return status == 429 or status >= 500
    if isinstance(exc, TRANSIENT_EXCEPTIONS):
        return True
    return False


def retry_on_transient(
    max_retries: int = 2,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    jitter: bool = True,
) -> Callable:
    """Decorator: retry a function on transient network errors.

    Uses exponential backoff: delay = base_delay * 2^attempt + jitter.
    Only retries on transient errors (network, timeout, 5xx).
    Non-transient errors (4xx, ValueError) propagate immediately.

    Args:
        max_retries: Maximum number of retry attempts (0 = no retries).
        base_delay: Base delay in seconds between retries.
        max_delay: Maximum delay cap in seconds.
        jitter: Add random jitter to prevent thundering herd.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt >= max_retries or not is_transient(exc):
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    if jitter:
                        delay += random.uniform(0, delay * 0.25)
                    logger.info(
                        "Retry %d/%d for %s after %.1fs (%s: %s)",
                        attempt + 1,
                        max_retries,
                        fn.__qualname__,
                        delay,
                        type(exc).__name__,
                        exc,
                    )
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
