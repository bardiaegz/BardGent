"""Small, dependency-light helpers used across several modules."""

import time
import requests

from bardgent import config


def with_retries(func, *args, retries=3, backoff=1.5, **kwargs):
    """Retry a `requests`-based call on transient network errors."""
    last_exc = None
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except requests.RequestException as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    raise last_exc


def count_tokens(text):
    if not text:
        return 0
    return max(1, len(text) // 4)


def truncate_output(text):
    if len(text) <= config.MAX_TOOL_OUTPUT:
        return text
    return text[:config.MAX_TOOL_OUTPUT] + f"\n... [output truncated, {len(text) - config.MAX_TOOL_OUTPUT} more chars not shown]"
