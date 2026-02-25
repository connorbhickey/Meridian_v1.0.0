"""Tests for retry logic with exponential backoff."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import requests

from portopt.utils.retry import is_transient, retry_on_transient


# ── is_transient classification ──────────────────────────────────────

class TestIsTransient:
    def test_connection_error_is_transient(self):
        assert is_transient(requests.ConnectionError("Connection refused"))

    def test_timeout_is_transient(self):
        assert is_transient(requests.Timeout("Request timed out"))

    def test_os_error_is_transient(self):
        assert is_transient(OSError("Network unreachable"))

    def test_connection_reset_is_transient(self):
        assert is_transient(ConnectionError("Connection reset"))

    def test_http_500_is_transient(self):
        resp = requests.Response()
        resp.status_code = 500
        assert is_transient(requests.HTTPError(response=resp))

    def test_http_503_is_transient(self):
        resp = requests.Response()
        resp.status_code = 503
        assert is_transient(requests.HTTPError(response=resp))

    def test_http_429_rate_limit_is_transient(self):
        resp = requests.Response()
        resp.status_code = 429
        assert is_transient(requests.HTTPError(response=resp))

    def test_http_404_is_not_transient(self):
        resp = requests.Response()
        resp.status_code = 404
        assert not is_transient(requests.HTTPError(response=resp))

    def test_http_401_is_not_transient(self):
        resp = requests.Response()
        resp.status_code = 401
        assert not is_transient(requests.HTTPError(response=resp))

    def test_value_error_is_not_transient(self):
        assert not is_transient(ValueError("bad data"))

    def test_key_error_is_not_transient(self):
        assert not is_transient(KeyError("missing"))

    def test_runtime_error_is_not_transient(self):
        assert not is_transient(RuntimeError("something wrong"))


# ── retry_on_transient decorator ─────────────────────────────────────

class TestRetryDecorator:
    def test_no_retry_on_success(self):
        call_count = 0

        @retry_on_transient(max_retries=3, base_delay=0.01)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    def test_retries_on_transient_then_succeeds(self):
        call_count = 0

        @retry_on_transient(max_retries=3, base_delay=0.01, jitter=False)
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.ConnectionError("Connection refused")
            return "recovered"

        assert fail_twice() == "recovered"
        assert call_count == 3

    def test_exhausts_retries_then_raises(self):
        call_count = 0

        @retry_on_transient(max_retries=2, base_delay=0.01, jitter=False)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise requests.Timeout("timed out")

        with pytest.raises(requests.Timeout):
            always_fail()
        assert call_count == 3  # 1 original + 2 retries

    def test_no_retry_on_non_transient(self):
        call_count = 0

        @retry_on_transient(max_retries=3, base_delay=0.01)
        def bad_data():
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid symbol")

        with pytest.raises(ValueError):
            bad_data()
        assert call_count == 1  # No retries for ValueError

    def test_no_retry_on_404(self):
        call_count = 0

        @retry_on_transient(max_retries=3, base_delay=0.01)
        def not_found():
            nonlocal call_count
            call_count += 1
            resp = requests.Response()
            resp.status_code = 404
            raise requests.HTTPError(response=resp)

        with pytest.raises(requests.HTTPError):
            not_found()
        assert call_count == 1

    def test_retries_on_500(self):
        call_count = 0

        @retry_on_transient(max_retries=1, base_delay=0.01, jitter=False)
        def server_error():
            nonlocal call_count
            call_count += 1
            resp = requests.Response()
            resp.status_code = 500
            raise requests.HTTPError(response=resp)

        with pytest.raises(requests.HTTPError):
            server_error()
        assert call_count == 2  # 1 original + 1 retry

    def test_zero_retries(self):
        call_count = 0

        @retry_on_transient(max_retries=0, base_delay=0.01)
        def no_retries():
            nonlocal call_count
            call_count += 1
            raise requests.ConnectionError("fail")

        with pytest.raises(requests.ConnectionError):
            no_retries()
        assert call_count == 1

    def test_preserves_function_metadata(self):
        @retry_on_transient(max_retries=1)
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_backoff_timing(self):
        """Verify exponential backoff happens (not too fast)."""
        import time

        call_count = 0

        @retry_on_transient(max_retries=2, base_delay=0.05, jitter=False)
        def slow_fail():
            nonlocal call_count
            call_count += 1
            raise requests.Timeout("timeout")

        t0 = time.time()
        with pytest.raises(requests.Timeout):
            slow_fail()
        elapsed = time.time() - t0

        assert call_count == 3
        # base_delay=0.05: first retry 0.05s, second retry 0.1s = 0.15s min
        assert elapsed >= 0.12  # Allow small timing margin
