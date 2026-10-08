from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class RemoteApiError(RuntimeError):
    pass


def get_json(url: str, params: dict[str, str] | None = None, timeout: int = 60) -> Any:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    return _request(url, timeout=timeout)


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: int = 60,
    headers: dict[str, str] | None = None,
    retries: int = 3,
) -> Any:
    body = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    return _request(
        url,
        timeout=timeout,
        data=body,
        headers=req_headers,
        retries=retries,
    )


def get_bytes(url: str, params: dict[str, str] | None = None, timeout: int = 300) -> bytes:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    return _request_bytes(url, timeout=timeout)


def _request(
    url: str,
    *,
    timeout: int,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    retries: int = 3,
) -> Any:
    raw = _request_bytes(
        url, timeout=timeout, data=data, headers=headers, retries=retries
    )
    if not raw.strip():
        return []
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RemoteApiError(f"Non-JSON response from {url}") from exc


def _request_bytes(
    url: str,
    *,
    timeout: int,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    retries: int = 3,
) -> bytes:
    req_headers = {"User-Agent": "agentic-radiogen/0.1"}
    if headers:
        req_headers.update(headers)
    else:
        req_headers["Accept"] = "*/*"
    request = urllib.request.Request(url, data=data, headers=req_headers)
    attempts = max(1, int(retries))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = RemoteApiError(f"{exc.code} {url}: {detail}")
            if exc.code not in {502, 503, 504} or attempt >= attempts - 1:
                raise last_error from exc
            time.sleep(1.5 * (attempt + 1))
        except urllib.error.URLError as exc:
            reason = str(exc.reason)
            last_error = RemoteApiError(f"Failed to reach {url}: {reason}")
            # Retry transient timeouts (common on WSL / slow Gemini free tier).
            if "timed out" in reason.lower() and attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise last_error from exc
        except TimeoutError as exc:
            last_error = RemoteApiError(f"Timed out contacting {url}")
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise last_error from exc
    raise last_error or RemoteApiError(f"Failed to reach {url}")
