"""Access control and rate limiting for a small shared deployment.

Scope, stated honestly: this stops drive-by abuse and accidental quota burn by
people you shared a link with. It is not protection against a determined
attacker, and it is not multi-tenancy.

  RESEARCHOS_API_TOKEN     when set, every request must carry it. Unset =
                           open, which is fine on a private network and not
                           fine on the public internet.
  RESEARCHOS_ALLOWED_ORIGINS  comma-separated origins for CORS. Defaults to
                           localhost so a fresh checkout works; set it to your
                           real frontend origin before deploying.
  RESEARCHOS_RATE_LIMIT    requests per minute per client. Default 20 - enough
                           for a person, not enough to drain a free Groq tier.

The rate limiter is in-process, so it resets on restart and does not coordinate
across replicas. With one process serving a handful of people that is exactly
what is needed; anything larger wants Redis and a real gateway.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"


def api_token() -> str | None:
    token = os.environ.get("RESEARCHOS_API_TOKEN", "").strip()
    return token or None


def allowed_origins() -> list[str]:
    raw = os.environ.get("RESEARCHOS_ALLOWED_ORIGINS", DEFAULT_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]


def rate_limit_per_minute() -> int:
    try:
        return max(1, int(os.environ.get("RESEARCHOS_RATE_LIMIT", "20")))
    except ValueError:
        return 20


class RateLimiter:
    """Fixed-window-per-client limiter using a deque of timestamps."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.time()
        with self._lock:
            window = self._hits[client]
            while window and now - window[0] > 60.0:
                window.popleft()
            if len(window) >= self.per_minute:
                return False, int(61 - (now - window[0]))
            window.append(now)
            # Keep the table from growing without bound on a long-lived process.
            if len(self._hits) > 5000:
                for key in [k for k, v in self._hits.items() if not v][:1000]:
                    self._hits.pop(key, None)
            return True, 0


_limiter = RateLimiter(rate_limit_per_minute())


def client_id(request: Request) -> str:
    """Best-effort client identity. Honours one proxy hop."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def guard(request: Request) -> None:
    """FastAPI dependency: enforce the token and the rate limit."""
    if request.method == "OPTIONS" or request.url.path in ("/health", "/docs", "/openapi.json"):
        return

    token = api_token()
    if token:
        supplied = request.headers.get("authorization", "")
        if supplied.startswith("Bearer "):
            supplied = supplied[7:]
        else:
            supplied = request.headers.get("x-api-token", "")
        if supplied.strip() != token:
            raise HTTPException(status_code=401, detail="Invalid or missing API token")

    allowed, retry_after = _limiter.check(client_id(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit is {_limiter.per_minute} requests/minute. Retry in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )
