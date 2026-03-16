"""FastAPI middleware – request logging, timing, and CORS configuration."""

from __future__ import annotations

import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs request method, path, status code, and duration for every request."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000
        logger.info(
            "%s %s → %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"
        return response
