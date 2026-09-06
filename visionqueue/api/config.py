"""API-specific configuration for the VisionQueue REST API.

Separates API concerns (host, port, CORS) from CV pipeline configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List
from visionqueue.persistence.config import PersistenceConfig


def _default_cors_origins() -> List[str]:
    raw = os.environ.get("VISIONQUEUE_CORS_ORIGINS")
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return ["http://localhost:3000", "http://localhost:5173"]


@dataclass
class APIConfig:
    """Configuration for the FastAPI REST API layer.

    Attributes:
        host: Bind address for the uvicorn server.
        port: Port for the uvicorn server.
        cors_origins: Allowed CORS origins. Empty list = no CORS middleware.
        api_prefix: Prefix for versioned endpoints (e.g. "/api/v1").
        debug: Enable FastAPI debug mode and verbose error responses.
        persistence: Configuration for database and storage persistence.
    """

    host: str = field(
        default_factory=lambda: os.environ.get("VISIONQUEUE_API_HOST", "127.0.0.1")
    )
    port: int = field(
        default_factory=lambda: int(os.environ.get("VISIONQUEUE_API_PORT", "8000"))
    )
    cors_origins: List[str] = field(default_factory=_default_cors_origins)
    api_prefix: str = "/api/v1"
    debug: bool = field(
        default_factory=lambda: os.environ.get("VISIONQUEUE_API_DEBUG", "false").lower()
        in ("1", "true", "yes")
    )
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
