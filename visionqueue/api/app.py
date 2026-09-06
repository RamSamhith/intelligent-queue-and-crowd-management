"""FastAPI application factory for the VisionQueue REST API.

Wires configuration, CORS middleware, dependency injection, and routers.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from visionqueue.api.config import APIConfig
from visionqueue.api.dependencies import init_cv_service
from visionqueue.api.routes import api_router, health_router, history_router, ws_router
from visionqueue.api.service import CVService

logger = logging.getLogger(__name__)


def create_app(
    api_config: Optional[APIConfig] = None,
    cv_service: Optional[CVService] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        api_config: API configuration. Defaults to APIConfig().
        cv_service: Pre-configured CVService instance. If None, one will be
            created with default configuration.

    Returns:
        Configured FastAPI application instance.
    """
    cfg = api_config or APIConfig()

    app = FastAPI(
        title="VisionQueue API",
        description="REST and WebSocket API for the Intelligent Queue & Crowd Management system.",
        version="1.0.0",
        debug=cfg.debug,
    )

    # CORS middleware
    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Dependency injection — initialize singleton CVService
    if cv_service is not None:
        from visionqueue.api import dependencies
        dependencies._cv_service = cv_service
    else:
        init_cv_service(persistence_config=cfg.persistence)

    # Mount routers
    app.include_router(health_router)
    app.include_router(api_router)
    app.include_router(history_router)
    app.include_router(ws_router)

    return app
