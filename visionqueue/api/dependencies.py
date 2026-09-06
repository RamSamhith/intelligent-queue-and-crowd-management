"""FastAPI dependency injection for the VisionQueue REST API.

Provides a singleton CVService instance that is shared across all request
handlers. The service is initialized once and accessed via get_cv_service().
"""

from __future__ import annotations

from typing import Optional
from fastapi import HTTPException

from visionqueue.api.service import CVService
from visionqueue.pipeline import CVPipeline, CVPipelineConfig
from visionqueue.persistence.config import PersistenceConfig
from visionqueue.persistence.repository import PersistenceRepository

_cv_service: Optional[CVService] = None


def init_cv_service(
    pipeline: Optional[CVPipeline] = None,
    config: Optional[CVPipelineConfig] = None,
    persistence_config: Optional[PersistenceConfig] = None,
) -> CVService:
    global _cv_service
    _cv_service = CVService(pipeline=pipeline, config=config, persistence_config=persistence_config)
    return _cv_service


def get_cv_service() -> CVService:
    if _cv_service is None:
        raise RuntimeError("CVService not initialized. Call init_cv_service() before using the API.")
    return _cv_service


def get_persistence_repository() -> PersistenceRepository:
    svc = get_cv_service()
    if not svc._persistence_config.enabled or not svc._repo:
        raise HTTPException(status_code=503, detail="Persistence is disabled or unavailable")
    return svc._repo


def reset_cv_service() -> None:
    global _cv_service
    _cv_service = None
