"""REST API and WebSocket endpoint handlers for the VisionQueue REST API.

All handlers are thin: they read cached state from CVService and return it.
No CV logic, inference, or state computation happens here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from visionqueue.api.dependencies import get_cv_service, get_persistence_repository
from visionqueue.api.schemas import (
    AlertHistoryResponse,
    AlertResponse,
    CountsResponse,
    CrowdResponse,
    ErrorResponse,
    EventResponse,
    HealthResponse,
    LiveStateResponse,
    MeasurementHistoryResponse,
    OccupancyResponse,
    PipelineActionResponse,
    ReadinessResponse,
    SessionResponse,
    StatusResponse,
    TrackResponse,
)
from visionqueue.api.service import CVService
from visionqueue.persistence.repository import PersistenceRepository

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------

health_router = APIRouter(tags=["health"])
api_router = APIRouter(prefix="/api/v1", tags=["visionqueue"])
history_router = APIRouter(prefix="/api/v1", tags=["history"])
ws_router = APIRouter(tags=["websocket"])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _require_state(service: CVService) -> dict:
    """Return the cached LiveState dict, or raise 503 if unavailable."""
    state = service.get_latest_state()
    if not state:
        raise HTTPException(
            status_code=503,
            detail="CV pipeline has not produced state yet. Start the pipeline first.",
        )
    return state


# ------------------------------------------------------------------
# GET /health & GET /health/ready & GET /ready
# ------------------------------------------------------------------

@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="API process liveness",
    description="Returns simple API liveness. Does not reflect CV pipeline health.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@health_router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
    summary="Pipeline readiness check",
    description="Indicates whether the CV pipeline has started and produced usable state.",
)
@health_router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
    include_in_schema=False,
)
async def readiness(
    service: CVService = Depends(get_cv_service),
):
    diag = service.get_diagnostics()
    has_state = bool(service.get_latest_state())
    is_ready = bool(diag.get("pipeline_running", False) and has_state)

    resp = ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        ready=is_ready,
        detail="Pipeline is operational and emitting state" if is_ready else "Pipeline is stopped or has not produced state yet",
        pipeline_running=diag.get("pipeline_running", False),
        has_state=has_state,
        active_model=diag.get("active_model"),
        execution_provider=diag.get("execution_provider"),
        is_gpu=diag.get("is_gpu"),
        persistence_healthy=diag.get("persistence_healthy"),
    )
    if not is_ready:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content=resp.model_dump())
    return resp


# ------------------------------------------------------------------
# GET /api/v1/status
# ------------------------------------------------------------------

@api_router.get(
    "/status",
    response_model=StatusResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Pipeline and system status",
)
async def status(
    service: CVService = Depends(get_cv_service),
) -> StatusResponse:
    state = _require_state(service)
    diag = service.get_diagnostics()
    return StatusResponse(
        system_state=state["system_state"],
        camera_state=state["camera_state"],
        is_healthy=state["is_healthy"],
        is_frozen=state["is_frozen"],
        status_reason=state["status_reason"],
        performance=state["performance"],
        vision=state["vision"],
        active_model=diag.get("active_model"),
        execution_provider=diag.get("execution_provider"),
        is_gpu=diag.get("is_gpu"),
        persistence_healthy=diag.get("persistence_healthy"),
        pipeline_running=diag.get("pipeline_running"),
        last_frame_timestamp=state.get("timestamp"),
    )


# ------------------------------------------------------------------
# GET /api/v1/live
# ------------------------------------------------------------------

@api_router.get(
    "/live",
    response_model=LiveStateResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Complete current LiveState",
    description="Primary endpoint for frontend/dashboard clients.",
)
async def live(
    service: CVService = Depends(get_cv_service),
) -> dict:
    state = _require_state(service)
    return state


# ------------------------------------------------------------------
# GET /api/v1/counts
# ------------------------------------------------------------------

@api_router.get(
    "/counts",
    response_model=CountsResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Current counting data",
)
async def counts(
    service: CVService = Depends(get_cv_service),
) -> CountsResponse:
    state = _require_state(service)
    c = state["counts"]
    return CountsResponse(
        current=c["current"],
        track_instances=c["track_instances"],
        unique_session_approx=c["unique_session_approx"],
        entries=c["entries"],
        exits=c["exits"],
        net_count=c["net_count"],
    )


# ------------------------------------------------------------------
# GET /api/v1/occupancy
# ------------------------------------------------------------------

@api_router.get(
    "/occupancy",
    response_model=OccupancyResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Current occupancy data",
)
async def occupancy(
    service: CVService = Depends(get_cv_service),
) -> OccupancyResponse:
    state = _require_state(service)
    o = state["occupancy"]
    return OccupancyResponse(
        capacity=o["capacity"],
        capacity_state=o["capacity_state"],
        percent=o["percent"],
    )


# ------------------------------------------------------------------
# GET /api/v1/crowd
# ------------------------------------------------------------------

@api_router.get(
    "/crowd",
    response_model=CrowdResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Crowd analytics",
)
async def crowd(
    service: CVService = Depends(get_cv_service),
) -> CrowdResponse:
    state = _require_state(service)
    cr = state["crowd"]
    return CrowdResponse(
        level=cr["level"],
        raw_level=cr["raw_level"],
        trend=cr["trend"],
        peak_count=cr["peak_count"],
        peak_occupancy_percent=cr.get("peak_occupancy_percent"),
        peak_timestamp=cr.get("peak_timestamp"),
    )


# ------------------------------------------------------------------
# GET /api/v1/alerts
# ------------------------------------------------------------------

@api_router.get(
    "/alerts",
    response_model=List[AlertResponse],
    responses={503: {"model": ErrorResponse}},
    summary="Active alerts",
)
async def alerts(
    service: CVService = Depends(get_cv_service),
) -> list:
    state = _require_state(service)
    return state["alerts"]


# ------------------------------------------------------------------
# GET /api/v1/tracks
# ------------------------------------------------------------------

@api_router.get(
    "/tracks",
    response_model=List[TrackResponse],
    responses={503: {"model": ErrorResponse}},
    summary="Currently active tracks",
    description="Application-level track data. Does not expose internal tracker objects.",
)
async def tracks(
    service: CVService = Depends(get_cv_service),
) -> list:
    if not service.get_latest_state():
        raise HTTPException(
            status_code=503,
            detail="CV pipeline has not produced state yet. Start the pipeline first.",
        )
    return service.get_latest_tracks()


# ------------------------------------------------------------------
# POST /api/v1/pipeline/start
# ------------------------------------------------------------------

@api_router.post(
    "/pipeline/start",
    response_model=PipelineActionResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Start the CV pipeline",
)
async def pipeline_start(
    service: CVService = Depends(get_cv_service),
) -> PipelineActionResponse:
    try:
        was_running = service.is_running
        service.start()
        if was_running:
            return PipelineActionResponse(
                status="ok", message="Pipeline was already running."
            )
        return PipelineActionResponse(
            status="ok", message="Pipeline started."
        )
    except Exception as exc:
        logger.exception("Failed to start pipeline")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------
# POST /api/v1/pipeline/stop
# ------------------------------------------------------------------

@api_router.post(
    "/pipeline/stop",
    response_model=PipelineActionResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Stop the CV pipeline",
)
async def pipeline_stop(
    service: CVService = Depends(get_cv_service),
) -> PipelineActionResponse:
    try:
        was_running = service.is_running
        service.stop()
        if not was_running:
            return PipelineActionResponse(
                status="ok", message="Pipeline was already stopped."
            )
        return PipelineActionResponse(
            status="ok", message="Pipeline stopped."
        )
    except Exception as exc:
        logger.exception("Failed to stop pipeline")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------
# POST /api/v1/pipeline/reset
# ------------------------------------------------------------------

@api_router.post(
    "/pipeline/reset",
    response_model=PipelineActionResponse,
    responses={503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Reset pipeline session state",
)
async def pipeline_reset(
    service: CVService = Depends(get_cv_service),
) -> PipelineActionResponse:
    if not service.is_running:
        raise HTTPException(
            status_code=503,
            detail="Cannot reset: pipeline is not running.",
        )
    try:
        service.reset_session()
        return PipelineActionResponse(
            status="ok", message="Pipeline session reset."
        )
    except Exception as exc:
        logger.exception("Failed to reset pipeline session")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------
# Historical Data API
# ------------------------------------------------------------------

@history_router.get("/sessions", response_model=List[SessionResponse])
def get_sessions(
    limit: int = 100, 
    offset: int = 0,
    repo: PersistenceRepository = Depends(get_persistence_repository)
):
    limit = min(limit, 1000)
    return repo.get_sessions(limit=limit, offset=offset)


@history_router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(
    session_id: str,
    repo: PersistenceRepository = Depends(get_persistence_repository)
):
    session = repo.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@history_router.get("/history", response_model=List[MeasurementHistoryResponse])
def get_history(
    session_id: str,
    limit: int = 100,
    offset: int = 0,
    repo: PersistenceRepository = Depends(get_persistence_repository)
):
    limit = min(limit, 1000)
    return repo.get_history(session_id, limit=limit, offset=offset)


@history_router.get("/history/events", response_model=List[EventResponse])
def get_events(
    session_id: str,
    limit: int = 100,
    offset: int = 0,
    repo: PersistenceRepository = Depends(get_persistence_repository)
):
    limit = min(limit, 1000)
    return repo.get_events(session_id, limit=limit, offset=offset)


@history_router.get("/history/alerts", response_model=List[AlertHistoryResponse])
def get_alerts_history(
    session_id: str,
    limit: int = 100,
    offset: int = 0,
    repo: PersistenceRepository = Depends(get_persistence_repository)
):
    limit = min(limit, 1000)
    return repo.get_alerts(session_id, limit=limit, offset=offset)


# ------------------------------------------------------------------
# WebSocket Live Streaming
# ------------------------------------------------------------------

async def _stream_live_state(websocket: WebSocket):
    await websocket.accept()
    try:
        service = get_cv_service()
    except Exception:
        await websocket.send_json({"error": "CVService not initialized"})
        await websocket.close()
        return

    last_frame_id = -1
    try:
        while True:
            state = service.get_latest_state()
            if state:
                fid = state.get("frame_id", 0)
                if fid != last_frame_id:
                    await websocket.send_json(state)
                    last_frame_id = fid
            await asyncio.sleep(0.04)  # ~25 Hz update check
    except (WebSocketDisconnect, ConnectionResetError):
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.debug("WebSocket client connection closed: %s", exc)


@ws_router.websocket("/ws/live")
async def websocket_endpoint_root(websocket: WebSocket):
    """WebSocket streaming endpoint at /ws/live."""
    await _stream_live_state(websocket)


@api_router.websocket("/ws/live")
async def websocket_endpoint_v1(websocket: WebSocket):
    """WebSocket streaming endpoint at /api/v1/ws/live."""
    await _stream_live_state(websocket)


# ------------------------------------------------------------------
# WebSocket Tracks Streaming
# ------------------------------------------------------------------

async def _stream_tracks(websocket: WebSocket):
    await websocket.accept()
    try:
        service = get_cv_service()
    except Exception:
        await websocket.send_json({"error": "CVService not initialized"})
        await websocket.close()
        return

    last_frame_id = -1
    try:
        while True:
            state = service.get_latest_state()
            if state:
                fid = state.get("frame_id", 0)
                if fid != last_frame_id:
                    tracks = service.get_latest_tracks()
                    packet = {
                        "frame_id": fid,
                        "timestamp": state.get("timestamp", 0.0),
                        "tracks": tracks,
                    }
                    await websocket.send_json(packet)
                    last_frame_id = fid
            await asyncio.sleep(0.04)  # ~25 Hz update check
    except (WebSocketDisconnect, ConnectionResetError):
        logger.info("WebSocket tracks client disconnected")
    except Exception as exc:
        logger.debug("WebSocket tracks connection closed: %s", exc)


@ws_router.websocket("/ws/tracks")
async def websocket_tracks_root(websocket: WebSocket):
    """WebSocket tracks streaming endpoint at /ws/tracks."""
    await _stream_tracks(websocket)


@api_router.websocket("/ws/tracks")
async def websocket_tracks_v1(websocket: WebSocket):
    """WebSocket tracks streaming endpoint at /api/v1/ws/tracks."""
    await _stream_tracks(websocket)
