# VisionQueue Computer Vision Subsystem Handoff & Public Contract Specification

**Author & Owner:** Ram Samhith — Computer Vision & System Integration  
**Subsystem:** Computer Vision & Real-Time Analytics Pipeline (`visionqueue`)  
**Status:** Frozen & Integration-Ready (`v1.0.0`)  
**Target Runtime:** Python 3.11+, NVIDIA CUDA on RTX GPU / CPU Fallback, ONNX Runtime  

---

## 1. Purpose

The VisionQueue Computer Vision subsystem is a production-grade, real-time edge computer vision engine designed for automated crowd monitoring and queue occupancy management.

It ingests live video streams, detects persons via deep learning inference, tracks individuals across frames with temporal stability, computes occupancy headcounts, evaluates crowd densities, triggers operational alerts, and exports a clean, JSON-serializable, unified **`LiveState`** snapshot per frame.

Downstream consumers (FastAPI service layers, WebSocket broadcasters, database persistence services, and frontend dashboards) interact with the CV engine exclusively through its documented public interface without requiring knowledge of YOLO, ByteTrack, Kalman filtering, or OpenCV.

---

## 2. End-to-End Architecture

```
                       Camera Source (RTSP / USB / File)
                                      ↓
                     YOLO26m ONNX Runtime (CUDA EP)
                                      ↓
                             Detection Contract
                      [bbox: (x1,y1,x2,y2), conf, class_id]
                                      ↓
                          DetectionTrackingAdapter
                                      ↓
                       Custom ByteTrack MOT Engine
                                      ↓
                               Track Contract
                      [track_id: int, bbox, confidence]
                                      ↓
                    Whole-Frame Headcount Counting (V1)
                         (Optional ROI Sub-region)
                                      ↓
                       Virtual Line Crossing (In / Out)
                                      ↓
                    Session Approximate Unique Counting
                                      ↓
                       Crowd Density Analytics & Trend
                      (LOW / MODERATE / HIGH / CRITICAL)
                                      ↓
                       P0 Operational Alert Engine
                      (Critical Occupancy / Failures)
                                      ↓
                       System Reliability & Watchdog
                   (STARTING / LIVE / DEGRADED / OFFLINE)
                                      ↓
                             Unified LiveState
```

---

## 3. Public API Contract

Downstream code interacts **only** with the `visionqueue.pipeline` module:

```python
from visionqueue.pipeline import CVPipeline, CVPipelineConfig, LiveState
from visionqueue.camera.types import CameraConfig
from visionqueue.analytics.types import AnalyticsConfig, CrowdThresholds
from visionqueue.counting.types import VirtualLine
```

### Core Operations

| Method / Property | Description | Thread-Safety |
| :--- | :--- | :--- |
| `pipeline = CVPipeline(config)` | Instantiate pipeline with configuration | Main / Worker Thread |
| `pipeline.start()` | Start background camera acquisition and warm up models | Worker Thread |
| `state = pipeline.step(timeout=1.0)` | Synchronously process the newest available frame | Worker Thread |
| `pipeline.process_frame(frame_data)` | Process an explicit `FrameData` or `None` (for custom loops) | Worker Thread |
| `pipeline.stop()` | Gracefully stop capture and mark system state `STOPPING` | Any Thread |
| `pipeline.reset_session()` | Clear all active tracks, cumulative counts, alerts, and peaks | Any Thread |
| `pipeline.is_running` | Boolean flag indicating whether pipeline is actively running | Thread-safe read |
| `pipeline.last_diagnostics` | In-depth diagnostic telemetry (track ages, IoUs, terminations) | Thread-safe read |
| `with CVPipeline(config) as p:` | Context manager for automated start/stop lifecycle | Worker Thread |

---

## 4. Authoritative LiveState Schema (v1.0)

Every frame processed by `pipeline.step()` returns an immutable `LiveState` instance. Calling `state.to_dict()` produces a 100% JSON-serializable dictionary matching this exact schema:

```json
{
  "schema_version": 1,
  "timestamp": 1740307200.123,
  "session_id": "8f3b2a14-4c8d-4e9b-9a1f-3e5d7c9a1b2c",
  "frame_id": 1420,

  "system_state": "LIVE",
  "camera_state": "CONNECTED",
  "is_healthy": true,
  "is_frozen": false,
  "status_reason": "System operating nominally",

  "counts": {
    "current": 6,
    "track_instances": 24,
    "unique_session_approx": 24,
    "entries": 18,
    "exits": 12,
    "net_count": 6
  },

  "occupancy": {
    "capacity": 20,
    "capacity_state": "SET",
    "percent": 30
  },

  "crowd": {
    "level": "LOW",
    "raw_level": "LOW",
    "trend": "STABLE",
    "peak_count": 9,
    "peak_occupancy_percent": 45,
    "peak_timestamp": 1740306800.456
  },

  "performance": {
    "processing_fps": 28.5,
    "inference_latency_ms": 7.42,
    "frame_age_ms": 11.20
  },

  "vision": {
    "person_detection": "OK",
    "face_detection": "OK",
    "tracking": "OK"
  },

  "alerts": [
    {
      "id": "alert-occ-9f2b",
      "type": "CRITICAL_OCCUPANCY",
      "severity": "CRITICAL",
      "fired_at": 1740307190.0,
      "cleared_at": null,
      "status": "ACTIVE",
      "reason": "Occupancy reached 95% (capacity: 20)"
    }
  ],

  "detections_count": 6,
  "tracks_count": 6,
  "faces_count": 0
}
```

### Field Definitions

* `system_state`: String enum (`STARTING`, `LIVE`, `DEGRADED`, `UNSTABLE`, `OFFLINE`, `STOPPING`).
* `camera_state`: String enum (`CONNECTED`, `RECONNECTING`, `OFFLINE`).
* `is_healthy`: `true` if and only if `system_state == "LIVE"`.
* `is_frozen`: `true` if headcount is frozen to the last reliable value during stream/detector degradation.
* `counts.current`: Active instantaneous person count (whole frame in V1).
* `counts.track_instances`: Total distinct track IDs observed during the session (each distinct ByteTrack ID counts once). This is an internal diagnostic count, NOT a unique human count. See also `unique_session_approx` (deprecated alias).
* `counts.unique_session_approx`: **Deprecated alias for `track_instances`.** Will be removed in a future schema-breaking release. Use `track_instances` instead. This count reflects distinct tracking IDs, not distinct persons.
* `counts.entries` / `counts.exits`: Cumulative counts across virtual entry/exit line.
* `occupancy.percent`: Integer percentage `(current / capacity * 100)` or `null` if capacity is `NOT_SET`.
* `crowd.level`: Debounced crowd density category (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`).
* `crowd.trend`: Moving trend (`INCREASING`, `STABLE`, `DECREASING`).
* `performance.processing_fps`: Measured pipeline throughput.
* `performance.inference_latency_ms`: YOLO26m neural network execution time on CUDA.

---

## 5. Backend & API Service Integration

The backend service layer (e.g. FastAPI) should run the CV pipeline in a **dedicated background worker thread or process**.

### Recommended Integration Pattern

```python
import asyncio
import threading
import time
from visionqueue.pipeline import CVPipeline, CVPipelineConfig
from visionqueue.camera.types import CameraConfig
from visionqueue.analytics.types import AnalyticsConfig

class CVService:
    def __init__(self, camera_source: int | str = 0, capacity: int = 25):
        self.config = CVPipelineConfig(
            camera=CameraConfig(source=camera_source, buffer_size=1),
            analytics=AnalyticsConfig(capacity=capacity),
            enable_roi=False,  # V1 Whole-Frame Default
        )
        self.pipeline = CVPipeline(config=self.config)
        self._thread: threading.Thread | None = None
        self._running = False
        self.latest_state_dict: dict = {}

    def start(self):
        self._running = True
        self.pipeline.start()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self):
        while self._running:
            state = self.pipeline.step(timeout=1.0)
            self.latest_state_dict = state.to_dict()
            time.sleep(0.005)

    def get_latest_state(self) -> dict:
        return self.latest_state_dict

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.pipeline.stop()

    def reset_session(self):
        self.pipeline.reset_session()
```

---

## 6. WebSocket Real-Time Streaming Integration

The backend WebSocket endpoint simply broadcasts `state.to_dict()` directly to subscribed frontend clients.

```python
# In FastAPI / Starlette route:
@app.websocket("/ws/livestate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = cv_service.get_latest_state()
            if payload:
                await websocket.send_json(payload)
            await asyncio.sleep(0.05)  # 20 Hz push rate
    except WebSocketDisconnect:
        pass
```

---

## 7. Database Persistence Guidelines

The Computer Vision engine **does not** write directly to any database. The backend persistence worker should listen to the state stream and persist normalized events:

1. **Periodic Metric Snapshots (e.g. every 5–30 seconds):**
   * Fields: `timestamp`, `session_id`, `current_count`, `occupancy_percent`, `crowd_level`, `processing_fps`.
2. **Line Crossing Events (On event change):**
   * Persist when `counts.entries` or `counts.exits` increments.
   * Fields: `timestamp`, `direction` (ENTRY/EXIT), `session_id`.
3. **Alert Lifecycle Events (On state transition):**
   * Persist when `alerts` contains a new active or cleared alert.
   * Fields: `alert_id`, `type`, `severity`, `fired_at`, `cleared_at`, `status`, `reason`.
4. **Session Summaries (On session termination):**
    * Fields: `session_id`, `start_time`, `end_time`, `track_instances`, `peak_count`, `peak_occupancy_percent`, `total_entries`, `total_exits`.

> [!IMPORTANT]
> **Privacy Compliance Rule:**  
> Do **NOT** persist raw facial images, biometric feature vectors, or visual Re-ID templates. Tracking IDs (`track_id`) are ephemeral session integers and must not be treated as permanent individual identities.

---

## 8. Frontend Dashboard Field Mapping

| UI Element / Widget | Source LiveState Field | Type | Expected Values |
| :--- | :--- | :--- | :--- |
| **Occupancy Gauge** | `occupancy.percent` | `int` or `null` | `0` to `100+` (or `--` if `null`) |
| **Current Headcount** | `counts.current` | `int` | $\ge 0$ |
| **Crowd Density Badge** | `crowd.level` | `str` | `"LOW"`, `"MODERATE"`, `"HIGH"`, `"CRITICAL"` |
| **Crowd Trend Arrow** | `crowd.trend` | `str` | `"INCREASING"` ($\uparrow$), `"STABLE"` ($\leftrightarrow$), `"DECREASING"` ($\downarrow$) |
| **Total In / Entries** | `counts.entries` | `int` | Cumulative count |
| **Total Out / Exits** | `counts.exits` | `int` | Cumulative count |
| **Track Instances** | `counts.track_instances` | `int` | Distinct tracking IDs observed |
| **System Status Indicator** | `system_state` | `str` | `"LIVE"` (Green), `"DEGRADED"` (Yellow), `"OFFLINE"` (Red) |
| **Camera Connectivity** | `camera_state` | `str` | `"CONNECTED"`, `"RECONNECTING"`, `"OFFLINE"` |
| **Live FPS Counter** | `performance.processing_fps` | `float` | e.g. `29.4` |
| **Inference Latency** | `performance.inference_latency_ms` | `float` | e.g. `7.5 ms` |
| **Active Alert Banner** | `alerts` | `list` | Active alert list with severity and reason |

---

## 9. Configuration Management

Configuration is strictly divided into 4 operational tiers:

```
+-------------------------------------------------------------------------+
| IMMUTABLE INTERNAL (Hardcoded in CV Engine)                             |
| - Kalman Filter state matrix dimensions (8x8)                           |
| - Pure IoU Hungarian matching algorithm (Jonker-Volgenant)              |
| - COCO Person Class ID (0)                                              |
+-------------------------------------------------------------------------+
| STARTUP CONFIG (Specified in CVPipelineConfig at instantiation)         |
| - camera.source (Device index / RTSP URL / Video file path)             |
| - detector.model_path (Default: "models/yolo26m_crowd.onnx")             |
|   Rollback baseline: "models/yolo26m.onnx"                              |
|   Env Var Override: VISIONQUEUE_DETECTOR_MODEL_PATH                     |
| - detector.confidence_threshold (Default: 0.25)                         |
| - enable_roi (Default: False for V1 Whole-Frame counting)               |
| - session_id (UUID string)                                              |
+-------------------------------------------------------------------------+
| SAFE RUNTIME CONFIG (Updatable without restarting pipeline)             |
| - analytics.capacity (Set venue capacity > 0 or None)                   |
| - roi (Update ROI bounding box coordinates)                             |
| - alerts debounce timings                                               |
+-------------------------------------------------------------------------+
| RESTART-REQUIRED CONFIG (Requires pipeline.stop() -> recreate -> start) |
| - Changing camera hardware device                                       |
| - Switching between CPU and CUDA execution providers                    |
+-------------------------------------------------------------------------+
```

---

## 10. Lifecycle State Machine

```
   [CREATED]
       │ pipeline.start()
       ▼
  [STARTING] (Awaiting 3 consecutive healthy frames)
       │ Healthy frames confirmed
       ▼
    [LIVE] ◄────────────────────────────────────────┐
       │                                            │ Stream recovered
       ├─► (FPS < 5 or Latency > 200ms) ─► [DEGRADED]
       │                                            │
       ├─► (5+ consecutive det failures) ─► [UNSTABLE]
       │                                            │
       └─► (Camera disconnected) ─────────► [OFFLINE] ───┘ (Goes via STARTING)
       │
       │ pipeline.stop()
       ▼
   [STOPPING] ──► [STOPPED]
```

---

## 11. Error Handling & Invariant Guarantees

1. **Last Reliable Count Invariant**: When camera disconnects or detector fails, `is_frozen = true` and `counts.current` **preserves the last trusted headcount**. The system **never** fabricates a sudden drop to $0$.
2. **Safe Restart Transition**: When a broken camera feed recovers, the system transitions `OFFLINE` $\to$ `STARTING` $\to$ `LIVE` (requiring 3 consecutive healthy frames) to prevent alert/count flapping.
3. **No Uncaught Exceptions**: All internal inference and tracking calls are wrapped with structured failure handlers that transition health states gracefully.

---

## 12. Privacy Guarantees

* **Anonymous Tracking**: Individuals are tracked using anonymous, non-persistent integers (`track_id = 0, 1, 2, ...`).
* **Zero Biometric Data**: No facial recognition models, no appearance embedding databases, no persistent cross-camera identity matching.
* **Transient Presence Detection**: Face detection (YuNet) is strictly an optional count/presence check and does not extract identity features.

---

## 13. Verified Hardware Performance Baseline

Measurements taken on development hardware (**NVIDIA GeForce RTX 5060 Laptop GPU, AMD Ryzen 7, Windows 11, CUDAExecutionProvider, ONNX Runtime 1.29**):

| Subsystem Component | Metric | Measured Value |
| :--- | :--- | :--- |
| **YOLO26m Person Detector (GPU)** | Inference Latency (p50) | **$7.2\text{ ms}$** ($138\text{ FPS}$ raw) |
| **YOLO26m Person Detector (GPU)** | Inference Latency (p95) | **$8.6\text{ ms}$** |
| **ByteTrack MOT Engine (CPU)** | Tracking Latency per frame | **$< 0.2\text{ ms}$** |
| **Full Pipeline Loop (End-to-End)** | Processing Latency (p50) | **$8.4\text{ ms}$** |
| **Effective Frame Rate** | Live Capture & Evaluation | **$30.0\text{ FPS}$** (Camera-capped) |
| **GPU Memory (VRAM)** | Dedicated Allocation | **$\approx 320\text{ MB}$** |
| **System Memory (RAM)** | Working Set | **$\approx 240\text{ MB}$** |

---

## 14. Verification & Test Baseline

The unified VisionQueue system has **436 passing unit, integration, diagnostic, and regression tests** with **0 failures**:

```bash
pytest
============================= 436 passed in 28.57s =============================
```

### Complete Test Suite Breakdown (28 Modules)

* `tests/test_pipeline.py` (19 tests): End-to-end coordinator, state transitions, failover.
* `tests/test_integration_contract.py` (8 tests): JSON round-trip, schema validation, NaN/Inf checks.
* `tests/test_bytetrack.py` (20 tests): Core two-stage association, tracking lifecycle.
* `tests/test_bytetrack_fixes_regression.py` (8 tests): 500-frame synthetic oscillation, Stage-2 recovery.
* `tests/test_bytetrack_diagnosis.py` (9 tests): Confidence threshold interactions.
* `tests/test_tracking_adapter.py` (8 tests): Detection $\to$ Track contract translation.
* `tests/test_detector.py` (35 tests): YOLO26 ONNX preprocessing, inference, clipping, degenerate box rejection, model rollback override.
* `tests/test_counting.py` (36 tests): Line crossing, whole-frame, session counters.
* `tests/test_crossing_robustness.py` (8 tests): Line crossing edge cases and direction validation.
* `tests/test_roi.py` (44 tests): Spatial point containment, resolution invariance.
* `tests/test_reliability.py` (15 tests): 6-state reliability machine, count freezing, error handling.
* `tests/test_analytics.py` (25 tests): Occupancy percentages, crowd tiers, trend.
* `tests/test_alerts.py` (13 tests): P0 alert firing, debouncing, hysteresis clearing.
* `tests/test_camera.py` (9 tests): OpenCV frame acquisition, reconnect logic, error counter reset.
* `tests/test_coordinator.py` (6 tests): Subsystem coordination and event dispatch.
* `tests/test_eval_framework.py` (48 tests): MOT and synthetic dataset evaluation framework.
* `tests/test_evaluator_telemetry.py` (6 tests): Model telemetry, active provider reporting, and rollback.
* `tests/test_long_run_soak.py` (13 tests): Long-running soak tests, memory leak checks.
* `tests/test_matching.py` (5 tests): Hungarian and greedy bipartite matching algorithms.
* `tests/test_persistence.py` (6 tests): SQLite WAL persistence, transactions, and async writer queue drain.
* `tests/test_prepare_mot20.py` (32 tests): MOT20 benchmark preparation and dataset integrity.
* `tests/test_scene_analyzer.py` (15 tests): Scene lighting, blur, and crowd density analysis.
* `tests/test_v11_regression.py` (16 tests): Regression tests for edge cases.
* `tests/test_viewer_ux.py` (7 tests): Viewer UI states and overlay contracts.
* `tests/test_api.py` (7 tests): REST API endpoints and WebSocket live streaming packets.
* `tests/test_api_integration.py` (1 test): Complete API integration cycle.
* `tests/test_api_lifecycle.py` (15 tests): CVService start/stop/reset lifecycle, diagnostics, and graceful termination.

---

## 15. Known System Boundaries & Limitations

1. **Track Instances ≠ Unique Humans**: `counts.track_instances` counts distinct ByteTrack IDs observed during the session. A single person who exits and re-enters (after the track buffer expires) will generate a new track ID and be counted again. This metric is a diagnostic indicator of tracker activity — it does NOT represent unique human visitors.
2. **Camera Occlusion**: Extreme physical occlusion (e.g. a person completely hidden behind a pillar for $>1\text{ second}$) will cause track termination upon buffer expiration.
3. **Physical / Real-CCTV Operational Deployment**: Software integration, synthetic validation, and benchmark dataset validation are 100% complete; physical camera site calibration remains an operational deployment validation item.

---

## 16. Future Extension Hooks (Post-V1)

* **ROI Activation**: To restrict counting to a physical queue corral, set `enable_roi=True` and pass `roi=ROIConfig(x, y, w, h)`.
* **Multi-Camera Expansion**: The `CVPipeline` class is designed to be instantiated per camera stream (`pipeline_cam1`, `pipeline_cam2`).
* **Detector Fine-Tuning**: Custom weights can be exported to ONNX format and loaded via `DetectorConfig(model_path="models/custom.onnx")` or `VISIONQUEUE_DETECTOR_MODEL_PATH`.

---

## 17. Ownership Boundaries

```
CV SUBSYSTEM (Ram Samhith Owned - DO NOT MODIFY DIRECTLY):
├── visionqueue/camera/
├── visionqueue/detection/
├── visionqueue/tracking/
├── visionqueue/roi/
├── visionqueue/counting/
├── visionqueue/analytics/
├── visionqueue/alerts/
├── visionqueue/reliability/
├── visionqueue/pipeline/
├── models/
└── tests/

BACKEND & INTEGRATION (Verified & Hardened):
├── visionqueue/api/
├── visionqueue/persistence/
└── docs/
```

---

## 18. Model & Dependency Manifest

### Deep Learning Models

| Artifact Name | Role | Family | Format | Dimensions | Size | Empirical Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `models/yolo26m_crowd.onnx` | **Default Production Model** | YOLO26m CrowdHuman-Trained | ONNX | $640 \times 640$ | $81.96\text{ MB}$ | Recall: 78.54% (vs 28.87%), Occluded Recall: 67.17% (vs 7.67%), F1: 83.32% (vs 44.09%), MAE: 6.59 (vs 38.34) |
| `models/yolo26m.onnx` | **Rollback Baseline Model** | YOLO26m COCO | ONNX | $640 \times 640$ | $81.96\text{ MB}$ | General-purpose person detection baseline |
| `models/yolo26n.onnx` | Low-Power / Embedded | YOLO26n (Nano) | ONNX | $640 \times 640$ | $9.94\text{ MB}$ | Lightweight edge device testing |
| `models/face_detection_yunet_2023mar.onnx` | Optional Presence | YuNet | ONNX | Dynamic | $232.58\text{ KB}$ | Transient presence detection (non-biometric) |

### Frozen Runtime Dependencies

```
numpy>=1.26.0,<3.0.0
opencv-python>=4.8.0,<6.0.0
onnxruntime-gpu>=1.18.0,<2.0.0
pytest>=8.0.0
fastapi>=0.110.0
pydantic>=2.0.0
starlette>=0.36.0
```

