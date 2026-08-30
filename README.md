# VisionQueue — Intelligent Queue & Crowd Management System

**Computer Vision Subsystem (`v1.0.0`)**  
**Engineering Owner:** Ram Samhith — Computer Vision & System Integration

---

## Overview

VisionQueue is a real-time, edge-optimized Computer Vision system for intelligent queue monitoring and crowd analytics. It performs deep learning person detection, temporal multi-object tracking, whole-frame and ROI counting, line-crossing flow measurement, crowd density classification, and automated operational alerting.

---

## Architecture Flow

```
Camera Source ──► YOLO26m (ONNX CUDA) ──► Custom ByteTrack ──► Headcount & Crossing
                                                                        │
LiveState (JSON) ◄── Reliability Watchdog ◄── Alert Engine ◄── Crowd Analytics
```

---

## Installation

**Prerequisites**
- Python 3.11 or 3.12 (3.11 recommended for the currently validated matrix)
- Windows 10/11 with an NVIDIA GPU and an up-to-date NVIDIA driver
  (Driver ≥ 525 recommended; CUDA 12.x runtime is supplied by the torch wheel
  — **the full CUDA Toolkit is not required**)

**Step 1 — Create a virtual environment and install PyTorch (CUDA 12.8 build)**

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip

# torch MUST come from the PyTorch CUDA wheel index.
# PyPI's default torch is CPU-only and will not enable CUDAExecutionProvider.
.venv\Scripts\python -m pip install torch==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

**Step 2 — Install the remaining VisionQueue dependencies**

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

**Step 3 — Verify the CUDA environment**

```bash
.venv\Scripts\python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected output:
```
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

If `CUDAExecutionProvider` is missing from the list, the torch CUDA wheel
did not install correctly — re-run Step 1 from the PyTorch CUDA index.

---

## 5-Minute Developer Quickstart

```python
import time
from visionqueue.pipeline import CVPipeline, CVPipelineConfig
from visionqueue.camera.types import CameraConfig
from visionqueue.analytics.types import AnalyticsConfig

# 1. Configure the pipeline (V1 Default: Whole-Frame Counting)
config = CVPipelineConfig(
    camera=CameraConfig(source=0),                # 0 for webcam, or "rtsp://..."
    analytics=AnalyticsConfig(capacity=20),       # Maximum venue capacity
    enable_roi=False,                             # Whole frame is active area
)

# 2. Initialize and run
pipeline = CVPipeline(config=config)
pipeline.start()

try:
    while True:
        # Step executes acquisition, inference, tracking, counting & analytics
        state = pipeline.step(timeout=1.0)
        
        # 100% JSON-serializable LiveState dictionary
        payload = state.to_dict()
        
        print(f"[{payload['system_state']}] People: {payload['counts']['current']} | "
              f"Occupancy: {payload['occupancy']['percent']}% | "
              f"Crowd: {payload['crowd']['level']} | "
              f"FPS: {payload['performance']['processing_fps']:.1f}")
        
        time.sleep(0.01)
finally:
    pipeline.stop()
```

---

## Verification & Testing

To run the complete test suite (265 tests):

```bash
pytest -v
```

---

## Documentation

For the complete public integration contract, schema specification, WebSocket guidelines, database persistence rules, and model manifests, refer to:

👉 **[docs/CV_HANDOFF.md](docs/CV_HANDOFF.md)**
