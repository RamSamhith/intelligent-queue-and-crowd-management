"""VisionQueue Unified CV Pipeline.

Provides:
- CVPipeline: Real-time CV pipeline coordinator
- CVPipelineConfig: Pipeline configuration
- LiveState: Authoritative live state data contract
"""

from visionqueue.pipeline.types import CVPipelineConfig, LiveState
from visionqueue.pipeline.coordinator import CVPipeline

__all__ = [
    "CVPipelineConfig",
    "LiveState",
    "CVPipeline",
]
