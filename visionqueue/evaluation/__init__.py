"""VisionQueue Evaluation Infrastructure.

Reusable evaluation frameworks, staging comparison tools, and soak testing
utilities for offline accuracy validation, model comparison, and long-run
stability profiling.
"""

from visionqueue.evaluation.eval_framework import (
    load_ground_truth,
    compute_occupancy_metrics,
    compute_peak_occupancy_metrics,
    compute_crossing_metrics,
    compute_track_diagnostics,
    frame_to_timestamp,
    timestamp_to_frame,
    run_offline_evaluation,
    run_model_comparison,
)

from visionqueue.evaluation.staging_comparison import (
    get_cpu_ram_mb,
    get_gpu_vram_mb,
    load_mot20_gt_trajectories,
    compute_box_iou,
    evaluate_pipeline_model,
    generate_comparison_report,
)

from visionqueue.evaluation.soak import (
    get_process_memory_mb,
    run_soak_test,
    analyze_soak_results,
)

from visionqueue.evaluation.mot_converter import (
    parse_seqinfo,
    compile_video,
    parse_gt_to_json,
    GroundTruthAnnotation,
)

__all__ = [
    # eval_framework
    "load_ground_truth",
    "compute_occupancy_metrics",
    "compute_peak_occupancy_metrics",
    "compute_crossing_metrics",
    "compute_track_diagnostics",
    "frame_to_timestamp",
    "timestamp_to_frame",
    "run_offline_evaluation",
    "run_model_comparison",
    # staging_comparison
    "get_cpu_ram_mb",
    "get_gpu_vram_mb",
    "load_mot20_gt_trajectories",
    "compute_box_iou",
    "evaluate_pipeline_model",
    "generate_comparison_report",
    # soak
    "get_process_memory_mb",
    "run_soak_test",
    "analyze_soak_results",
    # mot_converter
    "parse_seqinfo",
    "compile_video",
    "parse_gt_to_json",
    "GroundTruthAnnotation",
]