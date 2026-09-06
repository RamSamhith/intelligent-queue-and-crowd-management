"""
Generic, dataset-agnostic ground truth conversion and validation tool for the
VisionQueue Offline Evaluation Framework.

Features:
1. Validates and parses sequence metadata (seqinfo.ini).
2. Compiles image sequences into MP4 video with strict frame-by-frame validation.
3. Converts MOT-format ground truth (gt.txt) into JSON conforming to
   tests/fixtures/ground_truth_schema.json.
4. Dataset-agnostic class resolution:
   - Preserves documented standards (MOT20, MOT17, MOT16, DanceTrack).
   - Resolves unambiguous single-class datasets automatically.
   - Strictly fails loudly on ambiguous multi-class datasets when no explicit policy is provided.
5. Strictly validates and fails on:
   - Missing/corrupt frames.
   - Dimension and frame-count mismatches.
   - Malformed bounding boxes (non-finite, non-positive dimensions, syntax errors).
   - Duplicate (frame, track_id) annotations.
   - Inconsistent frame indices outside sequence boundaries.
   - Invalid metadata.
6. Preserves zero-occupancy frames across the entire sequence.

Usage:
    python scratch/prepare_mot20.py --seq-dir /path/to/sequence --out-dir ./eval_data
    python scratch/prepare_mot20.py --seq-dir /path/to/sequence --out-dir ./eval_data --target-class-ids 1,2
"""

import argparse
import configparser
import datetime
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Documented standard target classes for known tracking benchmark datasets
KNOWN_DATASET_TARGET_CLASSES: Dict[str, Optional[Set[int]]] = {
    "mot20": {1},       # MOT20: class 1 is Pedestrian (active target)
    "mot17": {1},       # MOT17: class 1 is Pedestrian (active target)
    "mot16": {1},       # MOT16: class 1 is Pedestrian (active target)
    "dancetrack": None, # DanceTrack: single-class (all bounding boxes evaluated)
}


class GroundTruthAnnotation:
    """Represents a single parsed ground truth annotation line."""
    __slots__ = ("frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "cls_id", "vis", "line_idx")

    def __init__(
        self,
        frame: int,
        track_id: int,
        bb_left: float,
        bb_top: float,
        bb_width: float,
        bb_height: float,
        conf: float,
        cls_id: Optional[int],
        vis: float,
        line_idx: int,
    ):
        self.frame = frame
        self.track_id = track_id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.cls_id = cls_id
        self.vis = vis
        self.line_idx = line_idx


def parse_seqinfo(seq_dir: Union[str, Path]) -> Dict[str, Any]:
    """Parse seqinfo.ini with strict metadata validation.

    Args:
        seq_dir: Path to the sequence directory containing seqinfo.ini.

    Returns:
        Dictionary containing verified sequence metadata.

    Raises:
        FileNotFoundError: If seqinfo.ini or the image directory does not exist.
        ValueError: If metadata fields are missing, malformed, or invalid.
    """
    seq_dir = Path(seq_dir)
    ini_path = seq_dir / "seqinfo.ini"
    if not ini_path.exists():
        raise FileNotFoundError(f"seqinfo.ini not found in {seq_dir}")

    config = configparser.ConfigParser()
    try:
        config.read(str(ini_path), encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Failed to parse seqinfo.ini in {seq_dir}: {e}")

    if "Sequence" not in config:
        raise ValueError("Invalid seqinfo.ini format: missing [Sequence] section")

    seq = config["Sequence"]

    name = seq.get("name", seq_dir.name).strip()
    if not name:
        name = seq_dir.name

    im_dir = seq.get("imDir", "img1").strip()
    if not im_dir:
        raise ValueError("Invalid metadata: imDir cannot be empty")

    try:
        frame_rate = float(seq.get("frameRate", "25.0"))
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError(f"frameRate must be positive and finite, got {frame_rate}")
    except ValueError as e:
        raise ValueError(f"Invalid frameRate in seqinfo.ini: {e}")

    try:
        seq_length = int(seq.get("seqLength", "0"))
        if seq_length <= 0:
            raise ValueError(f"seqLength must be a positive integer, got {seq_length}")
    except ValueError as e:
        raise ValueError(f"Invalid seqLength in seqinfo.ini: {e}")

    try:
        im_width = int(seq.get("imWidth", "1920"))
        if im_width <= 0:
            raise ValueError(f"imWidth must be a positive integer, got {im_width}")
    except ValueError as e:
        raise ValueError(f"Invalid imWidth in seqinfo.ini: {e}")

    try:
        im_height = int(seq.get("imHeight", "1080"))
        if im_height <= 0:
            raise ValueError(f"imHeight must be a positive integer, got {im_height}")
    except ValueError as e:
        raise ValueError(f"Invalid imHeight in seqinfo.ini: {e}")

    im_ext = seq.get("imExt", ".jpg").strip()
    if not im_ext:
        im_ext = ".jpg"
    if not im_ext.startswith("."):
        im_ext = f".{im_ext}"

    dataset = seq.get("dataset", "").strip().lower()

    return {
        "name": name,
        "imDir": im_dir,
        "frameRate": frame_rate,
        "seqLength": seq_length,
        "imWidth": im_width,
        "imHeight": im_height,
        "imExt": im_ext,
        "dataset": dataset,
    }


def compile_video(seq_dir: Union[str, Path], meta: Dict[str, Any], out_mp4: Union[str, Path]) -> None:
    """Compile image sequence into an MP4 video with strict verification.

    Args:
        seq_dir: Path to sequence root directory.
        meta: Sequence metadata dictionary from parse_seqinfo.
        out_mp4: Destination path for the generated MP4 video.

    Raises:
        FileNotFoundError: If image directory or any frame image is missing.
        ValueError: If an image is corrupt, unreadable, or dimension-mismatched.
        RuntimeError: If VideoWriter cannot be initialized or frame count mismatches.
    """
    seq_dir = Path(seq_dir)
    out_mp4 = Path(out_mp4)
    logger.info(f"Compiling video from image sequence to {out_mp4} ...")

    img_dir = seq_dir / meta["imDir"]
    if not img_dir.exists() or not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    ext = meta["imExt"]
    fps = meta["frameRate"]
    width = meta["imWidth"]
    height = meta["imHeight"]
    total = meta["seqLength"]

    if total <= 0:
        raise ValueError(f"Invalid sequence length: {total}")

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_mp4), fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to open cv2.VideoWriter for {out_mp4}")

    frames_written = 0

    try:
        for i in range(1, total + 1):
            img_path = img_dir / f"{i:06d}{ext}"
            if not img_path.exists():
                raise FileNotFoundError(f"Missing expected image file: {img_path}")

            frame = cv2.imread(str(img_path))
            if frame is None or frame.size == 0:
                raise ValueError(f"Failed to read image or corrupt file: {img_path}")

            h, w = frame.shape[:2]
            if w != width or h != height:
                raise ValueError(
                    f"Frame {i} dimension mismatch: actual ({w}x{h}), expected ({width}x{height})"
                )

            out.write(frame)
            frames_written += 1

            if frames_written % 500 == 0:
                logger.info(f"Processed {frames_written}/{total} frames...")

        if frames_written != total:
            raise RuntimeError(
                f"Frame count mismatch: wrote {frames_written} frames, expected {total}"
            )

    finally:
        out.release()

    logger.info(f"Video saved to {out_mp4} ({frames_written} frames).")


def _detect_dataset_standard(meta: Dict[str, Any], explicit_type: Optional[str] = None) -> Optional[str]:
    """Detect dataset standard from explicit type, metadata dataset key, or sequence name."""
    if explicit_type:
        return explicit_type.strip().lower()

    if meta.get("dataset"):
        ds = meta["dataset"].strip().lower()
        if ds in KNOWN_DATASET_TARGET_CLASSES:
            return ds

    name_lower = meta.get("name", "").strip().lower()
    for known in ("mot20", "mot17", "mot16", "dancetrack"):
        if name_lower.startswith(known):
            return known

    return None


def parse_gt_to_json(
    seq_dir: Union[str, Path],
    meta: Dict[str, Any],
    out_json: Union[str, Path],
    video_filename: str,
    dataset_type: Optional[str] = None,
    target_class_ids: Optional[Union[Set[int], List[int], int]] = None,
    min_conf: Optional[float] = None,
) -> Dict[str, Any]:
    """Parse MOT-format ground truth and convert to standard evaluation JSON schema.

    Enforces strict data integrity:
    - Bounding boxes must have positive finite dimensions.
    - Frame indices must be 1-indexed and within [1, seqLength].
    - Duplicate (frame, track_id) annotations are strictly rejected.
    - Class semantics are resolved automatically when unambiguous; multi-class datasets
      with unknown semantics fail loudly unless explicit target classes are provided.
    - Zero-occupancy frames are preserved across the entire sequence.

    Args:
        seq_dir: Path to sequence directory containing gt/gt.txt.
        meta: Sequence metadata from parse_seqinfo.
        out_json: Destination path for ground truth JSON.
        video_filename: Base filename of the corresponding video.
        dataset_type: Optional explicit dataset standard (e.g. 'mot20', 'mot17', 'dancetrack').
        target_class_ids: Optional explicit set/list/int of target class IDs to evaluate.
        min_conf: Optional minimum confidence threshold.

    Returns:
        Ground truth dictionary conforming to tests/fixtures/ground_truth_schema.json.

    Raises:
        FileNotFoundError: If gt.txt does not exist.
        ValueError: If GT is malformed, duplicate tracks occur, frame range is violated,
                    or class semantics are ambiguous and unspecified.
    """
    seq_dir = Path(seq_dir)
    out_json = Path(out_json)
    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"gt.txt not found at {gt_path}")

    logger.info(f"Parsing ground truth from {gt_path} ...")

    seq_length = meta["seqLength"]
    detected_standard = _detect_dataset_standard(meta, dataset_type)

    # 1. Parse and strictly validate every row in gt.txt
    annotations: List[GroundTruthAnnotation] = []
    discovered_class_ids: Set[int] = set()
    has_class_column = False
    has_conf_column = False

    with open(gt_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue

            parts = [p.strip() for p in line_str.split(",")]
            if len(parts) < 6:
                raise ValueError(
                    f"Malformed ground truth row on line {line_idx} in {gt_path}: "
                    f"expected at least 6 comma-separated fields, got {len(parts)} ({line_str})"
                )

            # Parse frame
            try:
                frame_num = int(parts[0])
            except ValueError:
                raise ValueError(f"Malformed frame index on line {line_idx} in {gt_path}: {parts[0]}")

            if frame_num < 1 or frame_num > seq_length:
                raise ValueError(
                    f"Inconsistent GT: frame {frame_num} on line {line_idx} is outside "
                    f"valid sequence range [1, {seq_length}]"
                )

            # Parse track ID
            try:
                track_id = int(parts[1])
            except ValueError:
                raise ValueError(f"Malformed track ID on line {line_idx} in {gt_path}: {parts[1]}")

            if track_id < 1:
                raise ValueError(
                    f"Inconsistent GT: invalid track ID {track_id} on line {line_idx}"
                )

            # Parse bounding box coordinates
            try:
                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_width = float(parts[4])
                bb_height = float(parts[5])
            except ValueError:
                raise ValueError(
                    f"Malformed bounding box coordinates on line {line_idx} in {gt_path}: {line_str}"
                )

            if not (
                math.isfinite(bb_left)
                and math.isfinite(bb_top)
                and math.isfinite(bb_width)
                and math.isfinite(bb_height)
            ):
                raise ValueError(
                    f"Malformed box on line {line_idx} in {gt_path}: non-finite coordinate values "
                    f"({bb_left}, {bb_top}, {bb_width}, {bb_height})"
                )

            if bb_width <= 0 or bb_height <= 0:
                raise ValueError(
                    f"Malformed box on line {line_idx} in {gt_path}: width and height must be positive, "
                    f"got width={bb_width}, height={bb_height}"
                )

            # Parse optional confidence
            conf = 1.0
            if len(parts) >= 7:
                has_conf_column = True
                try:
                    conf = float(parts[6])
                except ValueError:
                    raise ValueError(f"Malformed confidence on line {line_idx} in {gt_path}: {parts[6]}")
                if not math.isfinite(conf):
                    raise ValueError(
                        f"Malformed confidence on line {line_idx} in {gt_path}: non-finite value {conf}"
                    )

            # Parse optional class ID
            cls_id: Optional[int] = None
            if len(parts) >= 8:
                has_class_column = True
                try:
                    cls_id = int(parts[7])
                    discovered_class_ids.add(cls_id)
                except ValueError:
                    raise ValueError(f"Malformed class ID on line {line_idx} in {gt_path}: {parts[7]}")

            # Parse optional visibility
            vis = 1.0
            if len(parts) >= 9:
                try:
                    vis = float(parts[8])
                except ValueError:
                    raise ValueError(f"Malformed visibility on line {line_idx} in {gt_path}: {parts[8]}")
                if not math.isfinite(vis):
                    raise ValueError(
                        f"Malformed visibility on line {line_idx} in {gt_path}: non-finite value {vis}"
                    )

            annotations.append(
                GroundTruthAnnotation(
                    frame=frame_num,
                    track_id=track_id,
                    bb_left=bb_left,
                    bb_top=bb_top,
                    bb_width=bb_width,
                    bb_height=bb_height,
                    conf=conf,
                    cls_id=cls_id,
                    vis=vis,
                    line_idx=line_idx,
                )
            )

    # 2. Resolve class semantics and confidence policy
    effective_target_classes: Optional[Set[int]] = None
    is_mot_challenge_standard = False

    if target_class_ids is not None:
        if isinstance(target_class_ids, int):
            effective_target_classes = {target_class_ids}
        else:
            effective_target_classes = set(target_class_ids)
    elif detected_standard in KNOWN_DATASET_TARGET_CLASSES:
        effective_target_classes = KNOWN_DATASET_TARGET_CLASSES[detected_standard]
        if detected_standard in ("mot20", "mot17", "mot16"):
            is_mot_challenge_standard = True
    elif not has_class_column or len(discovered_class_ids) == 0:
        # 6-column or 7-column format: no class column exists, single class is unambiguous
        effective_target_classes = None
    elif len(discovered_class_ids) == 1:
        # Single unambiguous class ID across the entire file
        effective_target_classes = discovered_class_ids
    else:
        # Multiple class IDs found in an unknown/unspecified dataset format -> FAIL LOUDLY
        sorted_classes = sorted(discovered_class_ids)
        raise ValueError(
            f"Ambiguous class semantics: found multiple class IDs {sorted_classes} in {gt_path} "
            f"without an explicit target class policy or known dataset specification. "
            f"Specify target class IDs via --target-class-ids or dataset standard via --dataset-type."
        )

    # 3. Filter active target annotations and compute per-frame occupancy
    occupancy_counts: Dict[int, int] = defaultdict(int)
    seen_tracks: Set[Tuple[int, int]] = set()

    for ann in annotations:
        # Check class match
        if effective_target_classes is not None and ann.cls_id is not None:
            if ann.cls_id not in effective_target_classes:
                continue

        # Check confidence filter
        if min_conf is not None:
            if ann.conf < min_conf:
                continue
        elif is_mot_challenge_standard:
            # MOT challenge standard evaluation considers conf == 1.0 (or 1) as active evaluation targets
            if ann.conf != 1.0:
                continue
        elif has_conf_column:
            # For generic datasets with conf column without explicit threshold, require conf > 0
            if ann.conf <= 0.0:
                continue

        frame_0idx = ann.frame - 1  # 0-indexed for schema representation
        track_pair = (frame_0idx, ann.track_id)
        if track_pair in seen_tracks:
            raise ValueError(
                f"Duplicate track annotation found for frame {ann.frame}, ID {ann.track_id} on line {ann.line_idx}"
            )
        seen_tracks.add(track_pair)
        occupancy_counts[frame_0idx] += 1

    peak_occupancy = 0
    peak_occupancy_frame = 0
    if occupancy_counts:
        peak_occupancy = max(occupancy_counts.values())
        peak_occupancy_frame = min(f for f, c in occupancy_counts.items() if c == peak_occupancy)

    # 4. Build standard schema ground truth JSON
    annotated_by = "MOTChallenge" if is_mot_challenge_standard else "GroundTruth"
    scenario_type = "MIXED"
    protocol_desc = (
        "MOT20 standard identity. Evaluator uses per-frame counts."
        if is_mot_challenge_standard
        else "Standard dataset identity. Evaluator uses per-frame counts."
    )

    gt_data: Dict[str, Any] = {
        "schema_version": 1,
        "metadata": {
            "video_filename": video_filename,
            "video_id": meta["name"],
            "source_description": f"Dataset sequence {meta['name']}",
            "fps": meta["frameRate"],
            "frame_width": meta["imWidth"],
            "frame_height": meta["imHeight"],
            "total_frames": meta["seqLength"],
            "duration_seconds": meta["seqLength"] / meta["frameRate"] if meta["frameRate"] > 0 else 0.0,
            "scenario_type": scenario_type,
            "camera_mount": "UNKNOWN",
            "annotated_by": annotated_by,
            "annotation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "notes": "Generated from gt.txt via automated script.",
        },
        "annotation_protocol": {
            "annotation_method": "MANUAL_COUNT",
            "track_identity_protocol": protocol_desc,
            "annotation_sampling_rate": "every_frame",
            "crossing_tolerance_seconds": 1.0,
        },
        "occupancy": [],
        "crossings": [],
        "trajectories": [],
        "metrics": {
            "total_entries": 0,
            "total_exits": 0,
            "peak_occupancy": peak_occupancy,
            "peak_occupancy_frame": peak_occupancy_frame,
            "annotation_id_count": len({tid for _, tid in seen_tracks}),
        },
        "critical_events": [],
    }

    # Populate occupancy preserving zero-occupancy frames across all frames
    total_frames = meta["seqLength"]
    for frame_idx in range(total_frames):
        gt_data["occupancy"].append(
            {
                "frame": frame_idx,
                "timestamp_seconds": frame_idx / meta["frameRate"] if meta["frameRate"] > 0 else 0.0,
                "person_count": occupancy_counts.get(frame_idx, 0),
                "confidence": "HIGH",
            }
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2)

    logger.info(
        f"Ground truth JSON saved to {out_json} (Peak occupancy: {peak_occupancy} at frame {peak_occupancy_frame})."
    )
    return gt_data


def main() -> None:
    """CLI entry point for dataset preparation and validation."""
    parser = argparse.ArgumentParser(
        description="Prepare video and ground truth annotations for Offline Evaluation."
    )
    parser.add_argument(
        "--seq-dir",
        required=True,
        help="Path to sequence directory containing seqinfo.ini, img1/, and gt/gt.txt",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to save the resulting .mp4 video and .json ground truth",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["mot20", "mot17", "mot16", "dancetrack", "generic"],
        default=None,
        help="Dataset standard profile to apply for class and confidence rules",
    )
    parser.add_argument(
        "--target-class-ids",
        default=None,
        help="Comma-separated target class IDs (e.g. '1' or '1,2')",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=None,
        help="Minimum confidence threshold for ground truth annotations",
    )
    parser.add_argument(
        "--video-only",
        action="store_true",
        help="Only compile the video from images",
    )
    parser.add_argument(
        "--gt-only",
        action="store_true",
        help="Only parse and generate the ground truth JSON",
    )

    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        meta = parse_seqinfo(seq_dir)
    except Exception as e:
        logger.error(f"Failed to parse seqinfo.ini: {e}")
        sys.exit(1)

    seq_name = meta["name"]
    out_mp4 = out_dir / f"{seq_name}.mp4"
    out_json = out_dir / f"{seq_name}_gt.json"

    logger.info(
        f"Processing sequence {seq_name} ({meta['imWidth']}x{meta['imHeight']} @ {meta['frameRate']} FPS)"
    )

    target_class_ids = None
    if args.target_class_ids:
        try:
            target_class_ids = {int(x.strip()) for x in args.target_class_ids.split(",") if x.strip()}
        except ValueError:
            logger.error(f"Invalid --target-class-ids argument: {args.target_class_ids}")
            sys.exit(1)

    try:
        if not args.gt_only:
            compile_video(seq_dir, meta, out_mp4)
        if not args.video_only:
            parse_gt_to_json(
                seq_dir,
                meta,
                out_json,
                video_filename=out_mp4.name,
                dataset_type=args.dataset_type,
                target_class_ids=target_class_ids,
                min_conf=args.min_conf,
            )
        logger.info(f"Successfully prepared sequence {seq_name}!")
    except Exception as e:
        logger.error(f"Error during preparation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
