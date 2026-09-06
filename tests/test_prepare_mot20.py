import json
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import cv2
import pytest

from visionqueue.evaluation.mot_converter import (
    parse_seqinfo,
    compile_video,
    parse_gt_to_json,
    main,
)


@pytest.fixture
def synthetic_mot20_dir():
    """Creates a temporary directory with synthetic MOT20-02 data."""
    temp_dir = tempfile.mkdtemp()
    seq_dir = Path(temp_dir) / "MOT20-02"
    seq_dir.mkdir(parents=True)

    # Create seqinfo.ini
    ini_content = """[Sequence]
name=MOT20-02
imDir=img1
frameRate=25
seqLength=3
imWidth=640
imHeight=480
imExt=.jpg
"""
    (seq_dir / "seqinfo.ini").write_text(ini_content)

    # Create img1 with 3 black frames
    img_dir = seq_dir / "img1"
    img_dir.mkdir()

    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(1, 4):
        cv2.imwrite(str(img_dir / f"{i:06d}.jpg"), black_frame)

    # Create gt/gt.txt with standard MOT20 annotations
    # Row format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir()

    gt_content = """1,1,10,10,50,100,1,1,1.0
1,2,100,10,50,100,1,1,1.0
1,3,200,10,50,100,0,1,1.0
2,1,10,10,50,100,1,1,0.5
2,4,300,10,50,100,1,8,1.0
3,1,10,10,50,100,1,1,0.0
"""
    (gt_dir / "gt.txt").write_text(gt_content)

    yield seq_dir

    shutil.rmtree(temp_dir)


def validate_schema(gt_data):
    """Schema validation matching tests/fixtures/ground_truth_schema.json."""
    assert gt_data["schema_version"] == 1
    assert "metadata" in gt_data
    assert "annotation_protocol" in gt_data
    assert "occupancy" in gt_data
    assert "metrics" in gt_data

    meta = gt_data["metadata"]
    assert "video_filename" in meta
    assert "fps" in meta and meta["fps"] > 0
    assert "frame_width" in meta and meta["frame_width"] > 0
    assert "frame_height" in meta and meta["frame_height"] > 0
    assert "total_frames" in meta and meta["total_frames"] > 0

    proto = gt_data["annotation_protocol"]
    assert "track_identity_protocol" in proto
    assert "crossing_tolerance_seconds" in proto

    metrics = gt_data["metrics"]
    assert "total_entries" in metrics
    assert "total_exits" in metrics
    assert "peak_occupancy" in metrics
    assert "peak_occupancy_frame" in metrics

    for occ in gt_data["occupancy"]:
        assert "frame" in occ
        assert "person_count" in occ
        assert occ["person_count"] >= 0
        assert "timestamp_seconds" in occ
        assert "confidence" in occ


# =============================================================================
# 1. MOT20-02 Behavior Preservation Tests
# =============================================================================

def test_parse_seqinfo(synthetic_mot20_dir):
    meta = parse_seqinfo(synthetic_mot20_dir)
    assert meta["name"] == "MOT20-02"
    assert meta["frameRate"] == 25.0
    assert meta["seqLength"] == 3
    assert meta["imWidth"] == 640
    assert meta["imHeight"] == 480
    assert meta["imDir"] == "img1"


def test_gt_parsing(synthetic_mot20_dir):
    meta = parse_seqinfo(synthetic_mot20_dir)
    out_json = synthetic_mot20_dir / "out_gt.json"
    parse_gt_to_json(synthetic_mot20_dir, meta, out_json, "video.mp4")

    with open(out_json) as f:
        gt = json.load(f)

    validate_schema(gt)

    assert gt["metrics"]["peak_occupancy"] == 2
    assert gt["metrics"]["peak_occupancy_frame"] == 0
    assert gt["metrics"]["annotation_id_count"] == 2  # IDs 1 and 2 (active pedestrians)

    occupancy = gt["occupancy"]
    assert len(occupancy) == 3

    assert occupancy[0]["frame"] == 0
    assert occupancy[0]["person_count"] == 2

    assert occupancy[1]["frame"] == 1
    assert occupancy[1]["person_count"] == 1

    assert occupancy[2]["frame"] == 2
    assert occupancy[2]["person_count"] == 1


def test_compile_video(synthetic_mot20_dir):
    meta = parse_seqinfo(synthetic_mot20_dir)
    out_mp4 = synthetic_mot20_dir / "out.mp4"
    compile_video(synthetic_mot20_dir, meta, out_mp4)

    assert out_mp4.exists()
    cap = cv2.VideoCapture(str(out_mp4))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    cap.release()


# =============================================================================
# 2. Generic Dataset & Class Semantics Tests
# =============================================================================

def test_generic_6col_single_class(tmp_path):
    """6-column MOT format without confidence or class column (e.g. DanceTrack/MOT15)."""
    seq_dir = tmp_path / "GENERIC-6COL"
    seq_dir.mkdir()
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nname=GENERIC-6COL\nimDir=img1\nframeRate=30\nseqLength=2\nimWidth=320\nimHeight=240\n")
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir()
    (gt_dir / "gt.txt").write_text("1,1,10,10,20,40\n1,2,50,50,20,40\n2,1,15,15,20,40\n")

    meta = parse_seqinfo(seq_dir)
    out_json = seq_dir / "out.json"
    gt = parse_gt_to_json(seq_dir, meta, out_json, "video.mp4")

    validate_schema(gt)
    assert gt["metrics"]["peak_occupancy"] == 2
    assert gt["occupancy"][0]["person_count"] == 2
    assert gt["occupancy"][1]["person_count"] == 1


def test_generic_7col_with_conf(tmp_path):
    """7-column MOT format with confidence values."""
    seq_dir = tmp_path / "GENERIC-7COL"
    seq_dir.mkdir()
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nname=GENERIC-7COL\nimDir=img1\nframeRate=30\nseqLength=2\nimWidth=320\nimHeight=240\n")
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir()
    # Row 1 is conf 1.0, row 2 is conf 0.0 (inactive), row 3 is conf 0.8
    (gt_dir / "gt.txt").write_text("1,1,10,10,20,40,1.0\n1,2,50,50,20,40,0.0\n2,1,15,15,20,40,0.8\n")

    meta = parse_seqinfo(seq_dir)
    out_json = seq_dir / "out.json"
    gt = parse_gt_to_json(seq_dir, meta, out_json, "video.mp4")

    validate_schema(gt)
    assert gt["occupancy"][0]["person_count"] == 1  # ID 2 ignored because conf <= 0
    assert gt["occupancy"][1]["person_count"] == 1  # ID 1 has conf 0.8 > 0


def test_generic_8col_single_unambiguous_class(tmp_path):
    """8-column format where all annotations share the same single class ID."""
    seq_dir = tmp_path / "GENERIC-8COL"
    seq_dir.mkdir()
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nname=GENERIC-8COL\nimDir=img1\nframeRate=30\nseqLength=2\nimWidth=320\nimHeight=240\n")
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir()
    (gt_dir / "gt.txt").write_text("1,1,10,10,20,40,1.0,3\n2,1,15,15,20,40,1.0,3\n")

    meta = parse_seqinfo(seq_dir)
    out_json = seq_dir / "out.json"
    gt = parse_gt_to_json(seq_dir, meta, out_json, "video.mp4")

    validate_schema(gt)
    assert gt["metrics"]["peak_occupancy"] == 1
    assert gt["occupancy"][0]["person_count"] == 1
    assert gt["occupancy"][1]["person_count"] == 1


def test_generic_multiclass_ambiguity_fails_loudly(tmp_path):
    """Multiple class IDs without explicit policy or dataset type must fail loudly."""
    seq_dir = tmp_path / "AMBIGUOUS-SEQ"
    seq_dir.mkdir()
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nname=AMBIGUOUS-SEQ\nimDir=img1\nframeRate=30\nseqLength=2\nimWidth=320\nimHeight=240\n")
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir()
    # Classes 1, 2, and 7 present
    (gt_dir / "gt.txt").write_text("1,1,10,10,20,40,1.0,1\n1,2,50,50,20,40,1.0,2\n2,3,15,15,20,40,1.0,7\n")

    meta = parse_seqinfo(seq_dir)
    out_json = seq_dir / "out.json"

    with pytest.raises(ValueError, match="Ambiguous class semantics: found multiple class IDs"):
        parse_gt_to_json(seq_dir, meta, out_json, "video.mp4")


def test_generic_multiclass_with_explicit_target_class(tmp_path):
    """Explicitly providing target_class_ids resolves multi-class datasets deterministically."""
    seq_dir = tmp_path / "MULTICLASS-SEQ"
    seq_dir.mkdir()
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nname=MULTICLASS-SEQ\nimDir=img1\nframeRate=30\nseqLength=2\nimWidth=320\nimHeight=240\n")
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir()
    # Classes 1, 2, and 3 present
    (gt_dir / "gt.txt").write_text("1,1,10,10,20,40,1.0,1\n1,2,50,50,20,40,1.0,2\n2,3,15,15,20,40,1.0,3\n")

    meta = parse_seqinfo(seq_dir)
    out_json = seq_dir / "out.json"

    # Filter for class 2 only
    gt_cls2 = parse_gt_to_json(seq_dir, meta, out_json, "video.mp4", target_class_ids={2})
    assert gt_cls2["occupancy"][0]["person_count"] == 1
    assert gt_cls2["occupancy"][1]["person_count"] == 0

    # Filter for classes {1, 3}
    gt_cls13 = parse_gt_to_json(seq_dir, meta, out_json, "video.mp4", target_class_ids={1, 3})
    assert gt_cls13["occupancy"][0]["person_count"] == 1
    assert gt_cls13["occupancy"][1]["person_count"] == 1


def test_explicit_dataset_type_override(tmp_path):
    """Passing dataset_type='mot20' applies MOT20 rules (class 1, conf 1.0)."""
    seq_dir = tmp_path / "CUSTOM-NAME"
    seq_dir.mkdir()
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nname=CUSTOM-NAME\nimDir=img1\nframeRate=30\nseqLength=2\nimWidth=320\nimHeight=240\n")
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir()
    # Class 1 (Pedestrian, conf 1.0), Class 8 (Distractor, conf 1.0)
    (gt_dir / "gt.txt").write_text("1,1,10,10,20,40,1.0,1\n1,2,50,50,20,40,1.0,8\n")

    meta = parse_seqinfo(seq_dir)
    out_json = seq_dir / "out.json"
    gt = parse_gt_to_json(seq_dir, meta, out_json, "video.mp4", dataset_type="mot20")

    assert gt["occupancy"][0]["person_count"] == 1


# =============================================================================
# 3. Strict Malformed Data & Inconsistency Checks
# =============================================================================

def test_malformed_gt_fewer_columns(synthetic_mot20_dir):
    with open(synthetic_mot20_dir / "gt" / "gt.txt", "a") as f:
        f.write("invalid_row,1,2\n")

    meta = parse_seqinfo(synthetic_mot20_dir)
    out_json = synthetic_mot20_dir / "out_gt.json"
    with pytest.raises(ValueError, match="Malformed ground truth row"):
        parse_gt_to_json(synthetic_mot20_dir, meta, out_json, "video.mp4")


def test_malformed_gt_negative_bbox_dimensions(synthetic_mot20_dir):
    with open(synthetic_mot20_dir / "gt" / "gt.txt", "a") as f:
        f.write("2,99,10,10,-50,-100,1,1,1.0\n")

    meta = parse_seqinfo(synthetic_mot20_dir)
    out_json = synthetic_mot20_dir / "out_gt.json"
    with pytest.raises(ValueError, match="Malformed box.*width and height must be positive"):
        parse_gt_to_json(synthetic_mot20_dir, meta, out_json, "video.mp4")


def test_malformed_gt_zero_bbox_dimensions(synthetic_mot20_dir):
    with open(synthetic_mot20_dir / "gt" / "gt.txt", "a") as f:
        f.write("2,100,10,10,0,100,1,1,1.0\n")

    meta = parse_seqinfo(synthetic_mot20_dir)
    out_json = synthetic_mot20_dir / "out_gt.json"
    with pytest.raises(ValueError, match="Malformed box.*width and height must be positive"):
        parse_gt_to_json(synthetic_mot20_dir, meta, out_json, "video.mp4")


def test_malformed_gt_non_finite_coords(synthetic_mot20_dir):
    with open(synthetic_mot20_dir / "gt" / "gt.txt", "a") as f:
        f.write("2,101,NaN,10,50,100,1,1,1.0\n")

    meta = parse_seqinfo(synthetic_mot20_dir)
    out_json = synthetic_mot20_dir / "out_gt.json"
    with pytest.raises(ValueError, match="Malformed box.*non-finite coordinate values"):
        parse_gt_to_json(synthetic_mot20_dir, meta, out_json, "video.mp4")


def test_malformed_gt_non_numeric_entry(synthetic_mot20_dir):
    with open(synthetic_mot20_dir / "gt" / "gt.txt", "a") as f:
        f.write("2,abc,10,10,50,100,1,1,1.0\n")

    meta = parse_seqinfo(synthetic_mot20_dir)
    out_json = synthetic_mot20_dir / "out_gt.json"
    with pytest.raises(ValueError, match="Malformed track ID"):
        parse_gt_to_json(synthetic_mot20_dir, meta, out_json, "video.mp4")


def test_inconsistent_gt_frame_out_of_range(synthetic_mot20_dir):
    with open(synthetic_mot20_dir / "gt" / "gt.txt", "a") as f:
        # Frame 10 is outside seqLength=3
        f.write("10,1,10,10,50,100,1,1,1.0\n")

    meta = parse_seqinfo(synthetic_mot20_dir)
    out_json = synthetic_mot20_dir / "out_gt.json"
    with pytest.raises(ValueError, match="Inconsistent GT: frame 10.*is outside valid sequence range"):
        parse_gt_to_json(synthetic_mot20_dir, meta, out_json, "video.mp4")


def test_inconsistent_gt_frame_zero(synthetic_mot20_dir):
    with open(synthetic_mot20_dir / "gt" / "gt.txt", "a") as f:
        # Frame 0 is invalid (MOT format is 1-indexed)
        f.write("0,1,10,10,50,100,1,1,1.0\n")

    meta = parse_seqinfo(synthetic_mot20_dir)
    out_json = synthetic_mot20_dir / "out_gt.json"
    with pytest.raises(ValueError, match="Inconsistent GT: frame 0.*is outside valid sequence range"):
        parse_gt_to_json(synthetic_mot20_dir, meta, out_json, "video.mp4")


def test_inconsistent_gt_invalid_track_id(synthetic_mot20_dir):
    with open(synthetic_mot20_dir / "gt" / "gt.txt", "a") as f:
        # Track ID 0 or negative
        f.write("1,0,10,10,50,100,1,1,1.0\n")

    meta = parse_seqinfo(synthetic_mot20_dir)
    out_json = synthetic_mot20_dir / "out_gt.json"
    with pytest.raises(ValueError, match="Inconsistent GT: invalid track ID"):
        parse_gt_to_json(synthetic_mot20_dir, meta, out_json, "video.mp4")


def test_duplicate_track_fails(synthetic_mot20_dir):
    with open(synthetic_mot20_dir / "gt" / "gt.txt", "a") as f:
        # Duplicate active pedestrian ID 1 on frame 1
        f.write("1,1,20,20,50,100,1,1,1.0\n")

    meta = parse_seqinfo(synthetic_mot20_dir)
    out_json = synthetic_mot20_dir / "out_gt.json"
    with pytest.raises(ValueError, match="Duplicate track annotation"):
        parse_gt_to_json(synthetic_mot20_dir, meta, out_json, "video.mp4")


# =============================================================================
# 4. Strict Metadata & Image/Video Integrity Tests
# =============================================================================

def test_missing_seqinfo_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="seqinfo.ini not found"):
        parse_seqinfo(tmp_path)


def test_invalid_seqinfo_missing_sequence_section(tmp_path):
    ini = tmp_path / "seqinfo.ini"
    ini.write_text("[InvalidSection]\nname=Test\n")
    with pytest.raises(ValueError, match="missing \\[Sequence\\] section"):
        parse_seqinfo(tmp_path)


def test_invalid_seqinfo_negative_seqlength(tmp_path):
    ini = tmp_path / "seqinfo.ini"
    ini.write_text("[Sequence]\nname=Test\nseqLength=-5\n")
    with pytest.raises(ValueError, match="seqLength must be a positive integer"):
        parse_seqinfo(tmp_path)


def test_invalid_seqinfo_negative_dimensions(tmp_path):
    ini = tmp_path / "seqinfo.ini"
    ini.write_text("[Sequence]\nname=Test\nseqLength=10\nimWidth=-640\n")
    with pytest.raises(ValueError, match="imWidth must be a positive integer"):
        parse_seqinfo(tmp_path)


def test_invalid_seqinfo_negative_fps(tmp_path):
    ini = tmp_path / "seqinfo.ini"
    ini.write_text("[Sequence]\nname=Test\nseqLength=10\nframeRate=0\n")
    with pytest.raises(ValueError, match="frameRate must be positive and finite"):
        parse_seqinfo(tmp_path)


def test_missing_image_frame_fails(synthetic_mot20_dir):
    meta = parse_seqinfo(synthetic_mot20_dir)
    out_mp4 = synthetic_mot20_dir / "out.mp4"
    (synthetic_mot20_dir / "img1" / "000002.jpg").unlink()

    with pytest.raises(FileNotFoundError, match="Missing expected image file"):
        compile_video(synthetic_mot20_dir, meta, out_mp4)


def test_corrupt_image_frame_fails(synthetic_mot20_dir):
    meta = parse_seqinfo(synthetic_mot20_dir)
    out_mp4 = synthetic_mot20_dir / "out.mp4"
    (synthetic_mot20_dir / "img1" / "000002.jpg").write_text("corrupt non-image binary bytes")

    with pytest.raises(ValueError, match="Failed to read image or corrupt file"):
        compile_video(synthetic_mot20_dir, meta, out_mp4)


def test_dimension_mismatch_fails(synthetic_mot20_dir):
    meta = parse_seqinfo(synthetic_mot20_dir)
    out_mp4 = synthetic_mot20_dir / "out.mp4"
    wrong_frame = np.zeros((500, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(synthetic_mot20_dir / "img1" / "000002.jpg"), wrong_frame)

    with pytest.raises(ValueError, match="dimension mismatch"):
        compile_video(synthetic_mot20_dir, meta, out_mp4)


def test_missing_img_dir_fails(synthetic_mot20_dir):
    meta = parse_seqinfo(synthetic_mot20_dir)
    out_mp4 = synthetic_mot20_dir / "out.mp4"
    shutil.rmtree(synthetic_mot20_dir / "img1")

    with pytest.raises(FileNotFoundError, match="Image directory not found"):
        compile_video(synthetic_mot20_dir, meta, out_mp4)


# =============================================================================
# 5. Zero-Occupancy Preservation & Empty Sequences
# =============================================================================

def test_zero_occupancy_frames_preserved(synthetic_mot20_dir):
    # Change seqLength to 5; frames 4 and 5 (0-indexed 3 and 4) must exist with 0 count
    with open(synthetic_mot20_dir / "seqinfo.ini", "r") as f:
        content = f.read().replace("seqLength=3", "seqLength=5")
    with open(synthetic_mot20_dir / "seqinfo.ini", "w") as f:
        f.write(content)

    meta = parse_seqinfo(synthetic_mot20_dir)
    out_json = synthetic_mot20_dir / "out_gt.json"
    gt = parse_gt_to_json(synthetic_mot20_dir, meta, out_json, "video.mp4")

    assert len(gt["occupancy"]) == 5
    assert gt["occupancy"][0]["person_count"] == 2
    assert gt["occupancy"][1]["person_count"] == 1
    assert gt["occupancy"][2]["person_count"] == 1
    assert gt["occupancy"][3]["person_count"] == 0
    assert gt["occupancy"][4]["person_count"] == 0


def test_completely_empty_gt(tmp_path):
    """Completely empty gt.txt produces zero-occupancy array across all frames."""
    seq_dir = tmp_path / "EMPTY-SEQ"
    seq_dir.mkdir()
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nname=EMPTY-SEQ\nimDir=img1\nframeRate=25\nseqLength=4\nimWidth=640\nimHeight=480\n")
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir()
    (gt_dir / "gt.txt").write_text("")

    meta = parse_seqinfo(seq_dir)
    out_json = seq_dir / "out.json"
    gt = parse_gt_to_json(seq_dir, meta, out_json, "video.mp4")

    validate_schema(gt)
    assert gt["metrics"]["peak_occupancy"] == 0
    assert gt["metrics"]["peak_occupancy_frame"] == 0
    assert len(gt["occupancy"]) == 4
    for occ in gt["occupancy"]:
        assert occ["person_count"] == 0


# =============================================================================
# 6. CLI Entrypoint Tests
# =============================================================================

def test_cli_main_success(synthetic_mot20_dir, tmp_path, monkeypatch):
    out_dir = tmp_path / "cli_out"
    test_args = [
        "prepare_mot20.py",
        "--seq-dir", str(synthetic_mot20_dir),
        "--out-dir", str(out_dir),
    ]
    monkeypatch.setattr("sys.argv", test_args)
    main()

    assert (out_dir / "MOT20-02.mp4").exists()
    assert (out_dir / "MOT20-02_gt.json").exists()


def test_cli_main_with_target_class(tmp_path, monkeypatch):
    seq_dir = tmp_path / "CLI-MULTICLASS"
    seq_dir.mkdir()
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nname=CLI-MULTICLASS\nimDir=img1\nframeRate=25\nseqLength=1\nimWidth=320\nimHeight=240\n")
    img_dir = seq_dir / "img1"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "000001.jpg"), np.zeros((240, 320, 3), dtype=np.uint8))
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir()
    (gt_dir / "gt.txt").write_text("1,1,10,10,20,40,1.0,2\n1,2,50,50,20,40,1.0,5\n")

    out_dir = tmp_path / "cli_out2"
    test_args = [
        "prepare_mot20.py",
        "--seq-dir", str(seq_dir),
        "--out-dir", str(out_dir),
        "--target-class-ids", "2",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    main()

    with open(out_dir / "CLI-MULTICLASS_gt.json") as f:
        gt = json.load(f)
    assert gt["metrics"]["peak_occupancy"] == 1


def test_cli_main_ambiguous_fails(tmp_path, monkeypatch):
    seq_dir = tmp_path / "CLI-AMBIGUOUS"
    seq_dir.mkdir()
    (seq_dir / "seqinfo.ini").write_text("[Sequence]\nname=CLI-AMBIGUOUS\nimDir=img1\nframeRate=25\nseqLength=1\nimWidth=320\nimHeight=240\n")
    img_dir = seq_dir / "img1"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "000001.jpg"), np.zeros((240, 320, 3), dtype=np.uint8))
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir()
    (gt_dir / "gt.txt").write_text("1,1,10,10,20,40,1.0,2\n1,2,50,50,20,40,1.0,5\n")

    out_dir = tmp_path / "cli_out3"
    test_args = [
        "prepare_mot20.py",
        "--seq-dir", str(seq_dir),
        "--out-dir", str(out_dir),
    ]
    monkeypatch.setattr("sys.argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
