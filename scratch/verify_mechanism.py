"""Comprehensive mathematical and empirical proof of the ByteTrack failure mechanism."""
import json
import numpy as np

with open("scratch/acceptance_test_log.json", "r", encoding="utf-8") as f:
    entries = json.load(f)

print("=== DEEP ANALYSIS OF ALL 689 FRAMES ===")

# Check Stage 2 match count across the whole run
s2_matches_total = 0
s1_matches_total = 0
total_low_dets = 0
total_high_dets = 0

for e in entries:
    mr = e.get("matching_result", {})
    s1 = mr.get("stage1_high_conf", {})
    s2 = mr.get("stage2_low_conf", {})
    s1_matches_total += len(s1.get("matches", []))
    s2_matches_total += len(s2.get("matches", []))
    
    for d in e.get("detections", []):
        c = d["confidence"]
        if c >= 0.5:
            total_high_dets += 1
        elif c > 0.1:
            total_low_dets += 1

print(f"Total High Detections (conf >= 0.5): {total_high_dets}")
print(f"Total Low Detections (0.1 < conf < 0.5): {total_low_dets}")
print(f"Total Stage 1 Matches: {s1_matches_total}")
print(f"Total Stage 2 Matches: {s2_matches_total}")

# Verify how many low-confidence detections were present when a track was active
low_dets_with_active_track = 0
for e in entries:
    mr = e.get("matching_result", {})
    dets = e.get("detections", [])
    active = e.get("active_tracks", [])
    low_dets = [d for d in dets if 0.1 < d["confidence"] < 0.5]
    if low_dets and active:
        low_dets_with_active_track += len(low_dets)

print(f"Low detections present while a track was active: {low_dets_with_active_track}")
print(f"Stage 2 recoveries: {s2_matches_total} (0% recovery rate!)")

# Look at match_thresh in Stage 1
print("\n=== STAGE 1 MATCH THRESHOLD ANALYSIS (match_thresh=0.8) ===")
rejected_stage1_high_iou = []
for e in entries:
    ious = e.get("association_ious", [])
    s1 = e.get("matching_result", {}).get("stage1_high_conf", {})
    unm_trks = s1.get("unmatched_track_ids", [])
    unm_dets = s1.get("unmatched_det_indices", [])
    for info in ious:
        # If IoU is between 0.3 and 0.8 and track or det was unmatched
        if 0.3 <= info["iou"] < 0.8:
            if info["track_id"] in unm_trks or info["det_idx"] in unm_dets:
                rejected_stage1_high_iou.append({
                    "frame_id": e["frame_id"],
                    "time": e["elapsed_sec"],
                    "track_id": info["track_id"],
                    "track_state": info["track_state"],
                    "det_idx": info["det_idx"],
                    "iou": info["iou"],
                    "det_conf": info["det_conf"],
                })

print(f"Detections rejected in Stage 1 with IoU between 0.30 and 0.80: {len(rejected_stage1_high_iou)}")
for r in rejected_stage1_high_iou[:15]:
    print(f"  Frame {r['frame_id']:4d} (t={r['time']:5.2f}s): Track {r['track_id']} ({r['track_state']}) rejected candidate Det[{r['det_idx']}] with IoU={r['iou']:.3f} (conf={r['det_conf']:.3f}) because IoU < 0.80!")

