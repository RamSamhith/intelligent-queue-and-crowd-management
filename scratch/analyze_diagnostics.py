"""Comprehensive analysis script for ByteTrack and pipeline diagnostic telemetry."""
import json
import numpy as np

with open("scratch/acceptance_test_log.json", "r", encoding="utf-8") as f:
    entries = json.load(f)

print(f"Total frames recorded: {len(entries)}")

# 1. Inspect Track Creations and ID Replacements
all_created = []
all_replacements = []
all_lost = []
all_terminations = []
unique_ids = set()

for e in entries:
    fid = e["frame_id"]
    t = e["elapsed_sec"]
    created = e.get("tracks_created", [])
    lost = e.get("tracks_lost", [])
    replacements = e.get("id_replacements", [])
    terminations = e.get("track_terminations", [])
    active = e.get("active_tracks", [])
    dets = e.get("detections", [])
    
    for c in created:
        unique_ids.add(c["track_id"])
        all_created.append({"frame_id": fid, "time": t, "created": c, "dets": dets, "active_count": len(active)})
        
    for r in replacements:
        all_replacements.append({"frame_id": fid, "time": t, "replacement": r})
        
    for l in lost:
        all_lost.append({"frame_id": fid, "time": t, "lost": l})
        
    for term in terminations:
        all_terminations.append({"frame_id": fid, "time": t, "termination": term})

print(f"\nTotal Unique IDs Created: {len(unique_ids)} -> {sorted(list(unique_ids))}")
print(f"Total Track Creation Events: {len(all_created)}")
print(f"Total Track Lost Events: {len(all_lost)}")
print(f"Total Track Termination Events: {len(all_terminations)}")
print(f"Total Spatial ID Replacement Events: {len(all_replacements)}")

print("\n--- ALL TRACK CREATION EVENTS ---")
for c in all_created:
    info = c["created"]
    print(f"Frame {c['frame_id']:4d} (t={c['time']:5.2f}s): Created Track ID {info['track_id']} | conf={info['confidence']} | bbox={info['bbox']} | reason: {info['reason']}")

print("\n--- ALL DETECTED ID REPLACEMENTS (Spatial overlaps with lost tracks) ---")
for r in all_replacements:
    rep = r["replacement"]
    print(f"Frame {r['frame_id']:4d} (t={r['time']:5.2f}s): New ID {rep['new_track_id']} replaced lost ID {rep['replaced_lost_id']} | IoU={rep['iou']} | frames_since_lost={rep['frames_since_lost']} | conf={rep['confidence']}")
    print(f"   lost_bbox: {rep['lost_bbox']} -> new_bbox: {rep['new_bbox']}")

# 2. Analyze why tracks were lost or failed matching
print("\n--- TRACK LIFECYCLE & FAILURE REASON ANALYSIS ---")
for e in entries:
    matching_res = e.get("matching_result", {})
    s1 = matching_res.get("stage1_high_conf", {})
    unmatched_trks = s1.get("unmatched_track_ids", [])
    unmatched_dets = s1.get("unmatched_det_indices", [])
    dets = e.get("detections", [])
    active = e.get("active_tracks", [])
    
    if unmatched_trks or unmatched_dets or e.get("tracks_lost") or e.get("tracks_created"):
        ious = e.get("association_ious", [])
        if unmatched_trks and dets:
            print(f"Frame {e['frame_id']:4d} (t={e['elapsed_sec']:5.2f}s): Unmatched Tracks: {unmatched_trks}, Unmatched Dets: {unmatched_dets}, Num Dets: {len(dets)}")
            for d in dets:
                print(f"   Det: conf={d['conf'] if 'conf' in d else d['confidence']}, bbox={d['bbox']}")
            for iou_info in ious:
                print(f"   Candidate IoU with track {iou_info['track_id']} (state {iou_info['track_state']}): IoU={iou_info['iou']}, det_conf={iou_info['det_conf']}")
