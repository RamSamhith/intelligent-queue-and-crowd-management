"""Trace exact lifecycle and failure mechanism for each track ID."""
import json

with open("scratch/acceptance_test_log.json", "r", encoding="utf-8") as f:
    entries = json.load(f)

# Track IDs seen
track_ids = set()
for e in entries:
    for c in e.get("tracks_created", []):
        track_ids.add(c["track_id"])

print(f"Track IDs created: {sorted(list(track_ids))}")

for tid in sorted(list(track_ids)):
    print(f"\n=======================================================")
    print(f"               LIFECYCLE FOR TRACK ID {tid}")
    print(f"=======================================================")
    
    creation_frame = None
    last_tracked_frame = None
    lost_frames = []
    termination_frame = None
    
    for e in entries:
        fid = e["frame_id"]
        t = e["elapsed_sec"]
        
        # Check created
        for c in e.get("tracks_created", []):
            if c["track_id"] == tid:
                creation_frame = (fid, t, c)
                
        # Check active
        for a in e.get("active_tracks", []):
            if a["track_id"] == tid:
                last_tracked_frame = (fid, t, a)
                
        # Check lost
        for l in e.get("tracks_lost", []):
            if l["track_id"] == tid:
                lost_frames.append((fid, t, l))
                
        # Check termination
        for term in e.get("track_terminations", []):
            if term["track_id"] == tid:
                termination_frame = (fid, t, term)

    print(f"Created: Frame {creation_frame[0]} (t={creation_frame[1]:.2f}s) | conf={creation_frame[2]['confidence']} | bbox={creation_frame[2]['bbox']}")
    if last_tracked_frame:
        print(f"Last Tracked: Frame {last_tracked_frame[0]} (t={last_tracked_frame[1]:.2f}s) | bbox={last_tracked_frame[2]['bbox']}")
    if lost_frames:
        print(f"First Marked Lost: Frame {lost_frames[0][0]} (t={lost_frames[0][1]:.2f}s)")
    if termination_frame:
        print(f"Terminated: Frame {termination_frame[0]} (t={termination_frame[1]:.2f}s) | reason={termination_frame[2]['reason']}")

    # Now let's inspect the frame where it became lost and why detections weren't matched!
    if lost_frames:
        lf_id = lost_frames[0][0]
        # Look around lf_id - 2 to lf_id + 5
        print(f"\n--- Analysis around frame {lf_id} (when Track {tid} was marked lost) ---")
        for e in entries:
            if lf_id - 2 <= e["frame_id"] <= lf_id + 10:
                fid = e["frame_id"]
                t = e["elapsed_sec"]
                dets = e.get("detections", [])
                s1 = e.get("matching_result", {}).get("stage1_high_conf", {})
                s2 = e.get("matching_result", {}).get("stage2_low_conf", {})
                s1_matches = s1.get("matches", [])
                s2_matches = s2.get("matches", [])
                unm_trks = s1.get("unmatched_track_ids", [])
                unm_dets = s1.get("unmatched_det_indices", [])
                ious = e.get("association_ious", [])
                
                print(f"Frame {fid:4d} (t={t:5.2f}s): {len(dets)} det(s)")
                for i, d in enumerate(dets):
                    print(f"   Det[{i}]: conf={d['confidence']:.3f} | bbox={d['bbox']}")
                for iou_info in ious:
                    if iou_info["track_id"] == tid:
                        print(f"   Stage 1 candidate IoU with Track {tid} (state={iou_info['track_state']}): IoU={iou_info['iou']:.3f}, det_conf={iou_info['det_conf']:.3f}")
                if s1_matches:
                    print(f"   Stage 1 matches: {s1_matches}")
                if s2_matches:
                    print(f"   Stage 2 matches: {s2_matches}")
                if unm_trks:
                    print(f"   Unmatched track IDs: {unm_trks}")
                if unm_dets:
                    print(f"   Unmatched det indices: {unm_dets}")
