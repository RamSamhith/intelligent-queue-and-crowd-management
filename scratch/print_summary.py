"""Print detailed summary of track creations, replacements, and terminations."""
import json

with open("scratch/acceptance_test_log.json", "r", encoding="utf-8") as f:
    entries = json.load(f)

print("=== TRACK CREATION SUMMARY ===")
for e in entries:
    for c in e.get("tracks_created", []):
        print(f"Frame {e['frame_id']:4d} (t={e['elapsed_sec']:5.2f}s): Track ID {c['track_id']} CREATED | conf={c['confidence']:.3f} | bbox={c['bbox']} | reason={c['reason']}")

print("\n=== TRACK REPLACEMENT (SPATIAL OVERLAP) SUMMARY ===")
for e in entries:
    for r in e.get("id_replacements", []):
        print(f"Frame {e['frame_id']:4d} (t={e['elapsed_sec']:5.2f}s): New ID {r['new_track_id']} replaced lost ID {r['replaced_lost_id']} | IoU={r['iou']:.3f} | frames_lost={r['frames_since_lost']} | conf={r['confidence']:.3f}")
        print(f"   lost_bbox={r['lost_bbox']} -> new_bbox={r['new_bbox']}")

print("\n=== TRACK TERMINATION SUMMARY ===")
for e in entries:
    for term in e.get("track_terminations", []):
        print(f"Frame {e['frame_id']:4d} (t={e['elapsed_sec']:5.2f}s): Track ID {term['track_id']} TERMINATED | reason={term['reason']}")
