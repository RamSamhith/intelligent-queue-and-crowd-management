"""Analyze the acceptance test log for ROI/tracking diagnosis."""
import json

with open('scratch/acceptance_test_log.json') as f:
    entries = json.load(f)

print(f"Total frames: {len(entries)}")
print(f"Total duration: {entries[-1]['elapsed_sec']:.1f}s")

# Phase 2: when person should be in ROI (t=8-16s)
phase2 = [e for e in entries if 8.0 <= e['elapsed_sec'] < 16.0]
print(f"\nPhase 2 frames (t=8-16s): {len(phase2)}")
for e in phase2[::10]:
    print(f"  t={e['elapsed_sec']:.1f}s frame={e['frame_id']} current={e['current_count']} unique={e['unique_approx']} yolo={e['yolo_latency_ms']:.1f}ms")

# Check any frame with current_count > 0
nonzero = [e for e in entries if e['current_count'] > 0]
print(f"\nFrames with current_count > 0: {len(nonzero)}")

# Track unique_approx growth (proves detections are happening)
print("\nunique_approx timeline (every 5s):")
for t in range(0, 50, 5):
    nearest = min(entries, key=lambda e: abs(e['elapsed_sec'] - t))
    print(f"  t={t}s: current={nearest['current_count']} unique={nearest['unique_approx']}")

# Check max unique_approx
max_unique = max(e['unique_approx'] for e in entries)
print(f"\nMax unique_approx: {max_unique}")
print(f"Max current_count: {max(e['current_count'] for e in entries)}")

# Find frames where unique increased (detections found)
prev_uniq = 0
detection_frames = 0
for e in entries:
    if e['unique_approx'] > prev_uniq:
        detection_frames += 1
        prev_uniq = e['unique_approx']
print(f"Frames with new unique ID detected: {detection_frames}")
