"""Deep-dive diagnosis: why are ROI counts always 0 despite detections.

The ROI in the acceptance test was:
  ROIConfig(x=280, y=50, width=340, height=390)
  -> x range: [280, 620), y range: [50, 440)

The ROIFilter is initialized in coordinator.py with:
  frame_width=self._config.roi.x2   = 280 + 340 = 620
  frame_height=self._config.roi.y2  = 50 + 390  = 440

The validate_roi call is:
  validate_roi(roi, frame_width=620, frame_height=440)

This calls roi.clamp_to_frame(620, 440):
  x1 = max(0, min(280, 619)) = 280
  y1 = max(0, min(50, 439))  = 50
  x2 = max(281, min(620, 620)) = 620
  y2 = max(51, min(440, 440))  = 440

So clamped ROI = ROIConfig(x=280, y=50, width=340, height=390)
That actually looks correct...

But wait - the acceptance test log shows unique_approx going to 21.
That means session counter IS counting unique tracks (via SessionCounter).
But OccupancyCounter is always 0.

Let me check the actual log data to see if tracks_count is in the log.
"""
import json

with open('scratch/acceptance_test_log.json') as f:
    entries = json.load(f)

# Check what fields are in the log
print("Log entry fields:", list(entries[0].keys()))

# Check some entries during phase 2 when person was in ROI
phase2 = [e for e in entries if 8.0 <= e['elapsed_sec'] < 16.0]
print(f"\nSample phase 2 entries (every 15 frames):")
for e in phase2[::15]:
    print(f"  t={e['elapsed_sec']:.1f}s | current={e['current_count']} | unique={e['unique_approx']}")
    # Print all fields
    for k, v in e.items():
        if k not in ('frame_id', 'phase', 'loop_latency_ms', 'yolo_latency_ms', 'frame_age_ms', 'fps', 'active_alerts'):
            print(f"    {k}: {v}")
    print()
