import os
import csv
from datetime import datetime

def log_detections_per_frame_wide(detections, log_path="detections_log.csv", frame_id=0, fps=30.0):
    timestamp = datetime.now().isoformat()
    video_time_sec = frame_id / fps

    row_data = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "video_time_sec": round(video_time_sec, 3)
    }

    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["box"]
        x1 = round(x1, 4)
        y1 = round(y1, 4)
        x2 = round(x2, 4)
        y2 = round(y2, 4)
        cx = round((x1 + x2) / 2, 4)
        cy = round((y1 + y2) / 2, 4)
        row_data[f"person_{idx}_id"] = det.get("track_id", -1)
        row_data[f"person_{idx}_box"] = str([x1, y1, x2, y2])
        row_data[f"person_{idx}_xy"] = str([cx, cy])

    is_new_file = not os.path.exists(log_path)
    with open(log_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row_data.keys())
        if is_new_file:
            writer.writeheader()
        writer.writerow(row_data)
