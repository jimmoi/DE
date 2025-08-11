import os
import csv
from datetime import datetime

# เก็บสถานะว่าไฟล์เริ่มเขียน header หรือยัง
log_file_initialized = False

def log_detections_per_frame_wide(detections, log_path="detections_log.csv", frame_id=0, fps=30.0):
    """
    เขียน log แบบ 1 แถวต่อ 1 คนที่ detect ได้
    Columns: frame_id, timestamp, video_time_sec, person_id, person_box, person_xy
    """
    global log_file_initialized

    timestamp = datetime.now().isoformat()
    video_time_sec = round(frame_id / fps, 3)

    # ถ้ายังไม่เริ่มต้นไฟล์ เขียน header ก่อน
    if not log_file_initialized:
        with open(log_path, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["frame_id", "timestamp", "video_time_sec", "person_id", "person_box", "person_xy"])
            writer.writeheader()
        log_file_initialized = True

    # เพิ่มข้อมูลทีละคน
    with open(log_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["frame_id", "timestamp", "video_time_sec", "person_id", "person_box", "person_xy"])

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            # ปรับความละเอียดตำแหน่งเป็นทศนิยม 4 ตำแหน่ง
            x1 = round(x1, 4)
            y1 = round(y1, 4)
            x2 = round(x2, 4)
            y2 = round(y2, 4)
            cx = round((x1 + x2) / 2, 4)
            cy = round((y1 + y2) / 2, 4)

            row_data = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "video_time_sec": video_time_sec,
                "person_id": det.get("track_id", -1),
                "person_box": str([x1, y1, x2, y2]),
                "person_xy": str([cx, cy])
            }
            writer.writerow(row_data)
