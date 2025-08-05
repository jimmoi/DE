from camera_module import VideoFileCamera
from ultralytics import YOLO
import cv2
import time
import csv
import os
from datetime import datetime

# --- วาดกรอบและ ID บนภาพ ---
def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        person_id = det["person_id"]
        label = f"ID {person_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# --- ดึงข้อมูลออกจากผลลัพธ์ของ YOLO track() ---
def extract_detections_with_id(result) -> list:
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            track_id = int(box.id[0]) if box.id is not None else -1

            detections.append({
                'person_id': track_id,
                'box': [x1, y1, x2, y2],
                'score': conf,
                'class_id': class_id,
                'class_name': class_name
            })
    return detections

# --- บันทึก log แบบ wide-format (1 row ต่อ 1 frame) ---
def log_detections_per_frame_wide(detections, log_path="detections_log.csv", frame_id=0):
    from datetime import datetime
    import os
    import csv

    timestamp = datetime.now().isoformat()
    row_data = {
        "frame_id": frame_id,
        "timestamp": timestamp
    }

    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["box"]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        row_data[f"person_{idx}_id"] = det.get("person_id", -1)
        row_data[f"person_{idx}_box"] = str([x1, y1, x2, y2])
        row_data[f"person_{idx}_xy"] = str([cx, cy])  # ✅ center point

    is_new_file = not os.path.exists(log_path)

    with open(log_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row_data.keys())
        if is_new_file:
            writer.writeheader()
        writer.writerow(row_data)


# --- main ---
def main():
    video_path = r"C:\Hemoglobin\project\DE\Vid_test\vdo_test_psdetec.mp4"
    camera = VideoFileCamera("VideoTest", video_path)
    # โหลด YOLOv8 + StrongSORT
    model = YOLO('yolov8n.pt')  # สามารถใช้ yolov8s.pt หรือ yolov8m.pt ได้


    camera.start()
    frame_counter = 0

    try:
        while camera.is_running:
            frame = camera.get_frame()
            if frame is not None:
                # ตรวจจับและติดตามคน (class_id 0)
                results = model.track(
                    source=frame,
                    persist=True,
                    classes=[0],
                    conf=0.5,
                    iou=0.7,
                    verbose=False
                )

                if results:
                    detections = extract_detections_with_id(results[0])

                    # วาด
                    draw_detections(frame, detections)

                    # log เป็น wide row
                    # log_detections_per_frame_wide(detections, frame_id=frame_counter)

                    cv2.imshow("YOLOv8 Tracking", frame)
                    frame_counter += 1
            else:
                print("Waiting for frame...")
                time.sleep(0.1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
