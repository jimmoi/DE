from camera_module import VideoFileCamera
from ai_model_module import YOLOv8HumanDetector
import cv2
import time
import csv
import os
from datetime import datetime

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"ID {det.get('person_id', '-')}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def log_detections_per_frame_wide(detections, log_path="detections_log.csv", frame_id=0, fps=30.0):
    from datetime import datetime
    import os
    import csv

    timestamp = datetime.now().isoformat()
    video_time_sec = frame_id / fps  # ✅ เวลาในวิดีโอ (เป็นวินาที)

    row_data = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "video_time_sec": round(video_time_sec, 3)  # ✅ บันทึกเวลาในวิดีโอ     
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


def main():
    video_path = r"C:\Users\kunka\Documents\GitHub\DE\test_vidio.mp4"
    camera = VideoFileCamera("VideoTest", video_path)

    # ✅ อ่าน FPS จริงของวิดีโอ
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Video FPS: {fps:.2f}")
    

    model = YOLOv8HumanDetector(
        model_name='yolov8n.pt',
        device='auto',
        confidence_threshold=0.5,
        iou_threshold=0.7,
        use_tracking=True  # ✅ เปิด tracking
    )
    camera.start()
    frame_counter = 0

    try:
        while camera.is_running:
            frame = camera.get_frame()
            if frame is not None:
                detections = model.predict(frame)
                draw_detections(frame, detections)
                 # ✅ ส่งค่า fps เข้าไปด้วย
                log_detections_per_frame_wide(detections, frame_id=frame_counter, fps=fps)

                cv2.imshow("Tracking", frame)
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