from camera_module import VideoFileCamera
from ai_model_module import YOLOv8HumanDetector
from draw_utils import draw_detections
from logger import log_detections_per_frame_wide

import cv2
import time

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
        
    )

    camera.start()
    frame_counter = 0

    try:
        while camera.is_running:
            frame = camera.get_frame()
            if frame is not None:
                detections = model.predict(frame)
                draw_detections(frame, detections)
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
