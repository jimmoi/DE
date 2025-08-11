from util_module.camera_module import VideoFileCamera
from util_module.ai_model_module import YOLOv8HumanDetector, STRONGSORT_DEFAULT_CFG
from collect_log import log_detections_per_frame_wide

import cv2
import time
from tqdm import tqdm

def main():
    # video_path = r"C:\Users\kunka\Documents\GitHub\DE\test_vidio.mp4"
    video_path = r"Vid_test\front_room.mp4"
    camera = VideoFileCamera(video_path, 0.5, "test")
    
    model = YOLOv8HumanDetector(
        model_name='yolov8n.pt',
        device='auto',
        confidence_threshold=0.01,
        iou_threshold=0.7,
        tracker_config_path=STRONGSORT_DEFAULT_CFG
    )


    camera.start()
    
    total_frames = 0
    while camera.is_running:
        temp_frame = camera.get_frame()
        if temp_frame is not None:
            height, width = temp_frame.shape[:2]
            fps = camera.fps
            total_frames = camera.frame_count
            break
    height, width = height*camera.scale_factor, width*camera.scale_factor

    frame_counter = 0
    progress_bar = tqdm(total=total_frames, desc="Processing Frames")
    
    try:
        while camera.is_running:
            frame = camera.get_frame()
            if frame is not None:
                frame = camera.preprocess_frame(frame, camera.scale_factor)
                detections = model.predict(frame)
    
                # draw_detections(frame, detections) 
                for track in detections:
                    x1, y1, x2, y2 = track['box']
                    track_id = track['track_id']
                    # Draw bounding box and track ID
                    x1 = x1 * width
                    y1 = y1 * height
                    x2 = x2 * width
                    y2 = y2 * height
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                log_detections_per_frame_wide(detections, frame_id=frame_counter, fps=fps)
                cv2.imshow("Tracking", frame)
                frame_counter += 1
                progress_bar.update(1)
            else:
                print("Waiting for frame...")
                time.sleep(0.1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        progress_bar.close()

if __name__ == "__main__":
    main()

