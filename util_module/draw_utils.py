import cv2

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"ID {det.get('person_id', '-')}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
