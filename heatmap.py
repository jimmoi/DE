import cv2
import pandas as pd
import numpy as np
import ast
import os

# ===== CONFIG =====
VIDEO_PATH = r"Vid_test/front_room.mp4"   # แก้ path ให้ตรงกับไฟล์จริง
LOG_PATH = "detections_log.csv"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
RADIUS = 20  # ขนาดจุดคนใน heatmap
# ==================

def parse_xy(xy_str):
    """แปลงจากสตริงพิกัด normalized -> พิกัด pixel"""
    try:
        xy = ast.literal_eval(str(xy_str))
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            cx = int(float(xy[0]) * FRAME_WIDTH)
            cy = int(float(xy[1]) * FRAME_HEIGHT)
            return cx, cy
    except:
        pass
    return None

def main():
    # อ่าน CSV โดยข้ามบรรทัดเสีย
    df = pd.read_csv(LOG_PATH, on_bad_lines='skip')

    # ตรวจว่า column frame_id เป็น int
    if 'frame_id' in df.columns:
        df['frame_id'] = pd.to_numeric(df['frame_id'], errors='coerce').fillna(-1).astype(int)
    else:
        raise ValueError("ไม่พบคอลัมน์ frame_id ใน log")

    cap = cv2.VideoCapture(VIDEO_PATH)
    heatmap_accumulator = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # filter เฉพาะแถวของเฟรมปัจจุบัน
        rows = df[df["frame_id"] == frame_id]

        if not rows.empty:
            for _, row in rows.iterrows():
                for col in rows.columns:
                    if "_xy" in col:
                        coords = parse_xy(row[col])
                        if coords:
                            cv2.circle(heatmap_accumulator, coords, RADIUS, 1, -1)

        # normalize และสร้าง heatmap
        heatmap_norm = cv2.normalize(heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = np.uint8(heatmap_norm)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # overlay กับวิดีโอ
        overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
        cv2.imshow("Heatmap Live", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

    # บันทึก heatmap สุดท้าย
    cv2.imwrite("final_heatmap.png", heatmap_color)
    print("บันทึก final_heatmap.png แล้ว")

if __name__ == "__main__":
    main()