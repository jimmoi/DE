import cv2

def get_video_fps(video_path):
    """
    Retrieves the Frames Per Second (FPS) of a video file.
    
    Args:
        video_path (str): The path to the video file.
        
    Returns:
        float: The FPS of the video. Returns -1.0 if the file cannot be opened.
    """
    # สร้าง object สำหรับอ่านวิดีโอ
    cap = cv2.VideoCapture(video_path)
    
    # ตรวจสอบว่าเปิดไฟล์วิดีโอได้หรือไม่
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return -1.0
    
    # ดึงค่า Frames Per Second (FPS) ของวิดีโอ
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # ปล่อย object สำหรับอ่านวิดีโอ
    cap.release()
    
    return fps

# ตัวอย่างการใช้งาน:
if __name__ == "__main__":
    # ใช้ video_path จากโค้ดที่คุณมีอยู่
    video_path = r"C:\Users\kunka\Documents\GitHub\DE\Vid_test\vdo_test_psdetec.mp4"
    
    fps_value = get_video_fps(video_path)
    
    if fps_value > 0:
        print(f"FPS ของวิดีโอคือ: {fps_value} fps")
    else:
        print("ไม่สามารถดึงค่า FPS จากวิดีโอได้")