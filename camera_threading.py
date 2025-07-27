from my_utils import timer
import numpy as np
import cv2
import json

class camera(CAM):
    def __init__(self, cam_data:dict) -> None:
        self.address:str = cam_data["address"]
        self.name:str = cam_data.get("name", "Unnamed Camera")
        self.width:int = None
        self.height:int = None
        self.fps:int = None
        self.resolution:list[int,int] = [self.width, self.height]
        self.status:int = 0 # 0: Not Initialized, 1: Initialized, 2: Error
        self.cap = None
        self.initialize_cam()
        
    def initialize_cam(self) -> None:
        with timer(f"timing cam: \"{self.name}\""):
            self.cap = cv2.VideoCapture(self.address)
            if not self.cap.isOpened():
                self.status = 2
                self.cap = None
                raise RuntimeError(f"Failed to open camera with name: {self.name} address: {self.address} ")
            
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.resolution = [self.width, self.height]
            self.status = 1
            print(f"Camera: \"{self.name}\" address: \"{self.address}\" initialized with resolution {self.resolution} and FPS {self.fps}. Status: {self.status}")
    
    @staticmethod
    def crop_frame(frame, x, y, width, height):
        """
        Crop the frame to the specified rectangle.
        
        :param frame: The input frame to crop.
        :param x: The x-coordinate of the top-left corner of the crop rectangle.
        :param y: The y-coordinate of the top-left corner of the crop rectangle.
        :param width: The width of the crop rectangle.
        :param height: The height of the crop rectangle.
        :return: Cropped frame.
        """
        return frame[y:y+height, x:x+width]
    
    @staticmethod
    def resize_frame(frame, factor):
        """
        Resize the frame by a given factor.
        
        :param frame: The input frame to resize.
        :param factor: The scaling factor.
        :return: Resized frame.
        """
        return cv2.resize(frame, None, fx=factor, fy=factor,interpolation=cv2.INTER_LINEAR)

        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    with open("camera_cf.json", "r") as file:
        cam_config = json.load(file)
        
        
    # for cam_name, cam_info in cam_config.items():
    #     try:
    #         cam = camera(cam_info["address"], cam_info.get("name", "Unnamed Camera"))
    #         cam.crop(100, 100, 640, 480)
    #         print(f"Camera cropped to resolution {cam.resolution}.")
    #     except RuntimeError as e:
    #         print(f"Error initializing camera {cam_name}: {e}")
    
    cam = camera(cam_config["camara_2"])
    while True:
        ret, frame = cam.cap.read()
        # frame = camera.crop_frame(frame, 0, 0, 200, 200)
        frame = camera.resize_frame(frame, 0.5)
        if not ret:
            print("Failed to grab frame")
            break
        
        cv2.imshow(cam.name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    