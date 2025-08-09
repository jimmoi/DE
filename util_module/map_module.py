import cv2
import time
import os
import json
import numpy as np
import random
from abc import ABC, abstractmethod

from util_module.camera_module import VideoFileCamera

class Base_ROI(ABC):
    """
    Abstract base class for map-related functionalities, including ROI creation and editing.
    """
    
    # Internal state variables for mouse events
    _drawing_roi = False
    _editing_mode = False
    _editing_point = None
    _selected_roi_name = None
    _start_point = None
    _end_point = None
    _roi_data_temp = {}
    _roi_colors = {}
    _roi_counter = 0
    _color_temp = None
    _roi_size_threshold = 30  # Minimum size for a valid ROI

    
    def __init__(self, map_name, map_path, map_dir, roi_dir, map_roi_dir, scale_factor):
        self._roi_dir = roi_dir
        self._map_dir = map_dir
        self._map_roi_dir = map_roi_dir
        self.map_name = map_name 
        self._map_path = map_path
        self._scale_factor = scale_factor
        self._source_image = None

        
        
    @abstractmethod
    def _load_source(self):
        """
        Abstract method to load the source image (map or camera frame).
        """
        pass
    
    @abstractmethod
    def _preprocess_map(self, image, scale_factor=1.0):
        """
        Abstract method to preprocess the map image if needed.
        """
        pass

    @abstractmethod
    def _get_roi_path(self):
        """
        Abstract to generates the path for the ROI JSON file.
        """

    def _load_roi_data(self):
        """
        Loads ROI data from the JSON file.
        """
        if os.path.exists(self._roi_path):
            with open(self._roi_path, 'r') as f:
                try:
                    roi_data = json.load(f)
                    print(f"ROI data loaded from {self._roi_path}")
                    return roi_data
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {self._roi_path}. Initializing with empty data.")
                    return {}
        else:
            print(f"No ROI data file found at {self._roi_path}. Creating a new one.")
            self._save_roi_data({})
            return {}

    def _save_roi_data(self, roi_data):
        """
        Saves ROI data to the JSON file.
        """
        with open(self._roi_path, 'w') as f:
            json.dump(roi_data, f, indent=4)
        print(f"ROI data saved to {self._roi_path}")
    
    def _draw_rois(self, image):
        """
        Draws all ROIs from _roi_data_temp onto the provided image.
        """
        overlay = image.copy()
        for roi_name, points_dict in self._roi_data_temp.items():
            if roi_name not in self._roi_colors:
                self._roi_colors[roi_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            color = self._roi_colors[roi_name]
            points = np.array(list(points_dict.values()), np.int32)
            
            cv2.fillPoly(overlay, [points], color)
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
            
            for i, p in enumerate(points):
                cv2.putText(image, str(i + 1), (p[0], p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)     

        for roi_name, points_dict in self._roi_data_temp.items():
            points = np.array(list(points_dict.values()), np.int32)
            color = self._roi_colors[roi_name]
            center_x = int(np.mean([p[0] for p in points]))
            center_y = int(np.mean([p[1] for p in points]))
            cv2.putText(image, roi_name, (center_x-30, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image

    def _is_point_near_roi_point(self, x, y, roi_name, radius=10):
        """
        Checks if a point (x, y) is near any of the four points of an ROI.
        Returns the index of the point if it's near, otherwise None.
        """
        points_dict = self._roi_data_temp.get(roi_name, {})
        for i, point in points_dict.items():
            px, py = point
            if np.sqrt((px - x) ** 2 + (py - y) ** 2) < radius:
                return i
        return None

    def _is_point_in_roi(self, x, y, roi_name):
        """
        Checks if a point (x, y) is inside the ROI rectangle.
        """
        points_dict = self._roi_data_temp.get(roi_name, {})
        if not points_dict:
            return False
        
        points = np.array(list(points_dict.values()), np.int32)
        return cv2.pointPolygonTest(points, (x, y), False) >= 0

    def _mouse_callback(self, event, x, y, flags, param):
        """
        Unified mouse callback function for both creation and editing.
        """
        if self._editing_mode:
            # --- EDITING MODE LOGIC ---
            if event == cv2.EVENT_LBUTTONDOWN:
                self._selected_roi_name = None
                for roi_name in self._roi_data_temp:
                    point_idx = self._is_point_near_roi_point(x, y, roi_name)
                    if point_idx:
                        self._editing_point = (roi_name, point_idx)
                        self._selected_roi_name = roi_name
                        break
                    
                    if self._is_point_in_roi(x, y, roi_name):
                        self._editing_point = (roi_name, 'center')
                        self._selected_roi_name = roi_name
                        self._start_point = (x, y)
                        break

            elif event == cv2.EVENT_LBUTTONUP:
                self._editing_point = None
            
            elif event == cv2.EVENT_MOUSEMOVE and self._editing_point:
                roi_name, point_type = self._editing_point
                if point_type == 'center':
                    dx = x - self._start_point[0]
                    dy = y - self._start_point[1]
                    for point_idx in self._roi_data_temp[roi_name]:
                        px, py = self._roi_data_temp[roi_name][point_idx]
                        self._roi_data_temp[roi_name][point_idx] = [int(px + dx), int(py + dy)]
                    self._start_point = (x, y)
                else:
                    self._roi_data_temp[roi_name][point_type] = [x, y]
        else:
            # --- CREATION MODE LOGIC ---
            if event == cv2.EVENT_LBUTTONDOWN:
                self._color_temp = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                self._drawing_roi = True
                self._start_point = (x, y)
                self._selected_roi_name = None
                for roi_name in self._roi_data_temp:
                    if self._is_point_in_roi(x, y, roi_name):
                        self._selected_roi_name = roi_name
                        break

            elif event == cv2.EVENT_LBUTTONUP:
                if self._drawing_roi:
                    self._drawing_roi = False
                    if self._start_point and self._start_point != (x, y):
                        self._roi_counter += 1
                        roi_name = input(f"Enter name for new ROI (default: area_{self._roi_counter}): ") or f"area_{self._roi_counter}"
                        x1, y1 = self._start_point
                        x2, y2 = self._end_point if self._end_point else (x, y)
                        self._roi_data_temp[roi_name] = {
                            1: [x1, y1],
                            2: [x2, y1],
                            3: [x2, y2],
                            4: [x1, y2],
                        }
                        self._roi_colors[roi_name] = self._color_temp
                        print(f"Created new ROI: {roi_name}")
                self._start_point = None
                self._end_point = None
                self._color_temp = None
            
            elif event == cv2.EVENT_MOUSEMOVE and self._drawing_roi:
                if self._start_point and np.sqrt((x - self._start_point[0]) ** 2 + (y - self._start_point[1]) ** 2) > self._roi_size_threshold:
                    self._end_point = (x, y)
                else:
                    if self._start_point:
                        self._end_point = (self._start_point[0] + self._roi_size_threshold, self._start_point[1] + self._roi_size_threshold)
                        
    def _write_image(self, image, path):
        """
        Writes the processed image to a file.
        """
        cv2.imwrite(path, image)
        print(f"Image saved as {self.map_name} in processed_images directory.")

    def create_roi(self):
        """
        Allows the user to create new ROIs with mouse clicks.
        """
        print("--- Starting ROI Creation Mode ---")
        print("Click and drag to create a rectangle. Press 'r' to remove a hovered ROI, 's' to save, 'q' to quit.")
        
        self._editing_mode = False
        self._roi_data_temp = self._roi.copy()
        
        cv2.namedWindow(self.map_name)
        cv2.setMouseCallback(self.map_name, self._mouse_callback)

        while True:
            temp_image = self._source_image.copy()

            if self._drawing_roi and self._start_point and self._end_point:
                x1, y1 = self._start_point
                x2, y2 = self._end_point
                
                overlay = temp_image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), self._color_temp, -1)
                temp_image = cv2.addWeighted(overlay, 0.5, temp_image, 0.5, 0)
                
                cv2.rectangle(temp_image, (x1, y1), (x2, y2), self._color_temp, 2)
                cv2.putText(temp_image, "1", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._color_temp, 2)
                cv2.circle(temp_image, (x1, y1), 5, self._color_temp, -1)
                
                cv2.putText(temp_image, "2", (x2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._color_temp, 2)
                cv2.circle(temp_image, (x2, y1), 5, self._color_temp, -1)
                
                cv2.putText(temp_image, "3", (x2, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._color_temp, 2)
                cv2.circle(temp_image, (x2, y2), 5, self._color_temp, -1)
                
                cv2.putText(temp_image, "4", (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._color_temp, 2)
                cv2.circle(temp_image, (x1, y2), 5, self._color_temp, -1)

            final_image = self._draw_rois(temp_image)
            cv2.imshow(self.map_name, final_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and not self._drawing_roi:
                # Exit without saving changes
                self._save_roi_data(self._roi_data_temp)
                self._roi = self._roi_data_temp.copy()
                self._write_image(final_image, os.path.join(self._map_roi_dir, self.map_name+"_roi.png"))
                break
            elif key == ord('s'):
                self._save_roi_data(self._roi_data_temp)
                self._roi = self._roi_data_temp.copy()
                self._write_image(final_image, os.path.join(self._map_roi_dir, self.map_name+"_roi.png"))
                print("Saved ROIs.")
            elif key == ord('r') and self._selected_roi_name:
                del self._roi_data_temp[self._selected_roi_name]
                del self._roi_colors[self._selected_roi_name]
                print(f"Removed ROI: {self._selected_roi_name}")
                self._selected_roi_name = None
                
        cv2.setMouseCallback(self.map_name, lambda *args: None)
        cv2.destroyAllWindows()

    def edit_roi(self):
        """
        Allows the user to edit existing ROIs.
        """
        print("--- Starting ROI Editing Mode ---")
        print("Click and drag a point to move it, or the center to move the whole ROI. Press 'r' to remove, 's' to save, 'q' to quit.")

        self._editing_mode = True
        self._roi_data_temp = self._roi.copy()

        cv2.namedWindow(self.map_name)
        cv2.setMouseCallback(self.map_name, self._mouse_callback)

        while True:
            temp_image = self._source_image.copy()
            final_image = self._draw_rois(temp_image)
            cv2.imshow(self.map_name, final_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Exit without saving changes
                self._save_roi_data(self._roi_data_temp)
                self._roi = self._roi_data_temp.copy()
                self._write_image(final_image, os.path.join(self._map_roi_dir, self.map_name+"_roi.png"))
                break
            elif key == ord('s'):
                self._save_roi_data(self._roi_data_temp)
                self._roi = self._roi_data_temp.copy()
                self._write_image(final_image, os.path.join(self._map_roi_dir, self.map_name+"_roi.png"))
                print("Saved edited ROIs.")
            elif key == ord('r') and self._selected_roi_name:
                del self._roi_data_temp[self._selected_roi_name]
                del self._roi_colors[self._selected_roi_name]
                print(f"Removed ROI: {self._selected_roi_name}")
                self._selected_roi_name = None

        cv2.setMouseCallback(self.map_name, lambda *args: None)
        cv2.destroyAllWindows()

# -----------------------------------------------------------------------------

class Map_ROI(Base_ROI):
    """
    Subclass for map-related functionalities.
    """
    
    def __init__(self, map_name, map_path, scale_factor=1.0):
        roi_dir = r"roi_data/map"
        map_roi_dir = r"ref_image/map/roi"
        map_dir = r"ref_image\map"
        
        super().__init__(
            map_name, 
            map_path,
            map_dir,
            roi_dir,
            map_roi_dir,
            scale_factor
        )
        
        os.makedirs(self._map_roi_dir, exist_ok=True)
        self._source_image = self._load_source()
        self._roi_path = self._get_roi_path()
        self._roi = self._load_roi_data()
        self._roi_data_temp = self._roi.copy()
        
        self._roi_counter = len(self._roi_data_temp)
        for roi_name in self._roi_data_temp.keys():
            self._roi_colors[roi_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
    def _load_source(self):
        """
        Load the map from the specified path and preprocess it.
        """
        print(f"Loading map: {self.map_name} from {self._map_path}")
        map_image = cv2.imread(self._map_path)
        if map_image is None:
            raise FileNotFoundError(f"Map image not found at {self._map_path}")
        
        # return self._preprocess_map(map_image)
        return map_image

    def _get_roi_path(self):
        """
        Generates the path for the ROI JSON file based on the map file.
        """
        map_name = os.path.basename(self._map_path)
        if map_name.endswith(('.png', '.jpg')):
            roi_name = map_name.rsplit('.', 1)[0] + '_roi.json'
        else:
            raise ValueError("Map file must be a PNG or JPG image.")
            
        os.makedirs(self._roi_dir, exist_ok=True)
        return os.path.join(self._roi_dir, roi_name)
    
    def _preprocess_map(self, image):
        """
        Preprocess the map image if needed.
        """
        result_image = cv2.resize(image, None, fx=self._scale_factor, fy=self._scale_factor)
        return result_image

# -----------------------------------------------------------------------------

class Camera_ROI(Base_ROI):
    """
    Subclass for camera-related functionalities, loading a frame from a camera module.
    """
    
    
    def __init__(self, camera_module):
        map_dir = r"ref_image/cam"
        roi_dir = r"roi_data/camera"
        map_roi_dir = r"ref_image/cam/roi"
        self._camera = camera_module
        super().__init__(
            self._camera.camera_id, 
            os.path.join(map_dir, self._camera.camera_id + ".png"),
            map_dir,
            roi_dir,
            map_roi_dir,
            None
        )
        
        os.makedirs(map_roi_dir, exist_ok=True)
        print(self._map_dir)
        self._source_image = self._load_source()
        self._roi_path = self._get_roi_path()
        self._roi = self._load_roi_data()
        self._roi_data_temp = self._roi.copy()
        
        self._roi_counter = len(self._roi_data_temp)
        for roi_name in self._roi_data_temp.keys():
            self._roi_colors[roi_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def _load_source(self):
        """
        Load the first frame from the camera module.
        """
        print(f"Loading map: {self.map_name} from {self._map_path}")
        map_image = cv2.imread(self._map_path)
        if map_image is None:
            print(f"Loading first frame from camera for map: {self.map_name}")
            self._camera.start()
            while self._camera.is_running:
                frame = self._camera.get_frame()
                if frame is not None:
                    map_image = frame
                    self._camera.stop()
                    break
            self._camera.stop()
            print(self._map_dir)
            os.makedirs(self._map_dir, exist_ok=True)
            self._write_image(map_image, self._map_path)
            
            
        # return self._preprocess_map(map_image)
        return map_image
    
    def _get_roi_path(self):
        """
        Generates the path for the ROI JSON file based on the map file.
        """
        roi_name = self.map_name + '_roi.json'
        print(roi_name)
        print(self._roi_dir)
        os.makedirs(self._roi_dir, exist_ok=True)
        return os.path.join(self._roi_dir, roi_name)
    
    def _preprocess_map(self, image):
        """
        Preprocess the map image if needed.
        """
        result_image = cv2.resize(image, None, fx=self._camera.scale_factor, fy=self._camera.scale_factor)
        return result_image
        
        
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Example usage with a map image
    # Create a dummy image file for demonstration
    image_path = r"ref_image\map\S__40026130.png"

    try:
        my_map = Map_ROI("book_plan", image_path, scale_factor=0.8)
        
        # Start in create mode
        my_map.create_roi()
        
        # Now, switch to edit mode (after saving and re-loading)
        my_map.edit_roi()

    except FileNotFoundError as e:
        print(e)
    finally:
        pass
    
    ##########################################################################################
    # # Example usage with a video file camera
    # video_path = r"Vid_test\book_fair.mp4"  # Replace with your video file path
    # camera_instance = VideoFileCamera(camera_id="front_room", video_path=video_path, scale_factor=1.5)
    # try:
    #     my_map = Camera_ROI(camera_instance)
        
    #     # Start in create mode
    #     my_map.create_roi()
        
    #     # Now, switch to edit mode (after saving and re-loading)
    #     my_map.edit_roi()

    # except FileNotFoundError as e:
    #     print(e)
    # finally:
    #     camera_instance.stop()
    
    
    