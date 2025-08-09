import cv2
import time
import os
import json
import numpy as np
import random
from abc import ABC, abstractmethod
import sys
sys.path.append(os.getcwd())

from util_module.camera_module import VideoFileCamera

class Base_ROI(ABC):
    """
    Abstract base class for creating and editing Regions of Interest (ROIs).
    All ROI data is stored as normalized coordinates [0.0, 1.0].
    """

    def __init__(self, map_name, scale_factor):
        """
        Initializes the Base_ROI class.
        
        Args:
            map_name (str): The name of the map or camera feed.
        """
        self.map_name = map_name
        
        # Instance state variables for mouse events
        self._drawing_roi = False
        self._editing_mode = False
        self._editing_point = None
        self._selected_roi_name = None
        self._start_point = None
        self._end_point = None
        self._roi_data_temp = {}
        self._roi_colors = {}
        self._roi_counter = 0
        self._color_temp = None
        
        # path
        self._roi_dir = None
        self._map_dir = None
        self._map_roi_dir = None
        self.map_name = map_name 
        self._map_path = None
        self._scale_factor = scale_factor
        
        # Image properties
        self._source_image = None
        self._res = None  # (H, W) tuple in pixels
        self._roi_size_threshold = 0.05  # Normalized minimum size for a valid ROI

    @abstractmethod
    def _load_source(self):
        """
        Abstract method to load the source image (map or camera frame).
        Must return the image and its resolution as a tuple (image, (H, W)).
        """
        pass

    @abstractmethod
    def _get_roi_path(self):
        """
        Abstract method to generate the path for the ROI JSON file.
        """
        pass

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
                    return {"roi":{}}
        else:
            print(f"No ROI data file found at {self._roi_path}. Creating a new one.")
            self._save_roi_data({"roi":{}})
            return {"roi":{}}

    def _save_roi_data(self, roi_data):
        """
        Saves ROI data to the JSON file.
        """
        os.makedirs(os.path.dirname(self._roi_path), exist_ok=True)
        with open(self._roi_path, 'w') as f:
            json.dump(roi_data, f, indent=4)
        print(f"ROI data saved to {self._roi_path}")
    
    def _to_pixel_coords(self, normalized_points):
        """
        Converts normalized coordinates to pixel coordinates.
        Args:
            normalized_points (list/array of [x, y]): Normalized points.
        Returns:
            np.array: Pixel coordinates.
        """
        height, width = self._res
        points = np.array(normalized_points)
        points[:, 0] *= width
        points[:, 1] *= height
        return points.astype(np.int32)
    
    def _draw_rois(self, image):
        """
        Draws all ROIs from _roi_data_temp onto the provided image.
        """
        if self._res is None:
            return image
        
        overlay = image.copy()
        
        for roi_name, points_dict in self._roi_data_temp["roi"].items():
            if roi_name not in self._roi_colors:
                self._roi_colors[roi_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            color = self._roi_colors[roi_name]
            points = self._to_pixel_coords(list(points_dict.values()))
            
            # Draw filled rectangle with alpha
            cv2.fillPoly(overlay, [points], color)
            
            # Draw outline
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
            
            # Draw point numbers
            for i, p in enumerate(points):
                cv2.putText(image, str(i + 1), (p[0], p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
        
        for roi_name, points_dict in self._roi_data_temp["roi"].items():
            points = self._to_pixel_coords(list(points_dict.values()))
            color = self._roi_colors[roi_name]
            center_x = int(np.mean([p[0] for p in points]))
            center_y = int(np.mean([p[1] for p in points]))
            cv2.putText(image, roi_name, (center_x - 30, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image

    def _is_point_near_roi_point(self, x_norm, y_norm, roi_name, radius_norm=0.015):
        """
        Checks if a normalized point (x_norm, y_norm) is near any of the four points of an ROI.
        Returns the index of the point if it's near, otherwise None.
        """
        points_dict = self._roi_data_temp["roi"].get(roi_name, {})
        for i, point in points_dict.items():
            px, py = point
            distance = np.sqrt((px - x_norm) ** 2 + (py - y_norm) ** 2)
            if distance < radius_norm:
                return i
        return None

    def _is_point_in_roi(self, x_norm, y_norm, roi_name):
        """
        Checks if a normalized point (x_norm, y_norm) is inside the ROI rectangle.
        """
        if self._res is None:
            return False
        
        points_dict = self._roi_data_temp["roi"].get(roi_name, {})
        if not points_dict:
            return False
        
        points = self._to_pixel_coords(list(points_dict.values()))
        x_pixel = int(x_norm * self._res[1])
        y_pixel = int(y_norm * self._res[0])
        
        return cv2.pointPolygonTest(points, (x_pixel, y_pixel), False) >= 0

    def _mouse_callback(self, event, x, y, flags, param):
        """
        Unified mouse callback function for both creation and editing.
        """
        # Normalize mouse coordinates
        height, width = self._res
        x_norm = x / width
        y_norm = y / height
        
        if self._editing_mode:
            # --- EDITING MODE LOGIC ---
            if event == cv2.EVENT_LBUTTONDOWN:
                self._selected_roi_name = None
                for roi_name in self._roi_data_temp["roi"]:
                    point_idx = self._is_point_near_roi_point(x_norm, y_norm, roi_name)
                    if point_idx:
                        self._editing_point = (roi_name, point_idx)
                        self._selected_roi_name = roi_name
                        break
                    
                    if self._is_point_in_roi(x_norm, y_norm, roi_name):
                        self._editing_point = (roi_name, 'center')
                        self._selected_roi_name = roi_name
                        self._start_point = (x_norm, y_norm)
                        break
            
            elif event == cv2.EVENT_LBUTTONUP:
                self._editing_point = None
            
            elif event == cv2.EVENT_MOUSEMOVE and self._editing_point:
                roi_name, point_type = self._editing_point
                if point_type == 'center':
                    dx = x_norm - self._start_point[0]
                    dy = y_norm - self._start_point[1]
                    for point_idx in self._roi_data_temp["roi"][roi_name]:
                        px, py = self._roi_data_temp["roi"][roi_name][point_idx]
                        self._roi_data_temp["roi"][roi_name][point_idx] = [px + dx, py + dy]
                    self._start_point = (x_norm, y_norm)
                else:
                    self._roi_data_temp["roi"][roi_name][point_type] = [x_norm, y_norm]

        else:
            # --- CREATION MODE LOGIC ---
            if event == cv2.EVENT_LBUTTONDOWN:
                self._color_temp = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                self._drawing_roi = True
                self._start_point = (x_norm, y_norm)
                self._selected_roi_name = None
                for roi_name in self._roi_data_temp["roi"]:
                    if self._is_point_in_roi(x_norm, y_norm, roi_name):
                        self._selected_roi_name = roi_name
                        break
            
            elif event == cv2.EVENT_LBUTTONUP:
                if self._drawing_roi:
                    self._drawing_roi = False
                    if self._start_point and self._end_point:
                        x1, y1 = self._start_point
                        x2, y2 = self._end_point
                        
                        # Only save if the ROI is above a minimum size
                        if abs(x1 - x2) > self._roi_size_threshold or abs(y1 - y2) > self._roi_size_threshold:
                            self._roi_counter += 1
                            roi_name = input(f"Enter name for new ROI (default: area_{self._roi_counter}): ") or f"area_{self._roi_counter}"
                            self._roi_data_temp["roi"][roi_name] = {
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
                self._end_point = (x_norm, y_norm)

    def _write_image(self, image, path):
        """
        Writes the processed image to a file.
        """
        cv2.imwrite(path, image)
        print(f"Image saved as {path}")

    def create_roi(self):
        """
        Allows the user to create new ROIs with mouse clicks.
        """
        print("--- Starting ROI Creation Mode ---")
        print("Click and drag to create a rectangle. Press 'r' to remove a hovered ROI, 's' to save, 'q' to quit.")
        
        self._editing_mode = False
        self._roi_data_temp = self._roi.copy()
        self._roi_data_temp["res_h_w"] = self._res
        self._roi_counter = len(self._roi_data_temp["roi"])
        
        if self._source_image is None:
            print("Error: Source image not loaded.")
            return

        cv2.namedWindow(self.map_name)
        cv2.setMouseCallback(self.map_name, self._mouse_callback)
        
        while True:
            temp_image = self._source_image.copy()

            if self._drawing_roi and self._start_point and self._end_point:
                x1_norm, y1_norm = self._start_point
                x2_norm, y2_norm = self._end_point
                
                x1, y1 = self._to_pixel_coords([[x1_norm, y1_norm]])[0]
                x2, y2 = self._to_pixel_coords([[x2_norm, y2_norm]])[0]

                overlay = temp_image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), self._color_temp, -1)
                temp_image = cv2.addWeighted(overlay, 0.5, temp_image, 0.5, 0)
                
                cv2.rectangle(temp_image, (x1, y1), (x2, y2), self._color_temp, 2)
                cv2.circle(temp_image, (x1, y1), 5, self._color_temp, -1)
                cv2.circle(temp_image, (x2, y1), 5, self._color_temp, -1)
                cv2.circle(temp_image, (x2, y2), 5, self._color_temp, -1)
                cv2.circle(temp_image, (x1, y2), 5, self._color_temp, -1)

            final_image = self._draw_rois(temp_image)
            cv2.imshow(self.map_name, final_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and not self._drawing_roi:
                self._save_roi_data(self._roi_data_temp)
                self._roi = self._roi_data_temp.copy()
                self._write_image(final_image, os.path.join(self._map_roi_dir, self.map_name+"_roi.png"))
                print("Saved ROIs.")
                break
            elif key == ord('s'):
                self._save_roi_data(self._roi_data_temp)
                self._roi = self._roi_data_temp.copy()
                self._write_image(final_image, os.path.join(self._map_roi_dir, self.map_name+"_roi.png"))
                print("Saved ROIs.")
            elif key == ord('r') and self._selected_roi_name:
                del self._roi_data_temp["roi"][self._selected_roi_name]
                if self._selected_roi_name in self._roi_colors:
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

        if self._source_image is None:
            print("Error: Source image not loaded.")
            return

        cv2.namedWindow(self.map_name)
        cv2.setMouseCallback(self.map_name, self._mouse_callback)

        while True:
            temp_image = self._source_image.copy()
            final_image = self._draw_rois(temp_image)
            cv2.imshow(self.map_name, final_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._save_roi_data(self._roi_data_temp)
                self._roi = self._roi_data_temp.copy()
                self._write_image(final_image, os.path.join(self._map_roi_dir, self.map_name+"_roi.png"))
                print("Saved ROIs.")
                break
            elif key == ord('s'):
                self._save_roi_data(self._roi_data_temp)
                self._roi = self._roi_data_temp.copy()
                self._write_image(final_image, os.path.join(self._map_roi_dir, self.map_name+"_roi.png"))
                print("Saved edited ROIs.")
            elif key == ord('r') and self._selected_roi_name:
                del self._roi_data_temp["roi"][self._selected_roi_name]
                if self._selected_roi_name in self._roi_colors:
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
    
    def __init__(self, map_path, map_name="",  scale_factor=1.0):
        self._scale_factor = scale_factor
        map_name = os.path.basename(map_path).split(".")[0] if map_name == "" else map_name
        
        super().__init__(map_name, scale_factor)
        self._roi_dir = r"roi_data/map"
        self._map_roi_dir = r"ref_image/map/roi"
        self._map_dir = r"ref_image/map"
        self._roi_dir = r"roi_data/map"
        self._map_path = map_path
        
        os.makedirs(self._map_roi_dir, exist_ok=True)
        self._source_image, self._res = self._load_source(self._map_path)
        self._roi_path = self._get_roi_path()
        self._roi = self._load_roi_data()
        self._roi_data_temp = self._roi.copy()
        
        
        self._roi_counter = len(self._roi_data_temp["roi"])
        for roi_name in self._roi_data_temp.keys():
            self._roi_colors[roi_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
    def _load_source(self, path):
        """
        Load the map from the specified path and preprocess it.
        """
        print(f"Loading map: {self.map_name} from {path}")
        map_image = cv2.imread(path)
        if map_image is None:
            raise FileNotFoundError(f"Map image not found at {path}")
        
        resolution = map_image.shape[:2]
        return map_image, resolution

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
    
    def __init__(self, camera_module, cam_name=""):
        
        
        self._camera = camera_module
        cam_name = camera_module.camera_id if not cam_name else cam_name
        
        super().__init__(cam_name, self._camera.scale_factor)
        self._roi_dir = r"roi_data/camera"
        self._map_dir = r"ref_image/cam"
        self._map_path = os.path.join(self._map_dir, cam_name + ".png")
        self._map_roi_dir = r"ref_image/cam/roi"
        os.makedirs(self._map_roi_dir, exist_ok=True)
        
        self._source_image, self._res = self._load_source()
        self._roi_path = self._get_roi_path()
        self._roi = self._load_roi_data()
        self._roi_data_temp = self._roi.copy()
        
        self._roi_counter = len(self._roi_data_temp["roi"])
        print(self._roi_counter)
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
                    print(resolution)
                    self._camera.stop()
                    break
            self._camera.stop()
            os.makedirs(self._map_dir, exist_ok=True)
            self._write_image(map_image, self._map_path)
        resolution = map_image.shape[:2]
            
        return map_image, resolution
    
    def _get_roi_path(self):
        """
        Generates the path for the ROI JSON file based on the map file.
        """
        roi_name = self.map_name + '_roi.json'
        os.makedirs(self._roi_dir, exist_ok=True)
        return os.path.join(self._roi_dir, roi_name)
    
    def _preprocess_map(self, image):
        """
        Preprocess the map image if needed.
        """
        result_image = cv2.resize(image, None, fx=self._scale_factor, fy=self._scale_factor)
        return result_image
    
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Example usage with a map image
    image_path = r"ref_image\map\ganj.png"

    try:
        my_map = Map_ROI(image_path, scale_factor=0.8)
        
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
    # video_path = r"Vid_test\front_room.mp4"  # Replace with your video file path
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