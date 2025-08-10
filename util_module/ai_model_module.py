import cv2
import numpy as np
from abc import ABC, abstractmethod
import torch
import yaml
import os
import time
from typing import Union, List, Dict, Any, Optional

# Try to import ultralytics. If not available, provide a mock for demonstration.
try:
    import ultralytics
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics library not found. YOLOv8HumanDetector will not function.")
    print("Please install it: pip install ultralytics")
    _YOLO_AVAILABLE = False

# --- Configuration Constants ---
# Define a path where models will be stored or downloaded
MODEL_DIR = "models"


class AIModel(ABC):
    """
    Abstract Base Class for all AI models.
    Defines the common interface for model loading, inference, and processing.
    """
    def __init__(self, model_path: str, device: str = 'auto', optimize: dict = None):
        """
        Initializes the AIModel.

        Args:
            model_path (str): Path to the model weights file.
            device (str): Device to run inference on ('cpu', 'cuda', 'auto').
                          'auto' will try 'cuda' if available, else 'cpu'.
            optimize (dict): Dictionary of optimization settings.
                             E.g., {'half': True, 'tensorrt': False}
        """
        if not model_path:
            raise ValueError("Model path cannot be empty.")
        self.model_path = model_path
        self.optimize = optimize if optimize is not None else {}
        self.model = None
        self.device = self._resolve_device(device)
        self._load_model() # Load model during initialization

        print(f"AIModel initialized: {self.__class__.__name__}, Device: {self.device}, Optimizations: {self.optimize}")

    def _resolve_device(self, device_pref: str) -> str:
        """Resolves the actual device to use based on preference and availability."""
        if device_pref == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device_pref == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return 'cpu'
        return device_pref

    @abstractmethod
    def _load_model(self):
        """
        Abstract method to load the specific AI model and apply optimizations.
        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Abstract method to preprocess a single image for model inference.
        This method must be implemented by subclasses.
        Args:
            image (np.ndarray): Input image (HWC, BGR).
        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        pass

    @abstractmethod
    def _inference(self, preprocessed_input: torch.Tensor):
        """
        Abstract method to run inference on preprocessed input.
        This method must be implemented by subclasses.
        Args:
            preprocessed_input (torch.Tensor): Preprocessed image tensor(s).
        Returns:
            Model's raw output.
        """
        pass

    @abstractmethod
    def _postprocess(self, model_output, original_shape: tuple) -> list:
        """
        Abstract method to postprocess model output into a standardized format.
        This method must be implemented by subclasses.
        Args:
            model_output: Raw output from the model.
            original_shape (tuple): (height, width) of the original image.
        Returns:
            list: A list of dictionaries, each representing a detection:
                  [{'box': [x1, y1, x2, y2], 'score': float, 'class_id': int, 'class_name': str}]
        """
        pass

    def predict(self, inputs: Union[np.ndarray, list[np.ndarray]]) -> Union[list, list[list]]:
        """
        Performs inference on one or more images.

        Args:
            inputs (Union[np.ndarray, list[np.ndarray]]): A single image (HWC, BGR)
                                                          or a list of images for batch processing.

        Returns:
            Union[list, list[list]]: If single input, returns a list of detections.
                                     If batch input, returns a list of lists of detections.
        """
        is_batch = isinstance(inputs, list)

        if not is_batch:
            inputs = [inputs] # Treat single image as a batch of one

        processed_outputs = []
        for img in inputs:
            if img is None:
                processed_outputs.append([]) # Append empty list for None input
                continue

            original_shape = img.shape[:2] # (H, W)
            preprocessed_img = self._preprocess(img)
            
            # Move preprocessed image to the correct device
            preprocessed_img = preprocessed_img.to(self.device)

            with torch.no_grad(): # Disable gradient calculation for inference
                if self.optimize.get('half', False) and self.device == 'cuda':
                    preprocessed_img = preprocessed_img.half() # Apply half precision if enabled
                
                model_output = self._inference(preprocessed_img)
            
            detections = self._postprocess(model_output, original_shape)
            processed_outputs.append(detections)

        return processed_outputs if is_batch else processed_outputs[0]

    def set_device(self, new_device: str):
        """Changes the device for the model."""
        resolved_device = self._resolve_device(new_device)
        if resolved_device != self.device:
            print(f"Changing device from {self.device} to {resolved_device} for {self.__class__.__name__}.")
            self.device = resolved_device
            self._load_model() # Reload model on new device
        else:
            print(f"Device already set to {self.device} for {self.__class__.__name__}. No change needed.")

    def set_optimization(self, optimize_settings: dict):
        """Applies new optimization settings and reloads the model."""
        print(f"Applying new optimization settings for {self.__class__.__name__}: {optimize_settings}")
        self.optimize.update(optimize_settings)
        self._load_model() # Reload model with new optimizations

ULTRA_TRACKER_CFG_PATH = os.path.join(
    os.path.dirname(ultralytics.__file__), 'cfg', 'trackers'
)
BYTETRACK_DEFAULT_CFG = os.path.join(ULTRA_TRACKER_CFG_PATH, "bytetrack.yaml")
STRONGSORT_DEFAULT_CFG = os.path.join(ULTRA_TRACKER_CFG_PATH, "botsort.yaml") # ultralytics uses 'botsort' for StrongSORT
CUSTOM_TRACKER_CFG_DIR = "custom_tracker_configs"

os.makedirs(CUSTOM_TRACKER_CFG_DIR, exist_ok=True)

def create_custom_tracker_config(base_config_path: str, params: dict) -> str:
    """
    Loads a base tracker config file, updates it with custom parameters,
    and saves it to a new file.

    Args:
        base_config_path (str): Path to the base tracker YAML file (e.g., 'bytetrack.yaml').
        params (dict): A dictionary of parameters to update.

    Returns:
        str: The path to the newly created custom config file.
    """
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config file not found at: {base_config_path}")

    # Generate a unique filename based on the base config and parameters
    config_name = os.path.basename(base_config_path).split('.')[0]
    param_str = '_'.join([f"{k}_{v}" for k, v in params.items()])
    new_filename = f"{config_name}_custom_{param_str}.yaml"
    new_config_path = os.path.join(CUSTOM_TRACKER_CFG_DIR, new_filename)

    # Check if the file already exists
    if os.path.exists(new_config_path):
        print(f"Using existing custom config file: {new_config_path}")
        return new_config_path

    # Read the default config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update the config with new parameters
    for key, value in params.items():
        if key in config:
            config[key] = value
        else:
            print(f"Warning: Parameter '{key}' not found in base config. It will be added but might not be used.")
            config[key] = value

    # Save the new config
    with open(new_config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    print(f"Custom tracker config saved to: {new_config_path}")
    return new_config_path

HUMAN_CLASS_ID_YOLO = 0 # YOLO models typically assign class_id 0 to 'person'
class YOLOv8HumanDetector(AIModel):
    """
    Concrete implementation of AIModel for YOLOv8 human detection.
    Uses ultralytics YOLO library.
    """
    def __init__(self, model_name: str = 'yolov8n.pt', tracker_config_path:str = None, device: str = 'auto',
                 optimize: dict = None, confidence_threshold: float = 0.5, iou_threshold: float = 0.7, verbose: bool = False):
        """
        Args:
            model_name (str): Name of the YOLOv8 model (e.g., 'yolov8n.pt', 'yolov8s.pt').
                              Will be downloaded if not found locally.
            device (str): Device to run inference on ('cpu', 'cuda', 'auto').
            optimize (dict): Optimization settings.
                             E.g., {'half': True, 'tensorrt': False}
            confidence_threshold (float): Minimum confidence score for a detection to be kept.
            iou_threshold (float): IoU threshold for Non-Maximum Suppression (NMS). Lower values
                                   result in fewer overlapping boxes (more aggressive NMS).
        """
        super().__init__(model_path=model_name, device=device, optimize=optimize)
        
        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, model_name)
        super().__init__(model_path, device, optimize)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold # Store the IoU threshold
        self.is_using_tracker = False
        self.tracker_config_path = tracker_config_path
        self.verbose = verbose # Verbose output for debugging
        
        if self.tracker_config_path is not None:
            self.is_using_tracker = True
        
        if not _YOLO_AVAILABLE:
            raise ImportError("ultralytics library is not installed. Cannot initialize YOLOv8HumanDetector.")

    def _load_model(self):
        """
        Loads the YOLOv8 model using ultralytics and applies optimizations.
        """
        print(f"Loading YOLOv8 model from {self.model_path} on {self.device}...")
        try:
            self.model = YOLO(self.model_path)
            
            # Apply half precision if enabled and on CUDA
            if self.optimize.get('half', False) and self.device == 'cuda':
                print("Applying half precision (FP16).")
                self.model.half() # Convert model to FP16
            
            # Export to TensorRT if enabled
            if self.optimize.get('tensorrt', False) and self.device == 'cuda':
                # Check if TensorRT engine already exists
                engine_path = self.model_path.replace('.pt', '.engine')
                if not os.path.exists(engine_path):
                    print(f"Exporting YOLOv8 model to TensorRT engine: {engine_path}")
                    # The export method handles the device automatically
                    self.model.export(format='engine', device=self.device)
                    self.model = YOLO(engine_path) # Load the exported engine
                    print("TensorRT engine loaded.")
                else:
                    print(f"TensorRT engine already exists at {engine_path}. Loading it.")
                    self.model = YOLO(engine_path)
            
            # Move model to the resolved device (ultralytics handles this well)
            self.model.to(self.device)
            print(f"YOLOv8 model loaded successfully on {self.device}.")

        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            raise
        
    def predict(self, inputs: Union[np.ndarray, list[np.ndarray]]) -> Union[list, list[list]]:
        """
        Performs inference on one or more images.

        Args:
            inputs (Union[np.ndarray, list[np.ndarray]]): A single image (HWC, BGR)
                                                          or a list of images for batch processing.

        Returns:
            Union[list, list[list]]: If single input, returns a list of detections.
                                     If batch input, returns a list of lists of detections.
        """
        is_batch = isinstance(inputs, list)

        if not is_batch:
            inputs = [inputs] # Treat single image as a batch of one

        processed_outputs = []
        for img in inputs:
            if img is None:
                processed_outputs.append([]) # Append empty list for None input
                continue

            original_shape = img.shape[:2] # (H, W)
            preprocessed_img = self._preprocess(img)
            
            # Move preprocessed image to the correct device
            preprocessed_img = preprocessed_img

            with torch.no_grad(): # Disable gradient calculation for inference
                if self.optimize.get('half', False) and self.device == 'cuda':
                    preprocessed_img = preprocessed_img.half() # Apply half precision if enabled
                
                model_output = self._inference(preprocessed_img)
            
            detections = self._postprocess(model_output, original_shape)
            processed_outputs.append(detections)

        return processed_outputs if is_batch else processed_outputs[0]

    def _preprocess(self, image: np.ndarray) -> np.ndarray: # Changed return type to np.ndarray
        """
        YOLOv8's `predict` method handles internal preprocessing,
        so this method can be a passthrough or minimal.
        """
        # ultralytics YOLO.predict expects a numpy array or list of arrays.
        # It handles resizing, normalization, and channel ordering internally.
        # So, we just return the original image.
        return image

    def _inference(self, preprocessed_input):
        """
        Runs inference using the YOLOv8 model.
        """
        # Pass the iou_threshold directly to the predict method
        if self.is_using_tracker:
            results = self.model.track(
            source=preprocessed_input,
            conf=self.confidence_threshold,
            iou=self.iou_threshold, # NMS IoU threshold added here
            persist=True, # Persist tracks between frames
            tracker=self.tracker_config_path,
            verbose=self.verbose,
            classes=[HUMAN_CLASS_ID_YOLO], # Track 'person' class (class ID 0)
            device=self.device
        )
        else:
            results = self.model.predict(
            source=preprocessed_input,
            conf=self.confidence_threshold,
            iou=self.iou_threshold, # NMS IoU threshold added here
            classes=[HUMAN_CLASS_ID_YOLO], # Filter for human class (0)
            verbose=self.verbose, # Suppress verbose output
            device=self.device # Ensure inference runs on the correct device
        )
            
        return results

    def _postprocess(self, model_output, original_shape: tuple) -> list:
        """
        Postprocesses YOLOv8 model output to a standardized detection format.
        """
        detections = []
        # model_output is a list of Results objects (one per image in batch)
        for result in model_output:
            # Each result object contains boxes, masks, keypoints, etc.
            # We are interested in result.boxes for detection.
            if result.boxes:
                    if self.is_using_tracker:
                        # if result.boxes.id is not None:
                        for i in range(len(result.boxes)):
                            track_id = result.boxes.id[i] if result.boxes.id is not None else -1
                            box = result.boxes[i]
                            x1, y1, x2, y2 = box.xyxyn[0].tolist()
                            conf = float(box.conf[0])
                            class_id = int(box.cls[0])
                            class_name = self.model.names[class_id]

                            detections.append({
                                'box': [x1, y1, x2, y2],
                                'score': float(conf),
                                'class_id': class_id,
                                'class_name': class_name,
                                'track_id': int(track_id)
                            })
                    else: 
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxyn[0].tolist()
                            conf = float(box.conf[0])
                            class_id = int(box.cls[0])
                            class_name = self.model.names[class_id] # Get class name from model

                            # Note: Confidence and class filtering are already handled by model.predict
                            # when `conf` and `classes` arguments are passed.
                            # We can still add an explicit check here for clarity or if pre-filtering
                            # logic changes in future versions of ultralytics.
                            if class_id == HUMAN_CLASS_ID_YOLO: # We only process human detections at this point
                                detections.append({
                                    'box': [x1, y1, x2, y2],
                                    'score': conf,
                                    'class_id': class_id,
                                    'class_name': class_name
                                })   
        return detections

# --- First Example Usage (for testing the module) ---
if __name__ == "__main__":
    model = YOLOv8HumanDetector(
        model_name='yolo12x.pt',
        iou_threshold=0.8,
        confidence_threshold=0.01,
        tracker_config_path=STRONGSORT_DEFAULT_CFG, 
        # tracker_config_path = BYTETRACK_DEFAULT_CFG,
    )

    # Load an example image (replace with your own image path)
    cap = cv2.VideoCapture(r"C:\Hemoglobin\project\DE\Vid_test\ิbook_fair.mp4")
    # cap = cv2.VideoCapture(r"C:\Hemoglobin\project\DE\Vid_test\vdo_test_psdetec.mp4")
    ret, frame = cap.read()
    height, width,  = frame.shape[:2]
    
    scale_factor = 1.5
    scaled_height, scaled_width, = int(height * scale_factor), int(width * scale_factor)


    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of stream or camera disconnected.")
            break
        # Resize frame to match the model's expected input size
        frame = cv2.resize(frame, (scaled_width, scaled_height))
        
        frame = cv2.bilateralFilter(
                                frame, 
                                d=9, 
                                sigmaColor=75, 
                                sigmaSpace=75
                            )
        
        # Update with StrongSORT
        result = model.predict(frame.copy())
        # print(result)
        
        # --- Visualization ---
        frame_strongsort = frame.copy()
        for track in result:
            x1, y1, x2, y2 = track['box']
            track_id = track['track_id']
            # Draw bounding box and track ID
            x1 = x1 * scaled_width
            y1 = y1 * scaled_height
            x2 = x2 * scaled_width
            y2 = y2 * scaled_height
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame_strongsort, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame_strongsort, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("StrongSORT (Custom Config)", frame_strongsort)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()