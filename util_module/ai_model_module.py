import cv2
import numpy as np
from abc import ABC, abstractmethod
import torch
import os
import time
from typing import Union, List

# Try to import ultralytics. If not available, provide a mock for demonstration.
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics library not found. YOLOv8HumanDetector will not function.")
    print("Please install it: pip install ultralytics")
    _YOLO_AVAILABLE = False

# --- Configuration Constants ---
# Define a path where models will be stored or downloaded
MODEL_DIR = "models"
HUMAN_CLASS_ID_YOLO = 0 # YOLO models typically assign class_id 0 to 'person'

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

class YOLOv8HumanDetector(AIModel):
    """
    Concrete implementation of AIModel for YOLOv8 human detection.
    Uses ultralytics YOLO library.
    """
    def __init__(self, model_name: str = 'yolov8n.pt', device: str = 'auto',
                 optimize: dict = None, confidence_threshold: float = 0.5, iou_threshold: float = 0.7):
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
        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, model_name)
        super().__init__(model_path, device, optimize)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold # Store the IoU threshold
        
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
        results = self.model.predict(
            source=preprocessed_input,
            conf=self.confidence_threshold,
            iou=self.iou_threshold, # NMS IoU threshold added here
            classes=[HUMAN_CLASS_ID_YOLO], # Filter for human class (0)
            verbose=False, # Suppress verbose output
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
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxyn[0].tolist())
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
    print("--- Testing AI Model Module with NMS Adjustment ---")

    # Create a dummy image for testing
    dummy_image = np.zeros((640, 480, 3), dtype=np.uint8) # Black image 640x480
    cv2.putText(dummy_image, "Test Image", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Simulate a human-like detection area (just for visual reference, model won't detect this as human)
    cv2.rectangle(dummy_image, (100, 100), (200, 300), (255, 0, 0), 2)

    # Test YOLOv8HumanDetector with default NMS
    print("\n--- Testing YOLOv8HumanDetector (Default NMS: conf=0.25, iou=0.7) ---")
    try:
        yolo_detector_default_nms = YOLOv8HumanDetector(model_name='yolov8n.pt', device='cpu', 
                                                       confidence_threshold=0.25, iou_threshold=0.7)
        detections_default = yolo_detector_default_nms.predict(dummy_image)
        print(f"Detections (default NMS): {detections_default}")

        # Test YOLOv8HumanDetector with adjusted NMS (more aggressive suppression)
        print("\n--- Testing YOLOv8HumanDetector (Aggressive NMS: conf=0.25, iou=0.4) ---")
        yolo_detector_aggressive_nms = YOLOv8HumanDetector(model_name='yolov8n.pt', device='cpu', 
                                                          confidence_threshold=0.25, iou_threshold=0.4) # Lower IoU
        detections_aggressive = yolo_detector_aggressive_nms.predict(dummy_image)
        print(f"Detections (aggressive NMS): {detections_aggressive}")

        # Test YOLOv8HumanDetector with less aggressive NMS
        print("\n--- Testing YOLOv8HumanDetector (Less Aggressive NMS: conf=0.25, iou=0.9) ---")
        yolo_detector_less_aggressive_nms = YOLOv8HumanDetector(model_name='yolov8n.pt', device='cpu', 
                                                               confidence_threshold=0.25, iou_threshold=0.9) # Higher IoU
        detections_less_aggressive = yolo_detector_less_aggressive_nms.predict(dummy_image)
        print(f"Detections (less aggressive NMS): {detections_less_aggressive}")

        if torch.cuda.is_available():
            print("\n--- Testing YOLOv8HumanDetector (CUDA + Adjusted NMS) ---")
            yolo_detector_cuda_nms = YOLOv8HumanDetector(model_name='yolov8n.pt', device='cuda', 
                                                        optimize={'half': True, 'tensorrt': True},
                                                        confidence_threshold=0.25, iou_threshold=0.6)
            detections_cuda_nms = yolo_detector_cuda_nms.predict(dummy_image)
            print(f"Detections (CUDA + Adjusted NMS): {detections_cuda_nms}")
        else:
            print("\nCUDA not available. Skipping CUDA tests.")

    except ImportError as ie:
        print(f"\nSkipping YOLOv8 tests due to missing ultralytics library: {ie}")
    except Exception as e:
        print(f"\nAn error occurred during YOLOv8 testing: {e}")

    print("\n--- Testing Finished ---")
    
# --- Second Example Usage (for testing the module) ---
if __name__ == "__main__":
    print("--- Testing AI Model Module ---")

    # Create a dummy image for testing
    dummy_image = np.zeros((640, 480, 3), dtype=np.uint8) # Black image 640x480
    cv2.putText(dummy_image, "Test Image", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Simulate a human-like detection area (just for visual reference, model won't detect this as human)
    cv2.rectangle(dummy_image, (100, 100), (200, 300), (255, 0, 0), 2)

    # Test YOLOv8HumanDetector
    print("\n--- Testing YOLOv8HumanDetector (CPU) ---")
    try:
        yolo_detector_cpu = YOLOv8HumanDetector(model_name='yolov8n.pt', device='cpu', confidence_threshold=0.25)
        
        # Single image prediction
        print("Predicting on a single image (CPU)...")
        detections_single = yolo_detector_cpu.predict(dummy_image)
        print(f"Detections (single image, CPU): {detections_single}")

        # Batch prediction
        print("\nPredicting on a batch of images (CPU)...")
        batch_images = [dummy_image, dummy_image.copy()]
        detections_batch = yolo_detector_cpu.predict(batch_images)
        print(f"Detections (batch, CPU): {detections_batch}")

        # Test changing device (if CUDA is available)
        if torch.cuda.is_available():
            print("\n--- Testing YOLOv8HumanDetector (CUDA) ---")
            yolo_detector_cuda = YOLOv8HumanDetector(model_name='yolov8n.pt', device='cuda', confidence_threshold=0.25)
            detections_cuda = yolo_detector_cuda.predict(dummy_image)
            print(f"Detections (single image, CUDA): {detections_cuda}")

            # Test half precision
            print("\n--- Testing YOLOv8HumanDetector (CUDA + Half Precision) ---")
            yolo_detector_half = YOLOv8HumanDetector(model_name='yolov8n.pt', device='cuda', optimize={'half': True}, confidence_threshold=0.25)
            detections_half = yolo_detector_half.predict(dummy_image)
            print(f"Detections (single image, CUDA + Half): {detections_half}")

            # Test TensorRT export and inference
            print("\n--- Testing YOLOv8HumanDetector (CUDA + TensorRT) ---")
            # Note: TensorRT export can take a while the first time.
            # It will save a .engine file next to the .pt file.
            yolo_detector_trt = YOLOv8HumanDetector(model_name='yolov8n.pt', device='cuda', optimize={'tensorrt': True}, confidence_threshold=0.25)
            detections_trt = yolo_detector_trt.predict(dummy_image)
            print(f"Detections (single image, CUDA + TensorRT): {detections_trt}")

            # Test changing device after init
            print("\n--- Testing changing device ---")
            yolo_detector_cpu.set_device('cuda') # Should reload model on GPU
            detections_changed_device = yolo_detector_cpu.predict(dummy_image)
            print(f"Detections (after changing device to CUDA): {detections_changed_device}")

            print("\n--- Testing changing optimization ---")
            yolo_detector_cuda.set_optimization({'half': True}) # Should reload model with half precision
            detections_changed_opt = yolo_detector_cuda.predict(dummy_image)
            print(f"Detections (after changing opt to Half): {detections_changed_opt}")

        else:
            print("\nCUDA not available. Skipping CUDA/Half/TensorRT tests.")

    except ImportError as ie:
        print(f"\nSkipping YOLOv8 tests due to missing ultralytics library: {ie}")
    except Exception as e:
        print(f"\nAn error occurred during YOLOv8 testing: {e}")

    print("\n--- Testing Finished ---")