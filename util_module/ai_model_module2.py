import cv2
import numpy as np
from abc import ABC, abstractmethod
import torch
import os
import time
from typing import Union

# Try to import ultralytics. If not available, provide a mock for demonstration.
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics library not found. YOLOv8HumanDetector will not function.")
    print("Please install it: pip install ultralytics")
    _YOLO_AVAILABLE = False

# --- Configuration Constants ---
MODEL_DIR = "models"
HUMAN_CLASS_ID_YOLO = 0  # class_id 0 = 'person'

class AIModel(ABC):
    def __init__(self, model_path: str, device: str = 'auto', optimize: dict = None):
        if not model_path:
            raise ValueError("Model path cannot be empty.")
        self.model_path = model_path
        self.optimize = optimize if optimize is not None else {}
        self.model = None
        self.device = self._resolve_device(device)
        self._load_model()

        print(f"AIModel initialized: {self.__class__.__name__}, Device: {self.device}, Optimizations: {self.optimize}")

    def _resolve_device(self, device_pref: str) -> str:
        if device_pref == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device_pref == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return 'cpu'
        return device_pref

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        pass

    @abstractmethod
    def _inference(self, preprocessed_input):
        pass

    @abstractmethod
    def _postprocess(self, model_output, original_shape: tuple) -> list:
        pass

    def predict(self, inputs: Union[np.ndarray, list[np.ndarray]]) -> Union[list, list[list]]:
        # Original code – standard batch-compatible prediction
        is_batch = isinstance(inputs, list)
        if not is_batch:
            inputs = [inputs]

        processed_outputs = []
        for img in inputs:
            if img is None:
                processed_outputs.append([])
                continue

            original_shape = img.shape[:2]
            preprocessed_img = self._preprocess(img)

            # ❌ Original code used `.to(self.device)`, removed for numpy-based ultralytics
            # preprocessed_img = preprocessed_img.to(self.device)

            with torch.no_grad():
                if self.optimize.get('half', False) and self.device == 'cuda':
                    pass  # if you convert to tensor, you can do .half()

                model_output = self._inference(preprocessed_img)

            detections = self._postprocess(model_output, original_shape)
            processed_outputs.append(detections)

        return processed_outputs if is_batch else processed_outputs[0]

    def set_device(self, new_device: str):
        resolved_device = self._resolve_device(new_device)
        if resolved_device != self.device:
            print(f"Changing device from {self.device} to {resolved_device} for {self.__class__.__name__}.")
            self.device = resolved_device
            self._load_model()
        else:
            print(f"Device already set to {self.device} for {self.__class__.__name__}. No change needed.")

    def set_optimization(self, optimize_settings: dict):
        print(f"Applying new optimization settings for {self.__class__.__name__}: {optimize_settings}")
        self.optimize.update(optimize_settings)
        self._load_model()

# --- Modified class with tracking ---
class YOLOv8HumanDetector(AIModel):
    def __init__(self, model_name: str = 'yolov8n.pt', device: str = 'auto',
                 optimize: dict = None, confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.7, use_tracking: bool = False):  # ✅ Added use_tracking
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, model_name)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_tracking = use_tracking  # ✅ Save the flag
        super().__init__(model_path, device, optimize)

        if not _YOLO_AVAILABLE:
            raise ImportError("ultralytics library is not installed. Cannot initialize YOLOv8HumanDetector.")

    def _load_model(self):
        print(f"Loading YOLOv8 model from {self.model_path} on {self.device}...")
        try:
            self.model = YOLO(self.model_path)

            if self.optimize.get('half', False) and self.device == 'cuda':
                print("Applying half precision (FP16).")
                self.model.half()

            if self.optimize.get('tensorrt', False) and self.device == 'cuda':
                engine_path = self.model_path.replace('.pt', '.engine')
                if not os.path.exists(engine_path):
                    print(f"Exporting YOLOv8 model to TensorRT engine: {engine_path}")
                    self.model.export(format='engine', device=self.device)
                    self.model = YOLO(engine_path)
                    print("TensorRT engine loaded.")
                else:
                    print(f"TensorRT engine already exists at {engine_path}. Loading it.")
                    self.model = YOLO(engine_path)

            self.model.to(self.device)
            print(f"YOLOv8 model loaded successfully on {self.device}.")

        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            raise

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        return image  # Original code – ultralytics handles preprocessing internally

    def _inference(self, preprocessed_input):
        # ✅ Modified to support tracking
        if self.use_tracking:
            return self.model.track(
                source=preprocessed_input,
                persist=True,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=[HUMAN_CLASS_ID_YOLO],
                verbose=False
            )
        else:
            return self.model.predict(
                source=preprocessed_input,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=[HUMAN_CLASS_ID_YOLO],
                verbose=False,
                device=self.device
            )

    def _postprocess(self, model_output, original_shape: tuple) -> list:
        detections = []
        for result in model_output:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    track_id = int(box.id[0]) if box.id is not None else -1  # ✅ Added tracking ID

                    if class_id == HUMAN_CLASS_ID_YOLO:
                        detections.append({
                            'box': [x1, y1, x2, y2],
                            'score': conf,
                            'class_id': class_id,
                            'class_name': class_name,
                            'person_id': track_id  # ✅ Include in output
                        })
        return detections
