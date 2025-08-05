import cv2
import threading
import time
from abc import ABC, abstractmethod
import queue

# --- Configuration Constants (Can be moved to a config file later) ---
BUFFER_SIZE = 5 # Number of frames to buffer for smooth real-time processing
RECONNECT_ATTEMPTS = 5 # How many times to try reconnecting to a CCTV
RECONNECT_DELAY_SEC = 5 # Delay between reconnect attempts

class Camera(ABC):
    """
    Abstract Base Class for all camera types.
    Defines the common interface for camera operations.
    """
    def __init__(self, camera_id: str):
        if not camera_id:
            raise ValueError("Camera ID cannot be empty.")
        self.camera_id = camera_id
        self._is_running = False
        self._frame_buffer = queue.Queue(maxsize=BUFFER_SIZE)
        self._thread = None
        self._latest_frame = None # To store the last read frame for direct access if buffer isn't needed
        self._frame_lock = threading.Lock() # Lock for accessing _latest_frame

    @abstractmethod
    def _open_camera(self):
        """Internal method to open the camera specific to the source."""
        pass

    @abstractmethod
    def _read_frame_internal(self):
        """Internal method to read a frame from the specific source."""
        pass

    @abstractmethod
    def _release_camera(self):
        """Internal method to release the camera specific to the source."""
        pass

    def start(self):
        """Starts the camera frame acquisition in a separate thread."""
        if self._is_running:
            print(f"Camera {self.camera_id} is already running.")
            return

        self._is_running = True
        self._thread = threading.Thread(target=self._run_acquisition, daemon=True)
        self._thread.start()
        print(f"Camera {self.camera_id} acquisition started.")

    def stop(self):
        """Stops the camera frame acquisition thread."""
        if self._is_running:
            self._is_running = False
            if self._thread:
                self._thread.join(timeout=5) # Wait for thread to finish gracefully
                if self._thread.is_alive():
                    print(f"Warning: Camera {self.camera_id} thread did not terminate gracefully.")
            print(f"Camera {self.camera_id} acquisition stopped.")
        else:
            print(f"Camera {self.camera_id} is not running.")


    def _run_acquisition(self):
        """
        Internal method run by the thread to continuously acquire frames.
        """
        cap = self._open_camera()
        if not cap or not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id} initially.")
            self._is_running = False
            return

        print(f"Camera {self.camera_id} successfully opened for acquisition.")
        while self._is_running:
            ret, frame = self._read_frame_internal(cap)
            if ret:
                # Put frame into buffer (non-blocking if buffer is full, drops oldest frame)
                if self._frame_buffer.full():
                    self._frame_buffer.get_nowait() # Discard oldest frame
                self._frame_buffer.put_nowait(frame)

                # Also update latest frame for direct access (e.g., if buffer not used)
                with self._frame_lock:
                    self._latest_frame = frame
            else:
                print(f"Warning: Could not read frame from {self.camera_id}. Attempting to reconnect...")
                self._release_camera(cap) # Release current resource
                cap = self._open_camera() # Try to reopen
                if not cap or not cap.isOpened():
                    print(f"Error: Reconnection failed for {self.camera_id}. Stopping acquisition.")
                    self._is_running = False # Stop if reconnection fails
                    break
                time.sleep(1) # Short delay before next read attempt after reconnection

        self._release_camera(cap) # Ensure camera is released when stopping
        print(f"Camera {self.camera_id} acquisition thread terminated.")


    def get_frame(self):
        """
        Retrieves the latest available frame from the buffer.
        This method is non-blocking. If no frame is available immediately, returns None.
        """
        if not self._is_running:
            print(f"Camera {self.camera_id} is not running. Cannot get frame.")
            return None

        try:
            # Try to get the latest frame from the buffer.
            # If the consumer is slower than the producer, older frames might be skipped.
            # For real-time, this is often desired to always get the freshest data.
            latest_frame = None
            while not self._frame_buffer.empty():
                latest_frame = self._frame_buffer.get_nowait()
            return latest_frame if latest_frame is not None else self._latest_frame
        except queue.Empty:
            # If buffer is empty, fall back to the last explicitly stored frame
            with self._frame_lock:
                return self._latest_frame
        except Exception as e:
            print(f"Error getting frame from {self.camera_id}: {e}")
            return None


    @property
    def is_running(self) -> bool:
        """Returns True if the camera acquisition thread is running."""
        return self._is_running

class CCTVCamera(Camera):
    """
    Handles live CCTV camera streams (e.g., RTSP).
    Implements a reconnection logic for robustness.
    """
    def __init__(self, camera_id: str, rtsp_url: str):
        super().__init__(camera_id)
        if not rtsp_url and rtsp_url != 0:
            raise ValueError("RTSP URL cannot be empty for CCTV camera.")
        self.rtsp_url = rtsp_url
        self._cap = None # OpenCV VideoCapture object

    def _open_camera(self):
        """
        Attempts to open the RTSP stream with reconnect logic.
        """
        for attempt in range(RECONNECT_ATTEMPTS):
            print(f"Camera {self.camera_id}: Attempting to connect to RTSP stream ({attempt + 1}/{RECONNECT_ATTEMPTS})...")
            cap =  cv2.VideoCapture(int(self.rtsp_url)) if self.rtsp_url.isnumeric() else cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)# Use FFMPEG backend for RTSP
            if cap.isOpened():
                self._cap = cap
                print(f"Camera {self.camera_id}: Successfully connected to RTSP stream.")
                return cap
            time.sleep(RECONNECT_DELAY_SEC)
        print(f"Camera {self.camera_id}: Failed to connect to RTSP stream after multiple attempts.")
        return None

    def _read_frame_internal(self, cap):
        """Reads a frame from the RTSP stream."""
        return cap.read()

    def _release_camera(self, cap):
        """Releases the RTSP stream capture object."""
        if cap and cap.isOpened():
            cap.release()
            print(f"Camera {self.camera_id}: RTSP stream released.")

            
class VideoFileCamera(Camera):
    """
    Handles local video files as camera sources.
    Reads the file once and then stops.
    """
    def __init__(self, camera_id: str, video_path: str):
        super().__init__(camera_id)
        if not video_path:
            raise ValueError("Video file path cannot be empty for video file camera.")
        self.video_path = video_path
        self._cap = None # OpenCV VideoCapture object
        self.fps = None # File pointer for the video file

    def _open_camera(self):
        """Opens the video file."""
        print(f"Camera {self.camera_id}: Opening video file: {self.video_path}...")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.camera_id}: {self.video_path}")
            return None
        self._cap = cap
        self.fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second of the video
        print(f"Camera {self.camera_id}: Video file opened.")
        return cap

    def _read_frame_internal(self, cap):
        """
        Reads a frame from the video file.
        This simplified version doesn't handle looping.
        """
        return cap.read()

    def _release_camera(self, cap):
        """Releases the video file capture object."""
        if cap and cap.isOpened():
            cap.release()
            print(f"Camera {self.camera_id}: Video file released.")

    def _run_acquisition(self):
        """
        Overrides the base class method. This version stops gracefully
        when the end of the video file is reached, without attempting to reconnect.
        """
        cap = self._open_camera()
        if not cap or not cap.isOpened():
            print(f"Error: Could not open video file {self.camera_id} initially. Stopping acquisition.")
            self._is_running = False
            return
        
        print(f"Camera {self.camera_id} successfully opened for acquisition.")
        while self._is_running:
            
            if self._frame_buffer.full():
                time.sleep(0.01)
                continue  # Skip if buffer is full to avoid blocking
            else:
                ret, frame = self._read_frame_internal(cap)
                
                if ret:
                    # Put frame into buffer.
                    self._frame_buffer.put_nowait(frame)

                    # Also update latest frame for direct access
                    with self._frame_lock:
                        self._latest_frame = frame
                else:
                    # The video has ended. Stop the thread gracefully.
                    print(f"Info: Video file {self.camera_id} has finished. Stopping acquisition.")
                    self._is_running = False
                    break
                

        # Ensure camera is released when stopping
        self._release_camera(cap)
        print(f"Camera {self.camera_id} acquisition thread terminated.")
        
    def get_fps(self):
        """Returns the frames per second of the video file."""
        if self.fps is not None:
            return self.fps
        else:
            print(f"Warning: FPS not available for camera {self.camera_id}.")
            return None

# --- Example Usage (for testing the module) ---
if __name__ == "__main__":
    video_path = r"C:\Hemoglobin\project\DE\Vid_test\vdo_test_psdetec.mp4"  # Replace with your video file path
    camera = VideoFileCamera(camera_id="test_video", video_path=video_path)
    camera.start()
    while camera.is_running:
        frame = camera.get_frame()
        if frame is not None:
            cv2.imshow("Video Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            continue
    print("Stopping camera...")