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
    Loops the video automatically when it ends.
    """
    def __init__(self, camera_id: str, video_path: str, loop: bool = True):
        super().__init__(camera_id)
        if not video_path:
            raise ValueError("Video file path cannot be empty for video file camera.")
        self.video_path = video_path
        self.loop = loop
        self._cap = None # OpenCV VideoCapture object

    def _open_camera(self):
        """Opens the video file."""
        print(f"Camera {self.camera_id}: Opening video file: {self.video_path}...")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_id}: {self.video_path}")
            return None
        self._cap = cap
        print(f"Camera {self.camera_id}: Video file opened.")
        return cap

    def _read_frame_internal(self, cap):
        """Reads a frame from the video file. Loops if configured."""
        ret, frame = cap.read()
        if not ret and self.loop:
            # If video ends and loop is true, reset to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read() # Read the first frame after reset
            if not ret:
                print(f"Warning: Could not read first frame after looping for {self.camera_id}.")
        return ret, frame

    def _release_camera(self, cap):
        """Releases the video file capture object."""
        if cap and cap.isOpened():
            cap.release()
            print(f"Camera {self.camera_id}: Video file released.")

# --- Example Usage (for testing the module) ---
if __name__ == "__main__":
    print("--- Testing Camera Module ---")

    # Example 1: CCTV Camera (replace with a real RTSP URL if you have one)
    # For testing, you might use a dummy RTSP server or a public stream
    # A common test stream: "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
    # Or your local CCTV URL: "rtsp://user:password@ip_address:port/stream"
    cctv_url = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4" # Replace with your actual CCTV URL
    if cctv_url == "your_cctv_url_here":
        print("\nWARNING: Please replace 'your_cctv_url_here' with an actual RTSP URL to test CCTVCamera.")
        print("Skipping CCTVCamera test for now.")
        cctv_cam = None
    else:
        cctv_cam = CCTVCamera(camera_id="DoorCam1", rtsp_url=cctv_url)
        cctv_cam.start()

    # Example 2: Video File Camera (create a dummy video file or use an existing one)
    # You might need to have a 'test_video.mp4' file in the same directory
    # For a quick test, you can download a sample video or create a small one.
    test_video_path = "test_video.mp4"
    try:
        # Attempt to create a dummy video if it doesn't exist for testing
        cap_dummy = cv2.VideoCapture(0) # Try to open default camera
        if cap_dummy.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(test_video_path, fourcc, 20.0, (640, 480))
            print(f"Creating a dummy video file '{test_video_path}' for testing...")
            for _ in range(50): # Capture 50 frames (approx 2.5 seconds)
                ret, frame = cap_dummy.read()
                if ret:
                    out.write(cv2.resize(frame, (640, 480)))
                else:
                    print("Could not read frame from default camera to create dummy video.")
                    break
            out.release()
            cap_dummy.release()
            print(f"Dummy video '{test_video_path}' created.")
        else:
            print(f"Could not open default camera to create dummy video. Please ensure '{test_video_path}' exists.")
            test_video_path = None # Mark as not available for testing
    except Exception as e:
        print(f"Error creating dummy video: {e}")
        test_video_path = None


    video_cam = None
    if test_video_path and cv2.VideoCapture(test_video_path).isOpened():
        video_cam = VideoFileCamera(camera_id="EventAreaCam1", video_path=test_video_path, loop=True)
        video_cam.start()
    else:
        print("Skipping VideoFileCamera test as no suitable video file found/created.")


    # Demonstrate getting frames
    frames_received_cctv = 0
    frames_received_video = 0
    print("\n--- Getting frames for 10 seconds ---")
    start_time = time.time()
    while time.time() - start_time < 10:
        if cctv_cam and cctv_cam.is_running:
            frame_cctv = cctv_cam.get_frame()
            if frame_cctv is not None:
                # cv2.imshow(f"CCTV {cctv_cam.camera_id}", frame_cctv)
                frames_received_cctv += 1
        
        if video_cam and video_cam.is_running:
            frame_video = video_cam.get_frame()
            if frame_video is not None:
                # cv2.imshow(f"Video {video_cam.camera_id}", frame_video)
                frames_received_video += 1
        
        # Add a small delay to prevent busy-waiting and allow other threads to run
        time.sleep(0.01) # Sleep for 10ms

        # Check for 'q' key to quit imshow windows
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    print(f"\nTotal frames received from CCTV Cam ({cctv_cam.camera_id if cctv_cam else 'N/A'}): {frames_received_cctv}")
    print(f"Total frames received from Video Cam ({video_cam.camera_id if video_cam else 'N/A'}): {frames_received_video}")

    # Cleanup
    print("\n--- Stopping Cameras ---")
    if cctv_cam:
        cctv_cam.stop()
    if video_cam:
        video_cam.stop()

    # cv2.destroyAllWindows()
    print("--- Testing Finished ---")