import cv2
import threading
import time
import logging
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CameraManager:
    """
    Manages camera initialization and frame grabbing in a separate background thread
    to maximize frame rate and eliminate input latency.
    """
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[cv2.Mat] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # FPS Calculation for camera stream
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()

    def start(self) -> bool:
        """Initializes the camera and starts the background acquisition thread."""
        logging.info(f"Initializing camera index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            logging.error(f"Failed to open camera index {self.camera_index}")
            return False
            
        # Set frame dimensions
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Verify set dimensions
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"Camera opened. Resolution set to: {actual_w}x{actual_h}")
        
        # Read initial frame
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
        else:
            logging.error("Failed to read initial frame from camera.")
            self.cap.release()
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, name="CameraGrabberThread", daemon=True)
        self.thread.start()
        logging.info("Background camera thread started successfully.")
        return True

    def _update_loop(self) -> None:
        """Target loop for the background thread to continuously fetch frames."""
        self.fps_start_time = time.time()
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.01)
                continue
                
            try:
                ret, frame = self.cap.read()
            except Exception as e:
                logging.error(f"OpenCV error during cap.read(): {e}")
                time.sleep(0.1)
                continue
                
            if ret:
                with self.lock:
                    self.frame = frame
                
                # Calculate FPS
                self.frame_count += 1
                now = time.time()
                elapsed = now - self.fps_start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.fps_start_time = now
            else:
                logging.warning("Camera read failed in background thread.")
                time.sleep(0.01)
                
            # Small sleep to prevent CPU hogging
            time.sleep(0.001)

    def get_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Returns the latest captured frame in a thread-safe manner."""
        with self.lock:
            if self.frame is None:
                return False, None
            # Return a copy to avoid race conditions if the caller modifies it
            return True, self.frame.copy()

    def get_camera_fps(self) -> float:
        """Returns the actual frame rate of the camera stream."""
        return self.fps

    def stop(self) -> None:
        """Stops the background thread and releases camera resources."""
        logging.info("Stopping camera manager...")
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.frame = None
        logging.info("Camera manager stopped and released.")

if __name__ == "__main__":
    # Quick standalone camera test
    cam = CameraManager()
    if cam.start():
        try:
            for _ in range(50):
                ret, frame = cam.get_frame()
                if ret and frame is not None:
                    print(f"Captured frame. Shape: {frame.shape}, Stream FPS: {cam.get_camera_fps():.2f}")
                time.sleep(0.1)
        finally:
            cam.stop()
