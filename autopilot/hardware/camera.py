"""Camera sources: Pi camera, USB webcam, or a video file. Frames are BGR uint8."""
import cv2

from ..config import CFG


class PiCamera:
    def __init__(self):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        conf = self.cam.create_video_configuration(
            main={"size": (CFG.cam_width, CFG.cam_height), "format": "RGB888"},
            controls={"FrameRate": CFG.cam_fps})
        self.cam.configure(conf)
        self.cam.start()

    def read(self):
        # picamera2 "RGB888" is actually BGR order in memory, which is what OpenCV wants.
        return self.cam.capture_array()

    def close(self):
        self.cam.stop()


class CvCamera:
    """USB webcam (index) or video file (path)."""

    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.cam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.cam_height)
        self.is_file = isinstance(source, str)

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.is_file:           # loop video files
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        if frame.shape[1] != CFG.cam_width:
            frame = cv2.resize(frame, (CFG.cam_width, CFG.cam_height))
        return frame

    def close(self):
        self.cap.release()
