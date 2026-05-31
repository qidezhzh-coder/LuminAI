import time
import cv2

class CameraManager:
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.cap = None
        self.picam2 = None
        self._use_picamera2 = False

    def open(self):
        # 1) Intento OpenCV / V4L2
        try:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

                ok, frame = self.cap.read()
                if ok and frame is not None:
                    self._use_picamera2 = False
                    return True

            if self.cap:
                self.cap.release()
            self.cap = None
        except Exception:
            self.cap = None

        # 2) Intento Picamera2
        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()

            config = self.picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameRate": float(self.fps)},
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(0.2)  # warm-up
            self._use_picamera2 = True
            print("[CameraManager] Usando Picamera2")
            return True
        except ModuleNotFoundError:
            print("[CameraManager] No se pudo abrir Picamera2: módulo no instalado")
        except Exception as e:
            print(f"[CameraManager] Error al abrir Picamera2: {e}")

        self.picam2 = None
        self._use_picamera2 = False
        return False

    def capture(self):
        if self._use_picamera2:
            if not self.picam2:
                return None
            try:
                rgb = self.picam2.capture_array()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                return bgr
            except Exception:
                return None

        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self.picam2:
            try:
                self.picam2.stop()
            except Exception:
                pass
            self.picam2 = None

        self._use_picamera2 = False

    def is_open(self):
        if self._use_picamera2:
            return self.picam2 is not None
        return self.cap is not None and self.cap.isOpened()
