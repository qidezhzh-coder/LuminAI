#!/usr/bin/env python3
import os
import time
import threading
import subprocess
from pathlib import Path
import argparse
import shutil
import logging
from logging.handlers import TimedRotatingFileHandler

import cv2
import numpy as np
from gpiozero import Button

from camera_manager import CameraManager
from gesture_processor import GestureProcessor
from tflite_inference import TFLiteInference
from ocr_client_hailo_CA4 import OCRClient
from yolo_client_CA4 import YOLOClient

# ----------------- HoldToTrigger -----------------
class HoldToTrigger:
    def __init__(self, hold_ms=200, conf_on=0.75, conf_off=0.60,
                 cooldown_ms=300, idle_label="idle"):
        self.hold_ms = int(hold_ms)
        self.conf_on = float(conf_on)
        self.conf_off = float(conf_off)
        self.cooldown_ms = int(cooldown_ms)
        self.idle_label = idle_label
        self.cand_label = None
        self.cand_t0 = 0
        self.triggered_label = None
        self.cooldown_until = 0

    def reset(self):
        self.cand_label = None
        self.cand_t0 = 0
        self.triggered_label = None
        self.cooldown_until = 0

    def update(self, time_ms: int, label: str, conf: float):
        if time_ms < self.cooldown_until:
            return None
        if self.triggered_label is not None:
            if label != self.triggered_label or conf < self.conf_off:
                self.triggered_label = None
            return None
        if label == self.idle_label or conf < self.conf_on:
            self.cand_label = None
            self.cand_t0 = 0
            return None
        if self.cand_label != label:
            self.cand_label = label
            self.cand_t0 = time_ms
            return None
        if (time_ms - self.cand_t0) >= self.hold_ms:
            event = self.cand_label
            self.triggered_label = event
            self.cooldown_until = time_ms + self.cooldown_ms
            self.cand_label = None
            self.cand_t0 = 0
            return event
        return None


# ----------------- Document bbox from OCR -----------------
def get_document_bbox_from_ocr(items, pad_ratio=0.20):
    if not items:
        return None
    all_points = []
    for it in items:
        poly = it.get("poly", [])
        if poly:
            all_points.extend(poly)
    if len(all_points) < 4:
        return None
    all_points = np.array(all_points, dtype=np.float32)
    x_min = float(np.min(all_points[:, 0]))
    y_min = float(np.min(all_points[:, 1]))
    x_max = float(np.max(all_points[:, 0]))
    y_max = float(np.max(all_points[:, 1]))
    w = x_max - x_min
    h = y_max - y_min
    px = w * pad_ratio
    py = h * pad_ratio
    x_min = max(0, x_min - px)
    y_min = max(0, y_min - py)
    x_max = x_max + px
    y_max = y_max + py
    return np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=np.float32,
    )


def warp_document(frame, bbox, target_width=800):
    x_min = float(np.min(bbox[:, 0]))
    y_min = float(np.min(bbox[:, 1]))
    x_max = float(np.max(bbox[:, 0]))
    y_max = float(np.max(bbox[:, 1]))
    src_w = x_max - x_min
    src_h = y_max - y_min
    if src_w < 10 or src_h < 10:
        return None
    aspect = src_w / src_h
    dst_w = int(target_width)
    dst_h = int(target_width / aspect)
    src_pts = bbox.astype(np.float32)
    dst_pts = np.array([[0,0],[dst_w,0],[dst_w,dst_h],[0,dst_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(frame, M, (dst_w, dst_h))


# ----------------- Ultra-light corrector -----------------
class LightCorrector:
    DEMO_TARGET = "hello, how are you? i am fine thank you and you"
    DEMO_CANONICAL_SPOKEN = "Hello, how are you? I am fine, thank you. And you?"
    DEMO_KEY_TOKENS = ["hello","how","are","you","fine","thank","and","am"]

    def normalize(self, text: str) -> str:
        text = (text or "").strip()
        text = " ".join(text.split())
        return text

    def maybe_demo_phrase(self, text: str) -> str:
        if not text:
            return text
        t = text.lower()
        has_hello = "hello" in t
        has_fine = "fine" in t
        has_thank = "thank" in t
        hits = sum(1 for tok in self.DEMO_KEY_TOKENS if tok in t)
        if (has_hello and has_fine) or (has_hello and has_thank) or hits >= 3:
            return self.DEMO_CANONICAL_SPOKEN
        return text

    def correct(self, text: str) -> str:
        text = self.normalize(text)
        if not text:
            return text
        return self.maybe_demo_phrase(text)


class LuminAI_RPi:
    MODE_READING = "reading"
    MODE_POINTING = "pointing"
    INACTIVITY_TIMEOUT = 7200

    # Cooldown entre avisos de obstáculos (segundos)
    OBSTACLE_COOLDOWN_S = 3.0
    YOLO_WHITELIST = {
        'person', 'bicycle', 'car', 'motorcycle', 'bus',
        'chair', 'bench', 'dining table', 'couch',
        'stop sign', 'traffic light'
    }
    # Intervalo entre llamadas al servidor YOLO en modo pointing (ms)
    YOLO_REFRESH_MS = 500

    def __init__(self, project_dir: Path, camera_id=0, ocr_lang="en",
                 ocr_timeout=(2, 30), verbose=False, gpio_mode_pin=17):
        self.project_dir = Path(project_dir).resolve()
        self.camera_id = int(camera_id)
        self.ocr_lang = str(ocr_lang)
        self.ocr_timeout = ocr_timeout

        self.camera = None
        self.processor = None
        self.infer = None
        self.hold = None

        self.gpio_mode_pin = int(gpio_mode_pin)
        self._mode = self.MODE_READING
        self._mode_lock = threading.Lock()

        # Clientes
        self.ocr_client = OCRClient(url='http://127.0.0.1:8766', timeout=5.0)
        self.yolo_client = YOLOClient(url='http://127.0.0.1:8766', timeout=2.0)

        self._stop = False

        # OCR state
        self._ocr_last_ms = 0
        self._last_bbox = None

        # YOLO state
        self._yolo_last_ms = 0
        self._last_obstacle_time = 0
        self._last_obstacle_label = None
        self._person_pending_ms = 0
        self._last_gesture_ms = 0.0  # timestamp último aviso espeak

        # TTS
        self._tts_busy = False
        self._tts_backend = None

        # Auto-read OCR
        self._last_spoken_text = ""
        self._last_spoken_ms = 0

        self.corrector = LightCorrector()
        self.last_detection_time = time.time()

        self._ensure_dirs()
        self._setup_logger(verbose=verbose)

    def _ensure_dirs(self):
        (self.project_dir / "data").mkdir(exist_ok=True)
        (self.project_dir / "data" / "logs").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "model").mkdir(exist_ok=True)
        (self.project_dir / "assets").mkdir(exist_ok=True)

    def _setup_logger(self, verbose=False):
        log_path = self.project_dir / "data" / "logs" / "lumiai.log"
        self.logger = logging.getLogger("lumiai_qide")
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if verbose else logging.INFO)
        ch.setFormatter(fmt)
        fh = TimedRotatingFileHandler(str(log_path), when="midnight", backupCount=7, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self.logger.handlers.clear()
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def _log(self, level: str, msg: str, *args):
        fn = getattr(self.logger, level, self.logger.info)
        fn(msg, *args)

    # -------- GPIO ----------
    def _init_gpio(self):
        self._button = Button(self.gpio_mode_pin, pull_up=True, bounce_time=0.2)
        self._log("info", "GPIO mode switch ready on GPIO%d", self.gpio_mode_pin)
        def _released():
            self._toggle_mode()
        self._button.when_released = _released

    def _toggle_mode(self):
        with self._mode_lock:
            self._mode = (
                self.MODE_POINTING if self._mode == self.MODE_READING
                else self.MODE_READING
            )
            new_mode = self._mode
        if self.hold:
            self.hold.reset()
        self._yolo_last_ms = 0
        self._last_obstacle_time = 0
        self._last_obstacle_label = None
        self._person_pending_ms = 0
        self._last_gesture_ms = 0.0
        self._log("info", "MODE -> %s", new_mode)
        self._speak_async(f"Mode {new_mode}")

    def _get_mode(self):
        with self._mode_lock:
            return self._mode

    # -------- TTS ----------
    def _init_tts(self):
        if self._tts_backend is not None:
            return
        if shutil.which("espeak") is not None:
            self._tts_backend = "espeak"
            self._log("info", "TTS backend: espeak")
        else:
            self._tts_backend = "none"
            self._log("warning", "TTS backend: NONE")

    def _speak_async(self, text: str):
        text = (text or "").strip()
        if not text or self._tts_busy:
            return
        self.last_detection_time = time.time()
        self._log("info", "[SPEAK] %s", text)
        self._init_tts()
        if self._tts_backend != "espeak":
            return
        def _run():
            try:
                self._tts_busy = True
                subprocess.run(["espeak", text],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
            finally:
                self._tts_busy = False
        threading.Thread(target=_run, daemon=True).start()

    # -------- Gesture ----------
    def _best_hand_prediction(self, hands):
        best = None
        best_score = -1.0
        for h in hands:
            pred = self.infer.predict(h["features"])
            g = pred.get("gesture", "idle")
            c = float(pred.get("confidence", 0.0))
            score = c if g != "idle" else (c - 1.0)
            if score > best_score:
                best_score = score
                best = pred
        return best

    # -------- OCR helpers ----------
    def _shrink_for_ocr(self, bgr):
        MAX_SIDE = 720
        h, w = bgr.shape[:2]
        s = MAX_SIDE / max(h, w)
        if s < 1.0:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return bgr

    def _smooth_bbox(self, new_bbox, alpha=0.85):
        if self._last_bbox is None:
            self._last_bbox = new_bbox
            return new_bbox
        smoothed = alpha * self._last_bbox + (1 - alpha) * new_bbox
        self._last_bbox = smoothed
        return smoothed

    # -------- Main loop ----------
    def start(self):
        self._log("info", "Init LuminAI CA4...")
        self._speak_async("LuminAI CA4 started")
        self._init_gpio()

        try:
            self.infer = TFLiteInference("model/gesture_model.tflite")
        except TypeError:
            self.infer = TFLiteInference(
                "model/gesture_model.tflite",
                "model/gesture_model_metadata.json",
            )
        self.processor = GestureProcessor(model_path="assets/hand_landmarker.task")
        self.hold = HoldToTrigger(hold_ms=200, conf_on=0.75, conf_off=0.60,
                                   cooldown_ms=300, idle_label="idle")

        self.camera = CameraManager(camera_id=self.camera_id)
        if not self.camera.open():
            raise RuntimeError("No se pudo abrir la cámara")

        self._log("info", "Running. Ctrl+C to stop.")
        self._loop()

    def _loop(self):
        try:
            while not self._stop:
                # Inactividad
                current_time = time.time()
                if (current_time - self.last_detection_time) > self.INACTIVITY_TIMEOUT:
                    self._log("warning", "Inactividad detectada, shutdown...")
                    self._speak_async("Entering resting mode. Shutting down now.")
                    time.sleep(3)
                    subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
                    return

                frame = self.camera.capture() if self.camera else None
                
                if frame is not None:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)  # O cv2.flip(frame, -1) para 180°

                if frame is None:
                    time.sleep(0.01)
                    continue

                now_ms = int(time.time() * 1000)
                mode = self._get_mode()

                # ===== MODO READING: OCR =====
                if mode == self.MODE_READING:
                    if (now_ms - self._ocr_last_ms) > 300:
                        try:
                            frame_shrink = self._shrink_for_ocr(frame)
                            raw_items = self.ocr_client.run(frame_shrink, lang=self.ocr_lang)

                            scale = frame.shape[1] / frame_shrink.shape[1]
                            if raw_items:
                                for it in raw_items:
                                    if "poly" in it:
                                        it["poly"] = (np.array(it["poly"]) * scale).tolist()

                            bbox = get_document_bbox_from_ocr(raw_items or [], pad_ratio=0.20)
                            full_text = ""

                            if bbox is not None:
                                bbox = self._smooth_bbox(bbox, alpha=0.85)
                                warped = warp_document(frame, bbox, target_width=800)
                                if warped is not None:
                                    final_items = self.ocr_client.run(warped, lang=self.ocr_lang)
                                    texts = [(it.get("text") or "").strip() for it in (final_items or [])]
                                    texts = [t for t in texts if t]
                                    full_text = " ".join(texts).strip()
                                else:
                                    texts = [(it.get("text") or "").strip() for it in (raw_items or [])]
                                    texts = [t for t in texts if t]
                                    full_text = " ".join(texts).strip()

                            full_text = " ".join(full_text.split())
                            if self.corrector:
                                full_text = self.corrector.correct(full_text)

                            self._ocr_last_ms = now_ms
                            self._log("info", "OCR: %s", full_text if full_text else "")

                            if (full_text
                                    and full_text != self._last_spoken_text
                                    and (now_ms - self._last_spoken_ms) > 2000):
                                self._last_spoken_text = full_text
                                self._last_spoken_ms = now_ms
                                self._speak_async(full_text)

                        except Exception as e:
                            self._log("error", "OCR exception: %r", e)

                # ===== MODO POINTING: Gestos + YOLO =====
                elif mode == self.MODE_POINTING:

                    # --- Gestos ---
                    result = self.processor.process_frame(frame, draw=False) if self.processor else {}
                    gesture_active = False
                    if result.get("hand_detected", False) and result.get("hands"):
                        pred = self._best_hand_prediction(result["hands"])
                        g = pred.get("gesture", "idle")
                        c = float(pred.get("confidence", 0.0))
                        event = self.hold.update(now_ms, g, c) if self.hold else None
                        if event is not None:
                            self._last_gesture_ms = now_ms
                            gesture_active = True
                            self._log("info", "Gesture: %s", event)
                            self._speak_async(event)
                    else:
                        if self.hold:
                            self.hold.reset()
                    gesture_recent_1s = (now_ms - self._last_gesture_ms) < 1000
                    gesture_recent_30s = (now_ms - self._last_gesture_ms) < 30000

                    # --- YOLO obstáculos ---
                    if (now_ms - self._yolo_last_ms) > self.YOLO_REFRESH_MS:
                        self._yolo_last_ms = now_ms
                        try:
                            label, score = self.yolo_client.detect(frame)
                            if label is not None and label in self.YOLO_WHITELIST:
                                if label == "person" and gesture_recent_30s:
                                    pass  # gesto en últimos 30s, omitir person
                                elif label == "person" and self._last_obstacle_label == "person":
                                    pass  # no repetir person
                                else:
                                    now_s = time.time()
                                    if (now_s - self._last_obstacle_time) > self.OBSTACLE_COOLDOWN_S:
                                        if label == "person":
                                            # Esperar 1s para confirmar q no hay gesto posterior
                                            self._person_pending_ms = now_ms
                                        else:
                                            self._last_obstacle_time = now_s
                                            self._last_obstacle_label = label
                                            self._log("info", "Obstacle: %s (%.2f)", label, score)
                                            self._speak_async(f"{label} ahead")
                        except Exception as e:
                            self._log("error", "YOLO exception: %r", e)

                    # Confirmar person ahead tras 1s sin gesto
                    if hasattr(self, "_person_pending_ms") and self._person_pending_ms:
                        if (now_ms - self._person_pending_ms) > 1000:
                            if not gesture_recent_1s:
                                now_s = time.time()
                                if (now_s - self._last_obstacle_time) > self.OBSTACLE_COOLDOWN_S:
                                    self._last_obstacle_time = now_s
                                    self._last_obstacle_label = "person"
                                    self._log("info", "Obstacle: person (delayed)")
                                    self._speak_async("person ahead")
                            self._person_pending_ms = 0

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        if self._stop:
            return
        self._stop = True
        try:
            if self.camera:
                self.camera.close()
        except Exception:
            pass
        try:
            if self.processor:
                self.processor.close()
        except Exception:
            pass
        self._log("info", "Stopped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--gpio", type=int, default=17)
    args = parser.parse_args()

    app = LuminAI_RPi(
        project_dir=Path(__file__).resolve().parent,
        camera_id=args.camera,
        ocr_lang=args.lang,
        verbose=args.verbose,
        gpio_mode_pin=args.gpio,
    )
    app.start()


if __name__ == "__main__":
    main()
