import time
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
vision = mp.tasks.vision
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

INDEX_TIP = 8
INDEX_PIP = 6
INDEX_MCP = 5


class GestureProcessor:
    def __init__(self, model_path="assets/hand_landmarker.task"):
        model_path = Path(model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo .task no encontrado: {model_path}")

        self._latest_result = None
        self._latest_ts = -1

        def _cb(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self._latest_result = result
            self._latest_ts = int(timestamp_ms)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35,
            result_callback=_cb,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def close(self):
        if self._landmarker:
            self._landmarker.close()

    def process_frame(self, frame_bgr, draw=True):
        # IMPORTANT: monotonic timestamp for detect_async [web:76]
        ts = int(time.monotonic() * 1000)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Async call: may drop frames; output not guaranteed per input [web:76]
        self._landmarker.detect_async(mp_image, ts)

        # allow more latency (Tkinter + OCR can stall)
        RESULT_MAX_AGE_MS = 1200
        age = ts - int(self._latest_ts if self._latest_ts is not None else -1)

        if self._latest_result is None or self._latest_ts < 0 or age > RESULT_MAX_AGE_MS:
            return {"hand_detected": False, "hands": [], "age_ms": int(age)}

        if not self._latest_result.hand_landmarks:
            return {"hand_detected": False, "hands": [], "age_ms": int(age)}

        H, W = frame_bgr.shape[:2]
        handedness_list = self._latest_result.handedness or []

        hands_out = []
        for i, lms in enumerate(self._latest_result.hand_landmarks):
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

            label = "Unknown"
            try:
                label = handedness_list[i][0].category_name
            except Exception:
                pass

            tip = landmarks[INDEX_TIP]
            tip_px = (int(tip[0] * W), int(tip[1] * H))

            # basic pointing heuristic (index extended upward in image coords)
            is_pointing = bool(
                (landmarks[INDEX_TIP, 1] < landmarks[INDEX_PIP, 1]) and
                (landmarks[INDEX_PIP, 1] < landmarks[INDEX_MCP, 1])
            )

            features = self._extract_features(landmarks)

            hands_out.append({
                "handedness": label,
                "landmarks": landmarks,
                "features": features,
                "index_tip_px": tip_px,
                "is_pointing": is_pointing,
            })

            if draw:
                self._draw_landmarks(frame_bgr, landmarks)

        return {"hand_detected": True, "hands": hands_out, "age_ms": int(age)}

    def _draw_landmarks(self, frame_bgr, landmarks_norm):
        h, w = frame_bgr.shape[:2]
        for x, y, _ in landmarks_norm:
            cv2.circle(frame_bgr, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)

    def _extract_features(self, landmarks_norm):
        if landmarks_norm.shape != (21, 3):
            return np.zeros((187,), dtype=np.float32)

        feats = []

        min_xy = landmarks_norm[:, :2].min(axis=0)
        max_xy = landmarks_norm[:, :2].max(axis=0)
        wh = (max_xy - min_xy) + 1e-6

        for x, y, z in landmarks_norm:
            feats.extend([(x - min_xy[0]) / wh[0], (y - min_xy[1]) / wh[1], z])

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        for i, j in connections:
            dx, dy, dz = (landmarks_norm[j] - landmarks_norm[i]).tolist()
            mag = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            feats.extend([dx, dy, dz, mag])

        fingers = [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16],
            [0, 17, 18, 19, 20]
        ]
        for finger in fingers:
            for k in range(4):
                a, b = finger[k], finger[k + 1]
                dx = landmarks_norm[b, 0] - landmarks_norm[a, 0]
                dy = landmarks_norm[b, 1] - landmarks_norm[a, 1]
                feats.extend([
                    float(np.degrees(np.arctan2(dy, dx))),
                    float(np.degrees(np.arctan2(dx, dy)))
                ])

        return np.array(feats, dtype=np.float32)
