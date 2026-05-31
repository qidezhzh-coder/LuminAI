#!/usr/bin/env python3
"""
YOLO Client CA4 - HTTP client para yolo_server_CA4 (puerto 5002).
"""
import requests
import cv2
import numpy as np

class YOLOClient:
    def __init__(self, url='http://127.0.0.1:8766', timeout=2.0):
        self.url = url.rstrip('/')
        self.timeout = timeout

    def detect(self, img_bgr):
        """
        Devuelve (label, score) del obstáculo más prominente, o (None, 0.0) si no hay.
        """
        ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None, 0.0
        try:
            resp = requests.post(
                f'{self.url}/detect',
                files={'image': ('frame.jpg', buf.tobytes(), 'image/jpeg')},
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f'[YOLO CLIENT CA4] Error: {e}')
            return None, 0.0

        top = data.get('top')
        if top:
            return top['label'], float(top['score'])
        return None, 0.0
