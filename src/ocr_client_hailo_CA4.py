#!/usr/bin/env python3
"""
OCR Client CA4 - HTTP client para hailo_server_CA4 (puerto 8766, endpoint /ocr).
Drop-in replacement de ocr_client_hailo.py sin subprocess.
"""
import requests
import numpy as np
import cv2

class OCRClient:
    def __init__(self, url='http://127.0.0.1:8766', timeout=5.0, **kwargs):
        self.url = url.rstrip('/')
        self.timeout = timeout

    def run(self, img_bgr, lang='en'):
        ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return []
        try:
            resp = requests.post(
                f'{self.url}/ocr',
                files={'image': ('frame.jpg', buf.tobytes(), 'image/jpeg')},
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f'[OCR CLIENT CA4] Error: {e}')
            return []

        out = []
        for it in data.get('items', []):
            poly = np.array(it['poly'], dtype=np.float32).reshape(4, 2)
            out.append({'poly': poly, 'text': it.get('text', ''), 'score': float(it.get('score', 1.0))})
        return out
