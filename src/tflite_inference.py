"""
Inferencia con TensorFlow Lite (prefer tflite_runtime en Raspberry Pi)
"""

import json
import numpy as np

# Preferible en Raspberry: paquete ligero solo con el intérprete.
# Fallback: TensorFlow completo si ya lo tienes instalado.
try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except Exception:
    from tensorflow.lite import Interpreter  # type: ignore


class TFLiteInference:
    def __init__(self, model_path):
        self.model_path = model_path

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Cargar metadata (mismo nombre que generas en training)
        meta_path = model_path.replace(".tflite", "_metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_size = int(self.metadata["feature_size"])
        self.class_names = list(self.metadata["class_names"])
        self.scaler_mean = np.array(self.metadata["scaler_mean"], dtype=np.float32)
        self.scaler_scale = np.array(self.metadata["scaler_scale"], dtype=np.float32)

        # indices de tensores
        self._in_idx = self.input_details[0]["index"]
        self._out_idx = self.output_details[0]["index"]

    def predict(self, features):
        """Predice gesto desde features"""
        x = np.asarray(features, dtype=np.float32).reshape(-1)

        if x.shape[0] != self.feature_size:
            raise ValueError(f"Expected {self.feature_size} features, got {x.shape[0]}")

        # Normalizar (evitar división por 0)
        x = (x - self.scaler_mean) / (self.scaler_scale + 1e-6)

        # TFLite suele esperar batch dimension: (1, feature_size)
        x = x.reshape(1, self.feature_size).astype(np.float32)

        # Inferencia típica: set_tensor -> invoke -> get_tensor
        self.interpreter.set_tensor(self._in_idx, x)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self._out_idx)[0].astype(np.float32)

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        return {
            "gesture": self.class_names[pred_idx],
            "confidence": confidence,
            "probabilities": probs.tolist(),
            "class_names": self.class_names,
        }
