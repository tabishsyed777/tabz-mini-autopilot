"""Runs the trained model on the car using ONNX Runtime (fast on a Raspberry Pi)."""
import numpy as np

from .config import CFG
from .perception import preprocess


class NeuralPilot:
    def __init__(self, path=None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(path or CFG.model_path), opts,
                                         providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def predict(self, frame_bgr) -> float:
        x = preprocess(frame_bgr)[None]
        return float(self.sess.run(None, {self.inp: x})[0].reshape(-1)[0])
