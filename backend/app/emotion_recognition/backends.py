"""Emotion classifiers. Swappable: each takes aligned face crops, returns probabilities.

Two backends:

* ``opencv-zoo`` (default): the pretrained OpenCV Zoo expression model,
  bundled in ``weights/``. Runs on CPU through OpenCV, no PyTorch needed.
* ``faceemotion``: SeizureAI's ``FaceEmotionModel`` (RGB + optical-flow
  input). The network is defined in the SeizureAI repository but no trained
  weights were supplied; pass a trained checkpoint with ``--checkpoint``.

Both output the same seven classes in the same order, so results are
comparable and the rest of the platform does not care which one ran.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

try:  # OpenCV's DNN prints a harmless target warning on every model load
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
YUNET_PATH = WEIGHTS_DIR / "face_detection_yunet_2023mar.onnx"
FER_PATH = WEIGHTS_DIR / "facial_expression_recognition_mobilefacenet_2022july.onnx"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class EmotionBackend:
    name = "base"
    input_size = 112
    labels = EMOTIONS

    def predict(self, faces_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """faces_bgr: aligned uint8 BGR crops of input_size. Returns (N, 7) probabilities."""
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


class OpenCVZooFER(EmotionBackend):
    """Progressive Teacher (MobileFaceNet) from OpenCV Zoo; 7 classes, RAF-DB trained."""

    name = "opencv-zoo"

    def __init__(self, model_path: Path = FER_PATH, batch_size: int = 32):
        self.model_path = Path(model_path)
        self.net = cv2.dnn.readNet(str(self.model_path))
        self.batch_size = batch_size

    def predict(self, faces_bgr):
        if len(faces_bgr) == 0:
            return np.zeros((0, len(EMOTIONS)), np.float32)
        out = []
        for i in range(0, len(faces_bgr), self.batch_size):
            batch = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in faces_bgr[i:i + self.batch_size]])
            x = (batch.astype(np.float32) / 255.0 - 0.5) / 0.5          # model's own normalisation
            self.net.setInput(np.ascontiguousarray(x.transpose(0, 3, 1, 2)), "data")
            out.append(softmax(self.net.forward(["label"])[0]))
        return np.concatenate(out).astype(np.float32)

    def describe(self):
        return {"backend": self.name,
                "model": "Progressive Teacher (MobileFaceNet), OpenCV Zoo 2022july",
                "trained_on": "RAF-DB (7 basic expressions)",
                "reported_accuracy": "88.27% on RAF-DB (OpenCV Zoo)",
                "weights": self.model_path.name, "weights_sha256": sha256(self.model_path)}


class SeizureAIFaceEmotion(EmotionBackend):
    """SeizureAI's FaceEmotionModel: input = RGB + optical-flow magnitude (4 channels).

    Assumptions, to confirm with whoever trains the weights: RGB scaled to
    [0, 1]; flow magnitude (Farneback, between consecutive aligned crops of
    the same face) scaled by 1/20 and clipped to [0, 1]; class order as in
    ``seizure_multimodal.facial.facial_model.EMOTIONS``.
    """

    name = "faceemotion"

    def __init__(self, checkpoint: Path, input_size: int = 112, device: Optional[str] = None):
        try:
            import torch
            from .seizureai_facial_model import EMOTIONS as SA_EMOTIONS, FaceEmotionModel
        except ImportError as e:
            raise SystemExit("The faceemotion backend needs PyTorch.") from e
        if tuple(SA_EMOTIONS) != EMOTIONS:
            raise SystemExit(f"Class order mismatch: {SA_EMOTIONS} vs {EMOTIONS}")
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise SystemExit(f"FaceEmotionModel checkpoint not found: {self.checkpoint}")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FaceEmotionModel().to(self.device).eval()
        state = torch.load(self.checkpoint, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)   # strict: a mismatched checkpoint must fail loudly
        self.input_size = input_size
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self):
        self._prev_gray = None

    def predict(self, faces_bgr):
        torch = self.torch
        if len(faces_bgr) == 0:
            return np.zeros((0, len(EMOTIONS)), np.float32)
        xs = []
        for f in faces_bgr:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            if self._prev_gray is None:
                mag = np.zeros(gray.shape, np.float32)
            else:
                flow = cv2.calcOpticalFlowFarneback(self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.clip(np.linalg.norm(flow, axis=2) / 20.0, 0, 1).astype(np.float32)
            self._prev_gray = gray
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            xs.append(np.concatenate([rgb, mag[..., None]], axis=2).transpose(2, 0, 1))
        with torch.no_grad():
            logits = self.model(torch.from_numpy(np.stack(xs)).to(self.device)).cpu().numpy()
        return softmax(logits).astype(np.float32)

    def describe(self):
        return {"backend": self.name, "model": "SeizureAI FaceEmotionModel (RGB + optical flow)",
                "weights": self.checkpoint.name, "weights_sha256": sha256(self.checkpoint)}


def make_backend(name: str, checkpoint: Optional[str] = None) -> EmotionBackend:
    if name == "opencv-zoo":
        return OpenCVZooFER()
    if name == "faceemotion":
        if not checkpoint:
            raise SystemExit("--checkpoint is required for the faceemotion backend "
                             "(no trained FaceEmotionModel weights are bundled)")
        return SeizureAIFaceEmotion(Path(checkpoint))
    raise SystemExit(f"unknown backend {name!r}; choose opencv-zoo or faceemotion")
