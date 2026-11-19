"""Finding the patient's face in ward video.

Monitoring cameras look down on a patient lying at any angle, so a stock face
detector misses most faces when run once on the upright frame. Each frame is
therefore searched at four rotations, and every detection is mapped back to
the original frame's coordinates, including all five landmarks.

The detector also fires on things that are not the patient's face (feet,
blanket folds, a headboard logo) and on other people. The patient is taken to
be the face position that keeps reappearing within a segment of video: false
hits flicker, the real face keeps coming back.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

ROTATIONS = (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Standard 5-point template for a 112x112 aligned face (right eye, left eye,
# nose tip, right mouth corner, left mouth corner), as used by the bundled
# expression model's own alignment code.
TEMPLATE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


@dataclass(frozen=True, eq=False)   # compared by identity, not by array values
class Detection:
    score: float
    center: np.ndarray        # (2,) original-frame pixels
    size: float               # longer side of the box, original-frame pixels
    landmarks: np.ndarray     # (5, 2) original-frame pixels


def _unrotate(pts: np.ndarray, rot, w: int, h: int) -> np.ndarray:
    """Map Nx2 points from a rotated copy back to the un-rotated (w x h) image."""
    u, v = pts[:, 0], pts[:, 1]
    if rot is None:
        return pts
    if rot == cv2.ROTATE_90_CLOCKWISE:
        return np.stack([v, h - 1 - u], 1)
    if rot == cv2.ROTATE_180:
        return np.stack([w - 1 - u, h - 1 - v], 1)
    return np.stack([w - 1 - v, u], 1)  # 90 counter-clockwise


class FaceFinder:
    """YuNet at several rotations inside a region of interest."""

    def __init__(self, model_path: str | Path, *, score_threshold: float = 0.3,
                 upscale: float = 1.5, rotations: Sequence = ROTATIONS,
                 min_face_px: float = 40.0):
        self.detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold)
        self.upscale = upscale
        self.rotations = tuple(rotations)
        self.min_face_px = min_face_px

    def detect(self, frame: np.ndarray, roi: Optional[Sequence[float]] = None) -> list[Detection]:
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = (0, 0, W, H) if roi is None else (
            int(roi[0] * W), int(roi[1] * H), int(roi[2] * W), int(roi[3] * H))
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return []
        img = cv2.resize(crop, None, fx=self.upscale, fy=self.upscale) if self.upscale != 1 else crop
        h, w = img.shape[:2]
        found: list[Detection] = []
        for rot in self.rotations:
            im = img if rot is None else cv2.rotate(img, rot)
            self.detector.setInputSize((im.shape[1], im.shape[0]))
            _, faces = self.detector.detect(im)
            if faces is None:
                continue
            for f in faces:
                bx, by, bw, bh = f[:4]
                pts = np.vstack([[bx + bw / 2, by + bh / 2], f[4:14].reshape(5, 2)]).astype(np.float32)
                pts = _unrotate(pts, rot, w, h) / self.upscale + (x0, y0)
                size = float(max(bw, bh) / self.upscale)
                if size >= self.min_face_px:
                    found.append(Detection(float(f[-1]), pts[0], size, pts[1:]))
        return merge_duplicates(found, min_dist=0.05 * W)


def merge_duplicates(dets: Iterable[Detection], min_dist: float) -> list[Detection]:
    """The same face is often found at two rotations; keep the higher score."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(np.linalg.norm(d.center - k.center) > min_dist for k in kept):
            kept.append(d)
    return kept


def select_patient(candidates: list[list[Detection]], frame_width: int) -> list[Optional[Detection]]:
    """Pick the patient's face in each frame of one segment.

    1. Anchor: the candidate position with the highest summed confidence of
       detections near it across the segment (persistence beats one strong hit).
    2. Track from the anchor: small moves are always accepted, larger jumps
       only for confident detections, so the track cannot hop onto a hand.
    """
    flat = [d for frame in candidates for d in frame]
    if not flat:
        return [None] * len(candidates)
    P = np.array([d.center for d in flat])
    S = np.array([d.score for d in flat])
    support = [(S * (np.linalg.norm(P - p, axis=1) < 0.06 * frame_width)).sum() for p in P]
    prev = P[int(np.argmax(support))]
    chosen: list[Optional[Detection]] = []
    for frame in candidates:
        ok = [d for d in frame
              if np.linalg.norm(d.center - prev) < 0.06 * frame_width
              or (np.linalg.norm(d.center - prev) < 0.12 * frame_width and d.score >= 0.6)]
        if ok:
            best = max(ok, key=lambda d: d.score - np.linalg.norm(d.center - prev) / (0.1 * frame_width))
            chosen.append(best)
            prev = best.center
        else:
            chosen.append(None)
    return chosen


def align_face(frame: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Similarity-warp the face so its 5 landmarks land on the standard template.

    This also undoes in-plane rotation, so a face lying sideways comes out upright.
    """
    dst = TEMPLATE_112 * (size / 112.0)
    src = landmarks.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:  # robust fit failed: plain least-squares similarity (Umeyama)
        M = _similarity_lstsq(src, dst)
    return cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _similarity_lstsq(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares rotation + uniform scale + translation mapping src onto dst (2x3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d.T @ s)
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    scale = (S * np.diag(D)).sum() / max((s ** 2).sum(), 1e-9)
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]]).astype(np.float32)
