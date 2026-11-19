"""SeizureTransformer (from FYP-MCS18) as MDS01's EEG seizure model.

SeizureTransformer scores every sample of 19 single-electrode EEG, while
MDS01 stores one score per 4-second window (2-second stride) on its bipolar
model grid. This adapter keeps MDS01's grid, so the timeline, reports,
retained clips and database schema are unchanged:

* ``probability`` of a window = the highest per-sample probability inside it;
* ``seizure_detected`` = the window overlaps a seizure event that survives
  MCS18's post-processing (opening/closing, events of at least 2 s).

So a window can show a score above the threshold without a flag when the
crossing was shorter than MCS18's minimum event length.

The weights file is not in Git. If ``SEIZURE_TRANSFORMER_SHA256`` is set, the
file must match it; either way its SHA-256 prefix is recorded as the model
version on every prediction.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from backend.app.ml.interface import WindowPrediction
from backend.app.ml.seizure_transformer import pipeline


class SeizureTransformerError(RuntimeError):
    """Raised when the SeizureTransformer runtime cannot be loaded or used safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SeizureTransformerService:
    """Score 19-electrode EEG with SeizureTransformer on MDS01's window grid."""

    model_name = "seizure-transformer-mcs18"
    score_type = "uncalibrated_probability"
    calibration_method: str | None = None
    # processing_service reads source electrodes instead of bipolar windows.
    requires_electrode_input = True

    def __init__(
        self,
        model_path: Path,
        expected_sha256: str = "",
        *,
        threshold: float = pipeline.DEFAULT_THRESHOLD,
        batch_size: int = 8,
        device: str | None = None,
        network=None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise SeizureTransformerError("PyTorch is required for MODEL_RUNTIME=seizure_transformer.") from exc
        self._torch = torch
        self.threshold = float(threshold)
        self.batch_size = int(batch_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if network is not None:          # injected for tests: no weights file involved
            self._network = network.to(self.device).eval()
            self.model_version = "injected-network"
            return
        path = Path(model_path)
        # A zero-byte file is the image's bind-mount placeholder, not weights.
        if not path.is_file() or path.stat().st_size == 0:
            raise SeizureTransformerError(
                f"SeizureTransformer weights not found at {path}. Get "
                "official_seizure_transformer_model.pth from the MCS18 team."
            )
        actual = _sha256_file(path)
        if expected_sha256 and actual != expected_sha256.lower():
            raise SeizureTransformerError("SeizureTransformer weights do not match SEIZURE_TRANSFORMER_SHA256.")
        from backend.app.ml.seizure_transformer.model import SeizureTransformer

        network = SeizureTransformer(in_channels=19, in_samples=pipeline.WINDOW_SAMPLES,
                                     dim_feedforward=2048, num_layers=8, num_heads=4)
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            network.load_state_dict(state, strict=True)
        except Exception as exc:
            raise SeizureTransformerError("SeizureTransformer weights could not be loaded.") from exc
        # Named _network, not model: MDS01's SHAP explainer targets Keras models
        # through ``inference.model`` and must skip this runtime.
        self._network = network.to(self.device).eval()
        self.model_version = f"sha256:{actual[:12]}"

    def predict(self, windows, window_starts, record_id, privacy_method="metadata-scrub"):
        raise SeizureTransformerError(
            "SeizureTransformer reads source electrodes; call predict_electrodes()."
        )

    def segment_probability(self, eeg: np.ndarray, sample_rate: float) -> np.ndarray:
        """Per-sample seizure probability at 256 Hz for one contiguous segment."""

        torch = self._torch
        eeg = pipeline.reference_and_normalise(eeg, sample_rate)
        n = eeg.shape[1]
        count = pipeline.window_count(n)
        outputs = []
        with torch.no_grad():
            for first in range(0, count, self.batch_size):
                batch = np.stack([pipeline.filtered_window(eeg, i)
                                  for i in range(first, min(first + self.batch_size, count))])
                out = self._network(torch.from_numpy(batch).float().to(self.device))
                outputs.append(out.cpu().numpy())
        return np.concatenate(outputs, axis=0).flatten()[:n].astype(np.float64)

    def predict_electrodes(
        self,
        segments: Sequence[tuple[float, np.ndarray]],
        sample_rate: float,
        window_starts: np.ndarray,
        window_seconds: float = 4.0,
    ) -> list[WindowPrediction]:
        """Return one prediction per MDS01 window start (source seconds)."""

        scored = []
        for start_seconds, eeg in segments:
            if eeg.ndim != 2 or eeg.shape[0] != 19 or eeg.shape[1] == 0:
                raise SeizureTransformerError("SeizureTransformer input must be (19, samples).")
            prob = self.segment_probability(eeg, sample_rate)
            scored.append((float(start_seconds), prob, pipeline.event_mask(prob, self.threshold)))
        self.last_events = [event for start, prob, mask in scored
                            for event in pipeline.events_from_mask(mask, prob, start)]

        span = int(round(window_seconds * pipeline.FS))
        tolerance = 1.0 / pipeline.FS
        predictions = []
        for index, start in enumerate(np.asarray(window_starts, dtype=np.float64)):
            for seg_start, prob, mask in scored:
                seg_end = seg_start + len(prob) / pipeline.FS
                if seg_start - tolerance <= start and start + window_seconds <= seg_end + tolerance:
                    i0 = max(0, int(round((start - seg_start) * pipeline.FS)))
                    i1 = min(len(prob), i0 + span)
                    score = float(np.max(prob[i0:i1]))
                    predictions.append(WindowPrediction(
                        window_index=index,
                        start_seconds=float(start),
                        end_seconds=float(start + window_seconds),
                        probability=score,
                        raw_score=score,
                        seizure_detected=bool(mask[i0:i1].any()),
                        score_type=self.score_type,
                    ))
                    break
            else:
                raise SeizureTransformerError(
                    f"Model window at {start:.2f} s is outside the electrode recording."
                )
        return predictions
