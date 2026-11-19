"""SeizureTransformer signal processing, ported unchanged in behaviour from FYP-MCS18.

Source: FYP-MCS18 ``web-app/backend/app/seizure_detection.py``. The steps,
constants and their order match MCS18 exactly so scores agree with the MCS18
platform for the same recording:

1. average reference across the 19 electrodes;
2. per-channel z-score;
3. resample to 256 Hz (FFT resampling, ``scipy.signal.resample``);
4. 60-second windows (15,360 samples), the last zero-padded;
5. per window: 3rd-order Butterworth band-pass 0.5-120 Hz, then notch filters
   at 1 Hz and 60 Hz (Q=30), applied with ``lfilter``;
6. the network outputs a seizure probability for every sample;
7. events: probability > threshold, binary opening and closing with a
   5-sample structure, then events shorter than 2 s removed.

Note: step 5 removes 60 Hz mains noise (US standard), as in MCS18. Malaysian
mains is 50 Hz; that is not changed here so results stay comparable.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import butter, iirnotch, lfilter, resample

FS = 256
WINDOW_SAMPLES = 15360          # 60 s at 256 Hz
DEFAULT_THRESHOLD = 0.25        # MCS18 detect_seizures default
MIN_EVENT_SECONDS = 2.0


def reference_and_normalise(eeg: np.ndarray, sample_rate: float) -> np.ndarray:
    """Steps 1-3: average reference, per-channel z-score, resample to 256 Hz."""

    eeg = np.asarray(eeg, dtype=np.float64)
    eeg = eeg - np.mean(eeg, axis=0, keepdims=True)
    mean = np.mean(eeg, axis=1, keepdims=True)
    std = np.std(eeg, axis=1, keepdims=True)
    std[std == 0] = 1
    eeg = (eeg - mean) / std
    if sample_rate != FS:
        eeg = resample(eeg, int(eeg.shape[1] * FS / sample_rate), axis=1)
    return eeg


def window_count(n_samples: int) -> int:
    if n_samples < WINDOW_SAMPLES:
        return 1
    return 1 + int(np.ceil((n_samples - WINDOW_SAMPLES) / WINDOW_SAMPLES))


_NOTCH_1 = iirnotch(1, Q=30, fs=FS)
_NOTCH_60 = iirnotch(60, Q=30, fs=FS)
_BAND = butter(3, [0.5 / (0.5 * FS), 120 / (0.5 * FS)], btype="band")


def filtered_window(eeg: np.ndarray, index: int) -> np.ndarray:
    """Steps 4-5 for one window, exactly as MCS18's ``OfficialSeizureDataset``."""

    clip = eeg[:, index * WINDOW_SAMPLES:(index + 1) * WINDOW_SAMPLES]
    if clip.shape[1] < WINDOW_SAMPLES:
        clip = np.pad(clip, ((0, 0), (0, WINDOW_SAMPLES - clip.shape[1])), mode="constant")
    clip = lfilter(_BAND[0], _BAND[1], clip, axis=1)
    clip = lfilter(_NOTCH_1[0], _NOTCH_1[1], clip, axis=1)
    clip = lfilter(_NOTCH_60[0], _NOTCH_60[1], clip, axis=1)
    return clip.astype(np.float32)


def remove_short_events(binary: np.ndarray, min_length: float = MIN_EVENT_SECONDS, fs: int = FS) -> np.ndarray:
    """MCS18's loop, kept as written: drop runs shorter than ``min_length`` seconds."""

    min_samples = int(min_length * fs)
    output = binary.copy()
    is_seizure, start_idx = False, 0
    for i in range(len(output)):
        if not is_seizure and output[i] == 1:
            is_seizure, start_idx = True, i
        elif is_seizure and (output[i] == 0 or i == len(output) - 1):
            end_idx = i if output[i] == 0 else i + 1
            if end_idx - start_idx < min_samples:
                output[start_idx:end_idx] = 0
            is_seizure = False
    return output


def event_mask(probability: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Step 7: per-sample seizure mask after MCS18's post-processing."""

    binary = (probability > threshold).astype(int)
    structure = np.ones(5, dtype=bool)
    binary = binary_opening(binary, structure=structure).astype(int)
    binary = binary_closing(binary, structure=structure).astype(int)
    return remove_short_events(binary, MIN_EVENT_SECONDS, FS).astype(bool)


def events_from_mask(mask: np.ndarray, probability: np.ndarray, offset_seconds: float = 0.0) -> list[dict]:
    """Seizure intervals (start, end, mean and max probability) from a mask."""

    changes = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
    return [{
        "start_seconds": round(float(offset_seconds + s / FS), 3),
        "end_seconds": round(float(offset_seconds + e / FS), 3),
        "mean_probability": round(float(np.mean(probability[s:e])), 4),
        "max_probability": round(float(np.max(probability[s:e])), 4),
    } for s, e in zip(starts, ends)]
