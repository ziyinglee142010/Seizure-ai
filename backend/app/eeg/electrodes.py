"""Single-electrode (referential) EEG access for models that need source electrodes.

MDS01's reviewed model contract uses 18 bipolar channels (electrode pairs such as
``FP1-F7``). SeizureTransformer instead reads 19 single 10-20 electrodes. Pairs
cannot be turned back into single electrodes, so this module reads the
electrodes from the recording itself:

* EDF and Nicolet ``.data`` files whose channels are single electrodes
  (``FP1``, ``EEG FP1-REF``, ``Fp1-LE`` and similar spellings);
* legacy Nicolet ``.e`` files, which store single electrodes and from which
  MDS01 derives its bipolar montage.

Files that contain only bipolar pairs raise :class:`ElectrodeInputUnavailable`
with a message the user can act on. Only signal values leave this module; no
header metadata is returned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.app.eeg.contracts import MODEL_CHANNELS

# Order expected by SeizureTransformer (MCS18 ``TARGET_CHANNELS``), in modern
# 10-20 names: T3=T7, T5=P7, T4=T8, T6=P8.
SEIZURE_TRANSFORMER_ELECTRODES = (
    "FP1", "F3", "C3", "P3", "O1",
    "F7", "T7", "P7",
    "FZ", "CZ", "PZ",
    "FP2", "F4", "C4", "P4", "O2",
    "F8", "T8", "P8",
)
_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
_REFERENCE_SUFFIXES = ("-REF", "-LE", "-AVG", "-AV", "-A1", "-A2", "-M1", "-M2", "-CAR")


class ModelInputUnavailable(ValueError):
    """The recording cannot be scored by the configured model.

    Messages contain only technical facts (electrode names, unsupported
    options), never recording metadata, so they are safe to show users.
    """


class ElectrodeInputUnavailable(ModelInputUnavailable):
    """The recording does not contain the 19 single electrodes the model needs."""


def canonical_electrode(label: str) -> str | None:
    """Return the 10-20 electrode name for a single-electrode label, else ``None``.

    Bipolar pairs (``FP1-F7``) return ``None``; reference suffixes such as
    ``-REF`` or ``-LE`` are treated as single electrodes.
    """

    text = str(label).strip().upper().replace(" ", "")
    if text.startswith("EEG"):
        text = text[3:]
    for suffix in _REFERENCE_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    if "-" in text or not text:
        return None
    return _ALIASES.get(text, text)


def electrode_index(labels: list[str]) -> dict[str, int]:
    """Map each needed electrode to its channel index; ambiguous labels fail closed."""

    found: dict[str, int] = {}
    for index, label in enumerate(labels):
        electrode = canonical_electrode(label)
        if electrode not in SEIZURE_TRANSFORMER_ELECTRODES:
            continue
        if electrode in found:
            raise ElectrodeInputUnavailable(
                f"The recording has more than one channel for electrode {electrode}."
            )
        found[electrode] = index
    return found


def _missing_message(found: dict[str, int], labels: list[str]) -> str:
    missing = [e for e in SEIZURE_TRANSFORMER_ELECTRODES if e not in found]
    looks_bipolar = sum("-" in str(label) and canonical_electrode(label) is None for label in labels) >= 10
    if looks_bipolar and not found:
        return (
            "SeizureTransformer needs the 19 single electrodes (FP1, F3, ... P8), but this "
            "recording contains only bipolar pairs such as FP1-F7. Pairs cannot be turned back "
            "into single electrodes, so this recording cannot be scored by SeizureTransformer."
        )
    return "SeizureTransformer needs 19 single electrodes; missing: " + ", ".join(missing) + "."


def select_electrodes(signals: np.ndarray, labels: list[str]) -> np.ndarray:
    """Return a (19, samples) float64 array in SeizureTransformer order."""

    found = electrode_index(labels)
    if len(found) != len(SEIZURE_TRANSFORMER_ELECTRODES):
        raise ElectrodeInputUnavailable(_missing_message(found, labels))
    return np.asarray(signals[[found[e] for e in SEIZURE_TRANSFORMER_ELECTRODES]], dtype=np.float64)


def derive_bipolar_montage(signals: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]] | None:
    """Build MDS01's 18 bipolar channels from single electrodes, when possible.

    Returns ``None`` when the input already has the bipolar channels or lacks
    the electrodes, so callers keep their existing contract checks.
    """

    normalized = {str(label).strip().upper() for label in labels}
    if set(MODEL_CHANNELS) <= normalized:
        return None
    try:
        found = electrode_index(labels)
    except ElectrodeInputUnavailable:
        return None
    needed = {e for pair in MODEL_CHANNELS for e in pair.split("-", 1)}
    if not needed <= set(found):
        return None
    out = np.empty((len(MODEL_CHANNELS), signals.shape[1]), dtype=np.float64)
    for row, pair in enumerate(MODEL_CHANNELS):
        left, right = pair.split("-", 1)
        np.subtract(signals[found[left]], signals[found[right]], out=out[row])
    return out, list(MODEL_CHANNELS)


def read_electrode_segments(path: str | Path) -> tuple[list[tuple[float, np.ndarray]], int | float]:
    """Read the 19 electrodes as contiguous segments ``[(start_seconds, (19, n))]``.

    The sampling rate is the file's own; the model pipeline resamples. EDF is
    therefore read without the H5 contract's exact-256 Hz rule.
    """

    from backend.app.eeg.edf_io import read_uniform_edf
    from backend.app.eeg.io import detect_eeg_format, read_uniform_eeg

    candidate = Path(path)
    eeg_format = detect_eeg_format(candidate)
    if eeg_format == "nicolet-e":
        from backend.app.eeg.legacy_nicolet import LegacyNicoletReader

        segments, rate = LegacyNicoletReader(candidate).read_electrode_segments(SEIZURE_TRANSFORMER_ELECTRODES)
        return segments, rate
    if eeg_format == "edf":
        # EDF+D is rejected by the reader, so an EDF is one contiguous segment.
        signals, rate, labels = read_uniform_edf(candidate, require_model_rate=False)
    else:
        signals, rate, labels = read_uniform_eeg(candidate)   # .data has no gaps
    return [(0.0, select_electrodes(signals, list(labels)))], rate
