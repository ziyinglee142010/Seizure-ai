"""SeizureTransformer (from FYP-MCS18) as MDS01's EEG model.

No real weights are needed: tests use a randomly initialised network of the
real architecture, or a small stand-in network with a known output.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyedflib
from fastapi import UploadFile
from sqlmodel import Session, SQLModel, create_engine

from backend.app.database.models.eeg import AnalysisStatus, EEGSession
from backend.app.database.repository import (
    get_session_by_public_id,
    list_predictions,
    list_recordings_for_session,
)
from backend.app.eeg.contracts import MODEL_CHANNELS
from backend.app.eeg.electrodes import (
    SEIZURE_TRANSFORMER_ELECTRODES,
    ElectrodeInputUnavailable,
    canonical_electrode,
    derive_bipolar_montage,
    read_electrode_segments,
    select_electrodes,
)
from backend.app.eeg.model_input import preprocess_eeg
from backend.app.ml.seizure_transformer import pipeline
from backend.app.ml.seizure_transformer_inference import (
    SeizureTransformerError,
    SeizureTransformerService,
)
from backend.app.privacy.deidentify import deidentify_edf, inspect_metadata
from backend.app.services.storage_service import SessionStorage

try:
    import torch
except ImportError:  # pragma: no cover - the EEG image installs torch with the video runtime
    torch = None

# Source spellings as found in clinical exports (TUH-style "-REF", old T3/T5 names).
SOURCE_LABELS = (
    "EEG FP1-REF", "EEG F3-REF", "EEG C3-REF", "EEG P3-REF", "EEG O1-REF",
    "EEG F7-REF", "EEG T3-REF", "EEG T5-REF",
    "EEG FZ-REF", "EEG CZ-REF", "EEG PZ-REF",
    "EEG FP2-REF", "EEG F4-REF", "EEG C4-REF", "EEG P4-REF", "EEG O2-REF",
    "EEG F8-REF", "EEG T4-REF", "EEG T6-REF",
)


def write_edf(path: Path, labels, seconds: int, *, seed: int = 0, rate: int = 256) -> np.ndarray:
    """EDF with identifying header text, at ``rate`` Hz, digital samples returned."""

    rng = np.random.default_rng(seed)
    samples = rng.integers(-1500, 1500, size=(len(labels), rate * seconds), dtype=np.int32)
    headers = [{
        "label": label, "dimension": "uV", "sample_frequency": rate,
        "physical_min": -1000, "physical_max": 1000, "digital_min": -2048, "digital_max": 2047,
        "prefilter": "", "transducer": "",
    } for label in labels]
    writer = pyedflib.EdfWriter(str(path), len(labels), file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setHeader({
            "technician": "Technician Name", "recording_additional": "Ward 2",
            "patientname": "Jane Doe", "patient_additional": "Address", "patientcode": "PATIENT-123",
            "equipment": "EEG Device", "admincode": "ADMIN-1", "sex": "F",
            "startdate": datetime(2025, 1, 1, 12, 0, 0), "birthdate": "01 jan 1990",
        })
        writer.setSignalHeaders(headers)
        writer.writeSamples(samples, digital=True)
        writer.writeAnnotation(1.0, 2.0, "Jane Doe seizure note")
    finally:
        writer.close()
    return samples


class ConstantNet(torch.nn.Module if torch else object):
    """Stand-in network: the same probability for every sample."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0], x.shape[2]), self.value)


class ElectrodeTests(unittest.TestCase):
    def test_canonical_electrode_names(self) -> None:
        self.assertEqual(canonical_electrode("EEG FP1-REF"), "FP1")
        self.assertEqual(canonical_electrode("Fp1-LE"), "FP1")
        self.assertEqual(canonical_electrode("T3"), "T7")
        self.assertEqual(canonical_electrode("EEG T6-REF"), "P8")
        self.assertIsNone(canonical_electrode("FP1-F7"))       # a bipolar pair

    def test_bipolar_montage_is_built_from_electrode_differences(self) -> None:
        rng = np.random.default_rng(3)
        signals = rng.normal(size=(len(SOURCE_LABELS), 100))
        derived, labels = derive_bipolar_montage(signals, list(SOURCE_LABELS))
        self.assertEqual(labels, list(MODEL_CHANNELS))
        index = {canonical_electrode(l): i for i, l in enumerate(SOURCE_LABELS)}
        for row, pair in enumerate(MODEL_CHANNELS):
            left, right = pair.split("-")
            np.testing.assert_allclose(derived[row], signals[index[left]] - signals[index[right]])

    def test_existing_bipolar_input_is_left_unchanged(self) -> None:
        self.assertIsNone(derive_bipolar_montage(np.zeros((18, 10)), list(MODEL_CHANNELS)))

    def test_bipolar_only_recording_gets_an_actionable_message(self) -> None:
        with self.assertRaises(ElectrodeInputUnavailable) as raised:
            select_electrodes(np.zeros((18, 10)), list(MODEL_CHANNELS))
        self.assertIn("only bipolar pairs", str(raised.exception))

    def test_duplicate_electrode_fails_closed(self) -> None:
        labels = list(SOURCE_LABELS) + ["FP1"]
        with self.assertRaises(ElectrodeInputUnavailable):
            select_electrodes(np.zeros((len(labels), 10)), labels)


class EventLogicTests(unittest.TestCase):
    def test_short_events_are_removed_and_long_ones_kept(self) -> None:
        trace = np.zeros(256 * 30)
        trace[256:256 + 300] = 0.9                 # 1.2 s: shorter than 2 s, removed
        trace[256 * 10:256 * 15] = 0.9             # 5 s: kept
        trace[256 * 20:256 * 20 + 3] = 0.9         # 3 samples: removed by opening
        events = pipeline.events_from_mask(pipeline.event_mask(trace, 0.25), trace)
        self.assertEqual(events, [{"start_seconds": 10.0, "end_seconds": 15.0,
                                   "mean_probability": 0.9, "max_probability": 0.9}])

    def test_window_scores_use_the_highest_sample_and_flags_follow_events(self) -> None:
        service = SeizureTransformerService(None, network=ConstantNet(0.0)) if torch else None
        if service is None:
            self.skipTest("PyTorch is not installed")
        prob = np.zeros(256 * 40)
        prob[256 * 10:256 * 15] = 0.9              # a 5 s event
        prob[256 * 30:256 * 30 + 128] = 0.6        # a 0.5 s crossing: too short to be an event
        with patch.object(service, "segment_probability", return_value=prob):
            predictions = service.predict_electrodes([(0.0, np.zeros((19, 10)))], 256,
                                                     np.arange(0, 37, 2.0))
        by_start = {p.start_seconds: p for p in predictions}
        self.assertTrue(by_start[8.0].seizure_detected)       # 8-12 s overlaps the event
        self.assertAlmostEqual(by_start[8.0].probability, 0.9)
        self.assertFalse(by_start[0.0].seizure_detected)
        self.assertFalse(by_start[16.0].seizure_detected)     # 16-20 s is after it
        self.assertAlmostEqual(by_start[28.0].probability, 0.6)
        self.assertFalse(by_start[28.0].seizure_detected)     # above threshold, not an event
        self.assertEqual([p.window_index for p in predictions], list(range(len(predictions))))

    def test_window_outside_the_recording_is_an_error(self) -> None:
        if torch is None:
            self.skipTest("PyTorch is not installed")
        service = SeizureTransformerService(None, network=ConstantNet(0.0))
        with patch.object(service, "segment_probability", return_value=np.zeros(256 * 8)):
            with self.assertRaises(SeizureTransformerError):
                service.predict_electrodes([(0.0, np.zeros((19, 10)))], 256, np.array([6.0]))

    def test_gapped_recordings_map_each_window_to_its_own_segment(self) -> None:
        if torch is None:
            self.skipTest("PyTorch is not installed")
        service = SeizureTransformerService(None, network=ConstantNet(0.0))
        first, second = np.zeros(256 * 10), np.full(256 * 10, 0.9)
        with patch.object(service, "segment_probability", side_effect=[first, second]):
            predictions = service.predict_electrodes(
                [(0.0, np.zeros((19, 10))), (100.0, np.zeros((19, 10)))], 256, np.array([2.0, 104.0]))
        self.assertEqual([p.probability for p in predictions], [0.0, 0.9])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class WeightsTests(unittest.TestCase):
    def _saved_random_model(self, directory: Path) -> Path:
        from backend.app.ml.seizure_transformer.model import SeizureTransformer

        torch.manual_seed(0)
        network = SeizureTransformer(in_channels=19, in_samples=15360, dim_feedforward=2048,
                                     num_layers=8, num_heads=4)
        path = directory / "official_seizure_transformer_model.pth"
        torch.save(network.state_dict(), path)
        return path

    def test_real_architecture_loads_and_scores_every_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._saved_random_model(Path(directory))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            service = SeizureTransformerService(path, digest)
            self.assertEqual(service.model_version, f"sha256:{digest[:12]}")
            self.assertFalse(hasattr(service, "model"))   # keeps the Keras SHAP step off
            prob = service.segment_probability(np.random.default_rng(0).normal(size=(19, 500 * 70)), 500)
            self.assertEqual(prob.shape, (256 * 70,))
            self.assertTrue(np.all((prob >= 0) & (prob <= 1)))

    def test_weights_that_do_not_match_the_pin_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._saved_random_model(Path(directory))
            with self.assertRaisesRegex(SeizureTransformerError, "do not match"):
                SeizureTransformerService(path, "0" * 64)

    def test_missing_weights_name_the_file_to_get(self) -> None:
        with self.assertRaisesRegex(SeizureTransformerError, "official_seizure_transformer_model.pth"):
            SeizureTransformerService(Path("/nonexistent/official_seizure_transformer_model.pth"))

    def test_empty_mountpoint_placeholder_is_treated_as_missing_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            placeholder = Path(directory) / "official_seizure_transformer_model.pth"
            placeholder.touch()
            with self.assertRaisesRegex(SeizureTransformerError, "weights not found"):
                SeizureTransformerService(placeholder)


class ReferentialEdfTests(unittest.TestCase):
    def test_deidentified_referential_edf_keeps_only_canonical_electrodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.edf", root / "deid.edf"
            write_edf(source, SOURCE_LABELS + ("ECG", "Patient Jane"), 8)
            deidentify_edf(source, output, "REC-TEST")
            reader = pyedflib.EdfReader(str(output))
            try:
                self.assertEqual(reader.getSignalLabels(), list(SEIZURE_TRANSFORMER_ELECTRODES))
                self.assertNotIn("Jane", reader.getPatientName())   # EDF+ writes "X" for empty
            finally:
                reader.close()
            self.assertTrue(inspect_metadata(output)["is_deidentified"])
            segments, rate = read_electrode_segments(output)
            self.assertEqual((rate, segments[0][1].shape), (256, (19, 256 * 8)))
            windows, starts, _ = preprocess_eeg(output)
            self.assertEqual(windows.shape[1:], (1024, 18))
            self.assertEqual(starts.tolist(), [0.0, 2.0, 4.0])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class FullPipelineTests(unittest.TestCase):
    storage_key = b"s" * 32

    def _process(self, root: Path, labels, session_id: str, value: float, rate: int = 256, service=None):
        from backend.app.services.processing_service import process_session

        source = root / "recording.edf"
        write_edf(source, labels, 70, rate=rate)
        archive = root / "session.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.write(source, source.name)
        database = create_engine("sqlite://", connect_args={"check_same_thread": False})
        self.addCleanup(database.dispose)
        SQLModel.metadata.create_all(database)
        storage = SessionStorage(root / "sessions", self.storage_key)
        upload = UploadFile(filename=archive.name, file=io.BytesIO(archive.read_bytes()))
        encrypted = asyncio.run(storage.save_upload(session_id, upload))
        with Session(database) as db:
            db.add(EEGSession(session_id=session_id, privacy_method="metadata-scrub",
                              original_filename="session.zip", original_path=str(encrypted)))
            db.commit()
        service = service or SeizureTransformerService(None, network=ConstantNet(value))
        with (
            patch("backend.app.services.processing_service.engine", database),
            patch("backend.app.services.processing_service.SessionStorage", return_value=storage),
            patch("backend.app.services.processing_service.get_inference_service", return_value=service),
        ):
            process_session(session_id)
        return database

    def test_referential_edf_is_scored_on_the_mds01_window_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._process(Path(directory), SOURCE_LABELS, "SES-ST", 0.9)
            with Session(database) as db:
                session = get_session_by_public_id(db, "SES-ST")
                record = list_recordings_for_session(db, session.id)[0]
                predictions = list_predictions(db, record.id)
                self.assertEqual(session.status, AnalysisStatus.COMPLETED)
                self.assertEqual(len(predictions), 34)        # 70 s, 4 s windows, 2 s stride
                self.assertTrue(all(p.seizure_detected for p in predictions))
                self.assertEqual({p.model_name for p in predictions}, {"seizure-transformer-mcs18"})
                self.assertAlmostEqual(predictions[0].probability, 0.9, places=5)

    def test_referential_edf_at_other_rates_is_scored_on_the_same_grid(self) -> None:
        for rate in (500, 250):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as directory:
                session_id = f"SES-ST-{rate}"
                database = self._process(Path(directory), SOURCE_LABELS, session_id, 0.9, rate=rate)
                with Session(database) as db:
                    session = get_session_by_public_id(db, session_id)
                    record = list_recordings_for_session(db, session.id)[0]
                    predictions = list_predictions(db, record.id)
                    self.assertIsNone(record.error_message)
                    self.assertEqual(session.status, AnalysisStatus.COMPLETED)
                    self.assertEqual(record.sampling_rate, rate)
                    self.assertEqual(len(predictions), 34)    # same grid as 256 Hz
                    self.assertAlmostEqual(predictions[0].probability, 0.9, places=5)
                    self.assertEqual(
                        sorted(p.window_index for p in predictions), list(range(34))
                    )

    def test_h5_contract_still_requires_256_hz_edf(self) -> None:
        class BipolarOnlyService:
            """Stands in for the H5 runtime: no electrode input, never called."""

            model_name, model_version, threshold = "h5-stand-in", "test", 0.5

            def predict(self, *args, **kwargs):
                raise AssertionError("H5 inference must not run on a 500 Hz EDF")

        with tempfile.TemporaryDirectory() as directory:
            database = self._process(
                Path(directory), MODEL_CHANNELS, "SES-H5-500", 0.9,
                rate=500, service=BipolarOnlyService(),
            )
            with Session(database) as db:
                session = get_session_by_public_id(db, "SES-H5-500")
                record = list_recordings_for_session(db, session.id)[0]
                self.assertIn("exactly 256 Hz", record.error_message or "")
                self.assertEqual(list_predictions(db, record.id), [])

    def test_bipolar_only_edf_fails_with_a_message_the_user_can_act_on(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._process(Path(directory), MODEL_CHANNELS, "SES-BIPOLAR", 0.9)
            with Session(database) as db:
                session = get_session_by_public_id(db, "SES-BIPOLAR")
                record = list_recordings_for_session(db, session.id)[0]
                self.assertIn("only bipolar pairs", record.error_message or "")
                self.assertEqual(list_predictions(db, record.id), [])


class WindowGridTests(unittest.TestCase):
    def test_duration_grid_matches_the_256_hz_preprocessing_grid(self) -> None:
        from backend.app.eeg.model_input import preprocess_eeg, window_start_grid

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bipolar.edf"
            write_edf(source, MODEL_CHANNELS, 23)
            _windows, starts, _details = preprocess_eeg(source)
        np.testing.assert_array_equal(window_start_grid(23.0), starts)

    def test_recordings_shorter_than_one_window_are_rejected(self) -> None:
        from backend.app.eeg.model_input import window_start_grid

        with self.assertRaisesRegex(ValueError, "full 4-second model window"):
            window_start_grid(3.9)

    def test_validation_keeps_256_hz_by_default_and_relaxes_it_on_request(self) -> None:
        from backend.app.services.validation_service import ValidationError, validate_edf

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rate500.edf"
            write_edf(source, SOURCE_LABELS, 6, rate=500)
            with self.assertRaisesRegex(ValidationError, "exactly 256 Hz"):
                validate_edf(source)
            self.assertEqual(validate_edf(source, require_model_rate=False)["sampling_rate"], 500)


class LegacyNicoletElectrodeTests(unittest.TestCase):
    """Electrode reading from Nicolet .e (HUKM's format), with a recording gap."""

    def _reader(self, labels):
        from contextlib import nullcontext
        from backend.app.eeg.legacy_nicolet import (
            LegacyNicoletReader, NicoletChannel, NicoletHeader, NicoletSegment,
        )

        channels = tuple(NicoletChannel(label, 500, 1.0, i) for i, label in enumerate(labels))
        header = NicoletHeader(sampling_rate=500, channels=channels,
                               segments=(NicoletSegment(0.0, 12.0, 6000, 0),
                                         NicoletSegment(24.0, 12.0, 6000, 6000)),
                               total_samples=12000, events=())
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "recording.e"
        path.write_bytes(b"synthetic")
        reader = LegacyNicoletReader(path)

        def samples(_stream, channel, _total, *, sample_start=0, sample_count=None):
            # value = channel section id, plus a ramp so segments are distinguishable
            return np.full(sample_count, float(channel.section_id)) + sample_start / 1e6

        for name, value in (("read_header", header), ("_open_source", nullcontext(None))):
            patcher = patch.object(reader, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(reader, "_read_channel_samples", side_effect=samples)
        patcher.start()
        self.addCleanup(patcher.stop)
        return reader

    def test_reads_19_electrodes_in_model_order_per_segment(self) -> None:
        labels = ["Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "Fz", "Cz", "Pz",
                  "Fp2", "F4", "C4", "P4", "O2", "F8", "T4", "T6", "EKG", "Photic"]
        reader = self._reader(labels[::-1])           # source order differs from model order
        segments, rate = reader.read_electrode_segments(SEIZURE_TRANSFORMER_ELECTRODES)
        self.assertEqual(rate, 500)
        self.assertEqual([start for start, _ in segments], [0.0, 24.0])
        self.assertEqual(segments[0][1].shape, (19, 6000))
        source_index = {label: i for i, label in enumerate(labels[::-1])}
        expected = [source_index[l] for l in labels[:19]]
        self.assertEqual(np.round(segments[0][1][:, 0]).astype(int).tolist(), expected)
        self.assertAlmostEqual(segments[1][1][0, 0] - round(segments[1][1][0, 0]), 0.006)

    def test_missing_electrode_names_what_is_missing(self) -> None:
        reader = self._reader(["Fp1", "F3"])
        with self.assertRaisesRegex(ElectrodeInputUnavailable, "missing"):
            reader.read_electrode_segments(SEIZURE_TRANSFORMER_ELECTRODES)


if __name__ == "__main__":
    unittest.main()
