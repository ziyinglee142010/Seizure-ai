"""Select and cache the configured inference runtime."""

from __future__ import annotations

from functools import lru_cache

from backend.app.core.config import (
    H5_CONTRACT_PATH,
    H5_MODEL_PATH,
    MODEL_RUNTIME,
    SEIZURE_TRANSFORMER_MODEL_PATH,
    SEIZURE_TRANSFORMER_SHA256,
    SEIZURE_TRANSFORMER_THRESHOLD,
)
from backend.app.ml.h5_inference import H5InferenceService
from backend.app.ml.interface import InferenceService
from backend.app.ml.stub_inference import StubInferenceService


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Construct the configured inference adapter.

    Returns
    -------
    InferenceService
        Development stub when ``MODEL_RUNTIME=stub``; SeizureTransformer when
        ``MODEL_RUNTIME=seizure_transformer``; the legacy MDS01 H5 model when
        ``MODEL_RUNTIME=h5``.

    Raises
    ------
    RuntimeError
        Raised when a real runtime is requested before its model adapter has
        been implemented and supplied with an artifact.
    """

    if MODEL_RUNTIME == "stub":
        return StubInferenceService()
    if MODEL_RUNTIME == "seizure_transformer":
        from backend.app.ml.seizure_transformer_inference import SeizureTransformerService

        return SeizureTransformerService(
            SEIZURE_TRANSFORMER_MODEL_PATH,
            SEIZURE_TRANSFORMER_SHA256,
            threshold=SEIZURE_TRANSFORMER_THRESHOLD,
        )
    if MODEL_RUNTIME == "h5":
        return H5InferenceService(H5_MODEL_PATH, H5_CONTRACT_PATH)
    raise RuntimeError(
        "MODEL_RUNTIME is not 'stub', but no real model adapter has been supplied."
    )
