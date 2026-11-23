# AI-vEEG integrated platform

This build combines four pieces into one application, using MDS01 (`4ffaad/eeg-viewer`) as the base:

| Module | Origin | Where it runs | What it outputs |
| --- | --- | --- | --- |
| EEG review, accounts, encrypted storage, reports | MDS01 | whole app | — |
| **EEG seizure detection** | FYP-MCS18 SeizureTransformer | EEG analysis | per-window seizure score + flagged windows |
| **Video seizure detection** | MDS01 VSViG | video analysis | per-window VSViG score |
| **Facial emotion recognition** | AI-vEEG Phase 2 (this project) | video analysis, independent of VSViG (runs at the same time) | per-second facial expression |

MDS01's original H5 EEG model is kept in the code as a legacy option (`--research-h5`) but is not used by default.

## Starting it

1. Put the SeizureTransformer weights from the MCS18 team at
   `backend/model/official_seizure_transformer_model.pth` (the file is ignored by git).
2. Run `node scripts/demo.mjs`.
3. Open http://127.0.0.1:3000 and create a local account.

| Command | EEG model |
| --- | --- |
| `node scripts/demo.mjs` | SeizureTransformer (default) |
| `node scripts/demo.mjs --development-stub` | synthetic scores, no weights needed |
| `node scripts/demo.mjs --research-h5` | legacy MDS01 H5 model |

Video detection and emotion recognition run in every mode. The first Docker build downloads the VSViG assets and Python packages, so it needs internet access.

Without Docker, `docker-compose.seizure-transformer.yml` lists the settings to copy into your environment: `MODEL_RUNTIME=seizure_transformer` and `SEIZURE_TRANSFORMER_MODEL_PATH=<path to the .pth>`.

## EEG seizure detection (SeizureTransformer)

The signal processing is ported from MCS18 `web-app/backend/app/seizure_detection.py` without changes, and `model.py` is copied byte for byte. On the same input, MDS01 and MCS18 gave identical probabilities (maximum difference 0.0) and identical event masks.

**Input.** The 19 single 10-20 electrodes FP1, F3, C3, P3, O1, F7, T7, P7, FZ, CZ, PZ, FP2, F4, C4, P4, O2, F8, T8 and P8. The old names T3, T4, T5 and T6 are accepted, and so are labels such as `EEG FP1-REF` and `Fp1-LE`. The electrodes are read from:
- EDF files with single-electrode (referential) channels, at any single sampling rate (for example 200, 250, 256, 500 or 512 Hz). The model's own step 3 below resamples to 256 Hz. With this model, an EDF is not held to the legacy H5 model's exact-256 Hz rule; that rule still applies when the H5 model is selected;
- Nicolet `.data`/`.head`;
- legacy Nicolet `.e` files, read straight from the source at 500 Hz.

**Processing.**
1. Average reference.
2. Z-score each channel.
3. Resample to 256 Hz.
4. Cut 60-second windows.
5. Apply a 0.5–120 Hz band-pass and notch filters at 1 Hz and 60 Hz.
6. Run the network, which gives a probability for every sample.
7. Mark samples above 0.25, apply opening and closing, and drop events shorter than 2 s.

**How the results appear in MDS01.** MDS01 shows 4-second windows with a 2-second stride. For each window:
- the score is the highest sample probability inside it;
- the window is flagged when it overlaps a detected event.

Acquisition gaps in `.e` files are scored per continuous segment and are never filled.

**Limits to know.**
- Files that contain only bipolar pairs, such as `FP1-F7` (CHB-MIT style), cannot be scored: pairs cannot be turned back into single electrodes. The app says so instead of guessing.
- Signal obfuscation is not supported with this model. Use metadata scrubbing only.
- The 60 Hz notch matches MCS18. Malaysian mains power is 50 Hz, and changing the notch would change the scores, so ask the MCS18 team before changing it.
- Scores are uncalibrated and are not a diagnosis.
- The optional `SEIZURE_TRANSFORMER_SHA256` setting makes the app refuse any other weights file.

## Video seizure detection (VSViG)

This is MDS01's own pipeline, unchanged except for one fix. The pose network weights (`pose.pth`) contain float32 *denormal* values, which made CPU inference 26–31 times slower. Those values are set to zero when the weights load. The weights file on disk is unchanged, and the asset integrity check still passes.

## Facial emotion recognition

Emotion recognition is a separate job from VSViG. It starts as soon as a video upload is stored and runs at the same time as VSViG, on its own worker. It does **not** depend on VSViG: it also runs on clips that VSViG rejects (for example a clip that fails the opening pose check or is not 1920×1080), and VSViG does not wait for it. One emotion run happens at a time; the backend container's default CPU limit is 4 so the two lanes do not slow each other down (`BACKEND_CPU_LIMIT`).

Each run decrypts the original into its own temporary area (`emotion-work`), separate from VSViG's work area, and sends it to an isolated worker process (`python -m backend.app.emotion_recognition`). The worker runs these steps:
1. Detect faces with YuNet at 4 rotations, so faces lying sideways are still found.
2. Choose the patient as the face that stays in place within each 10-second segment, then follow it.
3. Align the face to a 112×112 crop using its 5 landmarks.
4. Classify the expression into 7 classes with the OpenCV Zoo MobileFaceNet model (trained on RAF-DB, 88.27% reported accuracy).
5. Combine the results into a timeline of one entry per second.

The results page shows:
- a colour strip with the main expression for each second;
- the share of seconds for each expression;
- the percentage of frames where a face was found;
- the note that these results have **no ground truth** on seizure video and are not validated. This follows the supervisor's direction.

**Privacy.** These rules match MDS01's VSViG rules:
- no face images are written anywhere;
- the decrypted copy and the worker's output files are deleted after each run, and any left by an interrupted run are deleted when the backend restarts;
- the per-second scores are stored encrypted in the database, in their own field bound to the job and its owner, and only the owner can open them (`GET /api/video-detection/jobs/{job_id}/emotion`);
- the API returns only fields on an allowlist, with every label and range checked.

Because emotion recognition runs on every uploaded analysis clip, faces in clips that VSViG rejects are now analysed as well, and the face detector also sees anyone else in frame (it then follows the face that stays in place). Confirm this is covered by your project's approval before using real patient video.

**Status.** Each analysis job reports `emotion_status`: `queued`, `processing`, `complete`, `failed`, `disabled` (turned off in settings), `unavailable` (no retained source), or `not_applicable` (reference clips and jobs from before this change). Jobs from before this change keep their emotion result inside the VSViG result, and the emotion endpoint still serves it. Deleting a case waits until any emotion run in it has finished.

**If it fails.** A failure in this step never affects the VSViG result, and a VSViG failure never stops this step. The page reports each one separately.

**Settings** (`.env`):

| Setting | Default | Meaning |
| --- | --- | --- |
| `EMOTION_RECOGNITION_ENABLED` | `true` | turn the step off |
| `EMOTION_FACE_REGION` | empty | limit the face search to `x0,y0,x1,y1` (fractions of the frame), e.g. the bed's head end |
| `EMOTION_SAMPLE_FPS` | `2` | frames analysed per second |
| `EMOTION_UPSCALE` | `1.0` | enlarge frames before detection (try 1.5 for small, distant faces) |
| `EMOTION_CHECKPOINT` | empty | path to SeizureAI `FaceEmotionModel` weights to use instead of the OpenCV Zoo model (native runs; in Docker, also mount the file and add the variable to the backend service) |

## Code map

| Area | Files |
| --- | --- |
| SeizureTransformer | `backend/app/ml/seizure_transformer/`, `backend/app/ml/seizure_transformer_inference.py` |
| Electrode reading | `backend/app/eeg/electrodes.py`, `LegacyNicoletReader.read_electrode_segments` |
| Emotion module | `backend/app/emotion_recognition/` |
| Emotion job lane | `backend/app/services/video_detection_service.py` (`process_emotion_job`, `_run_emotion_recognition`) |
| Public emotion fields and endpoint | `backend/app/api/video_detection.py` (`_public_emotion_result`, `GET /jobs/{job_id}/emotion`) |
| Emotion columns | `backend/migrations/versions/038_independent_emotion.py` |
| Results panel | `frontend/src/components/VideoEmotionPanel.tsx` |
| Pose speed fix | `backend/app/video_detection/runtime.py` (`_zero_denormal_weights`) |
| Launcher and Compose | `scripts/demo.mjs`, `docker-compose.seizure-transformer.yml` |

## Tests

```sh
PYTHONPATH=. python -m unittest discover -s backend/tests      # 524 tests
node --test scripts/*.test.mjs
cd frontend && npx tsc --noEmit && npm run lint
cd frontend && npx playwright test tests/e2e/video-detection.spec.ts
```

The new tests are:
- `test_seizure_transformer.py` (17 tests);
- `test_emotion_recognition.py` (14 tests);
- `test_pose_denormals.py` (2 tests);
- two browser tests for the emotion panel;
- a Compose check for the new overlay.
