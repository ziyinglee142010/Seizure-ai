FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV XDG_CACHE_HOME=/tmp/xdg-cache

COPY backend/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir "pip==26.2.1"
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ARG INSTALL_RESEARCH=false
COPY backend/requirements-research.txt /tmp/requirements-research.txt
RUN case "$INSTALL_RESEARCH" in \
      1|true|TRUE|yes|YES) pip install --no-cache-dir -r /tmp/requirements-research.txt ;; \
      *) echo "Building without research model dependencies" ;; \
    esac

ARG INSTALL_VIDEO_DETECTION=false
COPY backend/requirements-video-detection.txt /tmp/requirements-video-detection.txt
RUN case "$INSTALL_VIDEO_DETECTION" in \
      1|true|TRUE|yes|YES) pip install --no-cache-dir -r /tmp/requirements-video-detection.txt ;; \
      *) echo "Building without VSViG video model dependencies" ;; \
    esac

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Fail the build if the video runtime cannot load in a clean installation.
RUN python -c "import cv2; assert cv2.__version__"

RUN groupadd --system mds01 && useradd --system --gid mds01 --home-dir /app --no-create-home mds01
RUN mkdir -p /tmp/matplotlib /tmp/xdg-cache /opt/vsvig /opt/eeg-model \
    && chown mds01:mds01 /tmp/matplotlib /tmp/xdg-cache /opt/vsvig /opt/eeg-model
# Empty mountpoint for the SeizureTransformer weights bind mount. Compose mounts
# an empty read-only volume at /opt/eeg-model, and Docker cannot create a file
# mountpoint inside a read-only volume. A new volume is seeded from this image
# directory, so the placeholder has to exist here. It is zero bytes and is
# always covered by the bind-mounted weights; the loader rejects it if unmounted.
RUN touch /opt/eeg-model/official_seizure_transformer_model.pth \
    && chmod 444 /opt/eeg-model/official_seizure_transformer_model.pth

COPY --chown=mds01:mds01 backend /app/backend

RUN mkdir -p /app/backend/storage /app/backend/database \
    && chown -R mds01:mds01 /app/backend/storage /app/backend/database

USER mds01

RUN python -m backend.scripts.verify_video_runtime

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
