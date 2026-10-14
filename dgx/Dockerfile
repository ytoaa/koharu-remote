ARG VISION_BASE_IMAGE=local/koharu-rorem:0.2.3
FROM ${VISION_BASE_IMAGE}
ARG VISION_BASE_IMAGE
LABEL org.koharu.remote.version="0.6.0" org.koharu.remote.base-image="${VISION_BASE_IMAGE}"
WORKDIR /app
COPY requirements.txt /tmp/vision-requirements.txt
COPY protect_torch.py /tmp/protect_torch.py
# Use system NVIDIA torch, not a new public PyTorch wheel. Only the NEW image changes.
RUN /usr/bin/python3 /tmp/protect_torch.py before \
    && /usr/bin/python3 -m venv --system-site-packages /opt/koharu-vision \
    && /opt/koharu-vision/bin/python -m pip install --no-cache-dir \
       -c /tmp/torch-constraints.txt -r /tmp/vision-requirements.txt --report /app/vision-pip-report.json \
    && /opt/koharu-vision/bin/python /tmp/protect_torch.py after \
    && /opt/koharu-vision/bin/python -m pip freeze > /app/vision-build-packages.txt
# supervision's pinned OpenCV wheel links XCB/GL even for headless detection.
# Keep this after Python installation so the existing dependency layer is reusable.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libxcb1 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && /opt/koharu-vision/bin/python -c "import cv2; import supervision"
ENV PATH="/opt/koharu-vision/bin:${PATH}" PYTHONUNBUFFERED=1
ENV VISION_BASE_IMAGE="${VISION_BASE_IMAGE}"
COPY vision /app/vision
CMD ["python", "-m", "uvicorn", "vision.app:app", "--host", "127.0.0.1", "--port", "8766", "--workers", "1", "--limit-concurrency", "8"]
