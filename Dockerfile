FROM python:3.12-slim AS lesson-base

ENV HF_HOME=/model-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu \
    && python -m pip install --no-cache-dir transformers==4.57.6

COPY lesson01_forward.py .

ENTRYPOINT ["python", "lesson01_forward.py"]

FROM lesson-base AS benchmark

RUN python -m pip install --no-cache-dir onnx==1.23.2 onnxruntime==1.26.0

COPY benchmark.py .

ENTRYPOINT ["python", "benchmark.py"]

FROM lesson-base AS lesson
