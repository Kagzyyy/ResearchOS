# Serving image for ResearchOS.
#
# CPU-only by design: the container serves the INT8 ONNX reranker (row 6), which
# is why that row exists. Training happens on the host GPU, not here.
#
#   docker build -t researchos .
#   docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY researchos

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# CPU wheel of torch keeps the image an order of magnitude smaller than the CUDA one.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Baked at build time: the exported ONNX reranker and its tokenizer.
# Indices and the SciFact corpus are built on first start and cached in the volume.
COPY models/onnx/ ./models/onnx/

VOLUME ["/app/cache", "/app/datasets", "/app/results"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENV RESEARCHOS_CONFIG=configs/app.docker.yaml

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
