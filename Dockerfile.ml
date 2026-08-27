# ResearchOS-ML backend.
#
# CPU-only image. The reranker runs ~10x slower on CPU than on the 4060, which is
# acceptable for a small deployment and keeps the image an order of magnitude
# smaller than a CUDA base. For GPU serving, run it on the host instead.
#
#   docker build -f Dockerfile.ml -t researchos-ml .
#   docker run --rm -p 8100:8100 \
#     -e GROQ_API_KEY=$GROQ_API_KEY \
#     -v "$PWD/data:/app/data:ro" \
#     -v "$PWD/cache/ml:/app/cache/ml" \
#     -v researchos_results:/app/results \
#     researchos-ml
#
# data/ holds arxiv_papers.jsonl and arxiv_chunks.jsonl - built offline by
# scripts/ingest_arxiv.py and mounted read-only. cache/ml holds the BM25 and
# FAISS indices; leave it writable so the first start can build them, or bake
# them in to make startup instant.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    RESEARCHOS_ML_CONFIG=configs/ml.docker.yaml

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bake the embedding and reranker weights so the first request is not a download.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('BAAI/bge-reranker-base', num_labels=1)"

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

VOLUME ["/app/data", "/app/cache", "/app/results"]
EXPOSE 8100

# start-period is generous: the first boot builds indices over 40k chunks.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8100/health || exit 1

CMD ["uvicorn", "src.api.ml_app:app", "--host", "0.0.0.0", "--port", "8100"]
