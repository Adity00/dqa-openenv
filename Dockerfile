# ─── DQA-OpenEnv Dockerfile ───────────────────────────────────────────
# Builds the Data Quality Assurance OpenEnv environment.
# Hugging Face Spaces requires Dockerfile at REPO ROOT.
# HF Spaces requires port 7860.
# Build context: repo root (D:\OpenEnv2\dqa-openenv)
# ──────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY server/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire project into container
COPY . /app

# Set Python path so all imports resolve from /app
ENV PYTHONPATH="/app:$PYTHONPATH"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
