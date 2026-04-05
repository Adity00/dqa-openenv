# ─── DQA-OpenEnv Dockerfile ───────────────────────────────────────────
# Builds the Data Quality Assurance OpenEnv environment.
# Compatible with Hugging Face Spaces (Docker SDK).
# Port 7860 is required by HF Spaces.
# ──────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY server/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . /app

# Set Python path so imports resolve correctly
ENV PYTHONPATH="/app:$PYTHONPATH"

# HF Spaces requires port 7860
EXPOSE 7860

# Health check — validator pings /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server on port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
