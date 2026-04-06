FROM python:3.10-slim

WORKDIR /app

# System dependencies: gcc for torch C extensions, curl for healthcheck, git for HuggingFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and config (NOT outputs/ — adapter weights are bind-mounted at runtime)
COPY src/ ./src/
COPY configs/ ./configs/

# Create runtime directories (adapter weights arrive via bind mount: -v ./outputs:/app/outputs)
RUN mkdir -p outputs/adapter_weights outputs/fine_tuned_model data

# Environment defaults — override with docker run -e or --env-file
ENV MODEL_NAME="microsoft/phi-2" \
    ADAPTER_PATH="./outputs/adapter_weights" \
    TEMPERATURE="0.7" \
    LOG_LEVEL="INFO"

EXPOSE 8000

# Poll /health after a 120s start-period to allow model loading time
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single worker — PyTorch inference is not fork-safe; scale via multiple containers + load balancer
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
