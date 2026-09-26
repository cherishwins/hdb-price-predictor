# syntax=docker/dockerfile:1

# Base image pinned by digest for reproducible builds (python:3.11-slim as of
# 2026-09-26). Bump deliberately: resolve a new digest with
#   docker buildx imagetools inspect python:3.11-slim
# and update both FROM lines together (keep them identical). A tool like
# Dependabot can track this on a schedule instead of drifting on every rebuild.
ARG PYTHON_BASE=python:3.11-slim@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e

# ---- Builder: install Python deps into an isolated venv --------------------
FROM ${PYTHON_BASE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# All requirements ship as manylinux wheels for cp311, so no build toolchain
# is needed here. Keep this stage lean.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt \
    # xgboost's wheel pulls nvidia-nccl-cu12 (CUDA multi-GPU collectives, ~300MB).
    # This app runs CPU inference only, so drop the unused CUDA payload to keep
    # the runtime image small. Remove this line if you ever deploy on GPU.
    && pip uninstall -y nvidia-nccl-cu12 || true

# ---- Runtime: slim image, non-root, only what the app needs ---------------
FROM ${PYTHON_BASE} AS runtime

# libgomp1: OpenMP runtime required by xgboost (import fails without it).
# curl: used by the container HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

# Bring in the pre-built virtualenv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Application code and the model/data files it loads by relative path.
COPY app.py ./
COPY scaler.joblib ./
COPY model.bst ./
COPY postal_data.json ./

# Run as an unprivileged user with a writable home for Streamlit's config.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
