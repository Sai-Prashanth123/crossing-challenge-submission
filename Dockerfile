# Crossing Challenge submission Dockerfile.
# Two-headed model: LightGBM intent + small GRU trajectory.
# Target image size: <= 2 GB. Built with CPU-only torch wheel.
#
# Build:
#   docker build -t my-crossing .
# Smoke test:
#   docker run --rm --network=none -v $(pwd)/data:/work my-crossing /work/dev.parquet /work/preds.csv

FROM python:3.11-slim

WORKDIR /app

# libgomp1 needed for lightgbm + xgboost runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps. CRITICAL: install CPU-only torch FIRST so pip sees the
# torch>=2.0,<3 constraint in requirements.txt as already satisfied. If we
# installed requirements.txt first, pip would grab CUDA torch (530 MB +
# ~1 GB of NVIDIA libs) from the default PyPI index — busting the 2 GB cap.
COPY requirements.txt .
RUN pip install --no-cache-dir "torch>=2.0,<3" --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Submission surface - predict.py, grade.py, features.py, weights.
# We do NOT unpickle model.pkl at build time.
COPY predict.py grade.py features.py ./
COPY model.pkl ./

ENTRYPOINT ["python", "grade.py"]
