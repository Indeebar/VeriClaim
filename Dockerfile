FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for sklearn, xgboost, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Create model directories (safety)
RUN mkdir -p models/damage_classifier models/fraud_classifier

# Expose HF Spaces port
EXPOSE 7860

# IMPORTANT: Download models BEFORE starting API
CMD ["bash", "-c", "python scripts/download_models.py && uvicorn api.main:app --host 0.0.0.0 --port 7860"]