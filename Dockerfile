FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create model directories
RUN mkdir -p models/damage_classifier models/fraud_classifier

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Start command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]