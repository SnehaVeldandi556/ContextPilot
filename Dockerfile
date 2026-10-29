# ContextPilot Dockerfile
FROM python:3.11-slim

# System dependencies for psycopg2 and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install CPU-only PyTorch first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000 8501

CMD ["uvicorn", "contextpilot.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
