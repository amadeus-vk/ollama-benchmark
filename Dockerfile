FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy benchmark scripts
COPY scripts/ ./scripts/
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make scripts executable
RUN chmod +x scripts/*.sh scripts/*.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python scripts/healthcheck.py

CMD ["python", "scripts/benchmark.py"]