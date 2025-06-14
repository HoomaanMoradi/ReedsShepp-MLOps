FROM python:3.10.17-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-credentials.json

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip --timeout=300 -i https://mirror-pypi.runflare.com/simple
RUN pip install --no-cache-dir -e . --timeout=3000 -i https://mirror-pypi.runflare.com/simple

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Default command
CMD ["/bin/sh", "-c", "python pipeline/run.py && python web/application.py"]