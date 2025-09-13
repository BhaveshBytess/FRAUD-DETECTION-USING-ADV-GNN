# hHGTN Fraud Detection - Production Docker Image
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p experiments/demo reports assets

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for web interface (if needed)
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import torch_geometric; print('Health check passed')" || exit 1

# Default command
CMD ["python", "scripts/collect_demo_artifacts.py", "--help"]

# Docker build and run instructions:
# 
# Build the image:
# docker build -t hhgtn-fraud-detection .
#
# Run interactive container:
# docker run -it --rm -v $(pwd)/experiments:/app/experiments hhgtn-fraud-detection bash
#
# Run demo:
# docker run --rm -v $(pwd)/experiments:/app/experiments hhgtn-fraud-detection python notebooks/demo.ipynb
