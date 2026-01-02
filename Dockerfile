FROM python:3.9-slim

LABEL maintainer="GAMBA++ Team"
LABEL description="GAMBA++ - High-performance MBA simplification framework"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for cache and output
RUN mkdir -p /app/cache /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port (if needed for API in future)
EXPOSE 8000

# Default command
CMD ["python", "-c", "from optimization.batch_advanced import process_expressions_batch_advanced; print('GAMBA++ is ready!')"]

