FROM python:3.11-slim

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy requirements and install Python dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir torch torchvision
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy application code
COPY ./app /code/app

# Create models directory
RUN mkdir -p /code/models

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/code
ENV MODEL_DIR=/code/models

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
