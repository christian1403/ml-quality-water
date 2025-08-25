# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data models logs

# Generate dataset and train model
RUN python src/data_processing/generate_data.py
RUN python src/models/train_model.py

# Expose port for API
EXPOSE 8000

# Default command
CMD ["python", "src/api/fastapi_server.py"]
