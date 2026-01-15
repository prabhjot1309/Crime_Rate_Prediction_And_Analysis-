FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies (added scikit-learn + joblib)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        numpy==1.21.6 \
        pandas==1.5.3 \
        Flask==2.3.2 \
        scikit-learn==1.3.0 \
        joblib==1.3.2

# Expose port
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]