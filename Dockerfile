# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed for scikit-learn and numpy
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Run the app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]