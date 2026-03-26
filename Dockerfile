# Use a slim Python 3.8 image for a smaller footprint
FROM python:3.8-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (required for some scikit-learn/pandas operations if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and models into the container
# Ensure your local structure has 'src', 'models', and 'reference_data' in the same root
COPY src/ ./src/
COPY models/ ./models/
COPY src/reference_data/ ./reference_data/

# Set the PYTHONPATH so the app can find the maco_automation package
ENV PYTHONPATH=/app/src

# Expose the port specified in your main.py
EXPOSE 8000

# Run the application
CMD ["python", "src/maco_automation/main.py"]