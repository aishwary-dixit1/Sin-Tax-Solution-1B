# Use a lightweight, AMD64-compatible Python base image
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Copy and install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your main application script
COPY main.py .

# Set the command to run your script
CMD ["python", "main.py"]