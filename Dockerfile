FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy the rest of the application
COPY . .

# Use the fixed main.py file
RUN cp fix_main.py main.py

# Environment variables will be set in the Cloud Run configuration
ENV PORT=8080

# Command to run
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app