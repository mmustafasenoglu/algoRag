FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Start the FastAPI server on 0.0.0.0 so it is accessible externally
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
