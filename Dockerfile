FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && apt-get clean

COPY requirements.txt .
# Install CPU-only versions of torch and torchvision
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu --timeout=100
RUN pip install --no-cache-dir --timeout=100 -r requirements.txt

COPY app/ /app/app
COPY models/ /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
