FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential libopenblas-dev \
    libx11-6 libglib2.0-0 libgl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

COPY . .

RUN mkdir -p checkpoints data/frames/fake data/frames/real

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
