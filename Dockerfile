FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies for llama-cpp-python and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

ENV AETHER_HOST=0.0.0.0
ENV AETHER_PORT=5002
ENV AETHER_ENV=production
ENV AETHER_GGUF_PATH=checkpoints/model.gguf

EXPOSE 5002

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:5002/health || exit 1

CMD ["python", "serving/server.py"]
