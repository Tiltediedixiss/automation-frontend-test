FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py pipeline.py query_rag.py ./
COPY generated/graphrag-documents.json ./generated/graphrag-documents.json

ENV PORT=8000

# Increase timeout to 20 min — pipeline takes ~10 min per video
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 1200