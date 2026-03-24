FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -c "import boto3; s=boto3.session.Session(); print('boto3', boto3.__version__); assert 's3vectors' in s.get_available_services(), 'boto3 build missing s3vectors service'"

COPY main.py pipeline.py query_rag.py ./
COPY generated/graphrag-documents.json ./generated/graphrag-documents.json

ENV PORT=8000

CMD uvicorn main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 300