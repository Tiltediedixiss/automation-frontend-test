"""
main.py
-------
One endpoint. Friend POSTs the s3_key, waits ~10 min, gets the result JSON back.
No SQS, no DynamoDB, no polling.

Run:
    pip install fastapi uvicorn boto3 python-dotenv twelvelabs google-genai
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import uuid
import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

import pipeline

API_SECRET = os.getenv("PIPELINE_API_SECRET", "")

app = FastAPI(title="Alims Pipeline API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def verify(x_api_key: str = Header(default="")) -> None:
    if API_SECRET and x_api_key != API_SECRET:
        raise HTTPException(401, "Invalid API key")


class ProcessRequest(BaseModel):
    s3_key:    str
    s3_bucket: str | None = None   # falls back to S3_BUCKET env var


@app.post("/process")
def process(req: ProcessRequest, _: None = Depends(verify)) -> dict:
    """
    Runs the full pipeline synchronously.
    Connection stays open for ~10 minutes while processing.

    Request:
        { "s3_key": "videos/uuid/recording.mp4" }

    Response:
        {
            "task_id":      "...",
            "summary":      "PM wants drag-and-drop...",
            "files":        ["src/components/student/folder/page.tsx", ...],
            "instructions": "## Requested changes\n\n..."
        }
    """
    bucket = req.s3_bucket or os.getenv("S3_BUCKET", "")
    if not bucket:
        raise HTTPException(400, "s3_bucket not provided and S3_BUCKET env var is not set")

    task_id = str(uuid.uuid4())

    try:
        return pipeline.run(s3_bucket=bucket, s3_key=req.s3_key, task_id=task_id)
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/health")
def health() -> dict:
    return {"ok": True}