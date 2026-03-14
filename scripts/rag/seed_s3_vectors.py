from __future__ import annotations

import json
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv
from google import genai


ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_PATH = ROOT / "generated" / "graphrag-documents.json"
EMBED_MODEL = "gemini-embedding-001"
KEY_PREFIX = "frontend-rag:"
UPSERT_BATCH_SIZE = 100
DELETE_BATCH_SIZE = 500


def get_env(name: str) -> str:
    value = os.getenv(name, "").strip().strip("\"'")
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def infer_region(index_arn: str) -> str | None:
    parts = index_arn.split(":")
    if len(parts) >= 4 and parts[2] == "s3vectors":
        return parts[3]
    return None


def load_documents() -> list[dict]:
    payload = json.loads(DOCUMENTS_PATH.read_text(encoding="utf-8"))
    return payload.get("documents", [])


def batched(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def list_all_existing_keys(client, index_arn: str) -> list[str]:
    keys: list[str] = []
    next_token = None
    while True:
        kwargs = {
            "indexArn": index_arn,
            "maxResults": 1000,
            "returnData": False,
            "returnMetadata": False,
        }
        if next_token:
            kwargs["nextToken"] = next_token
        result = client.list_vectors(**kwargs)
        for vector in result.get("vectors", []):
            key = vector.get("key")
            if key:
                keys.append(key)
        next_token = result.get("nextToken")
        if not next_token:
            break
    return keys


def cleanup_stale_vectors(client, index_arn: str, wanted_keys: set[str]) -> None:
    existing_keys = list_all_existing_keys(client, index_arn)
    stale_keys = [key for key in existing_keys if key.startswith(KEY_PREFIX) and key not in wanted_keys]
    if not stale_keys:
        print("No stale GraphRAG vectors to delete.")
        return

    print(f"Deleting {len(stale_keys)} stale GraphRAG vector(s)...")
    for index, batch in enumerate(batched(stale_keys, DELETE_BATCH_SIZE), start=1):
        client.delete_vectors(indexArn=index_arn, keys=batch)
        print(f"Deleted stale batch {index} ({len(batch)} keys).")


def embed_text(ai: genai.Client, text: str) -> list[float] | None:
    response = ai.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    if not response.embeddings:
        return None
    values = response.embeddings[0].values
    return list(values) if values else None


def main() -> None:
    load_dotenv()

    google_api_key = get_env("GOOGLE_API_KEY")
    index_arn = get_env("AWS_S3_VECTOR_INDEX_ARN")
    aws_region = os.getenv("AWS_REGION", "").strip() or infer_region(index_arn)
    if not aws_region:
        raise RuntimeError(
            "Missing AWS_REGION and could not infer the AWS region from AWS_S3_VECTOR_INDEX_ARN."
        )

    ai = genai.Client(api_key=google_api_key)
    s3vectors = boto3.client("s3vectors", region_name=aws_region)
    documents = load_documents()

    wanted_keys = {f"{KEY_PREFIX}{doc['key']}" for doc in documents}
    cleanup_stale_vectors(s3vectors, index_arn, wanted_keys)

    vectors = []
    for doc in documents:
        print(f"Embedding {doc['key']} ({doc['type']})")
        embedding = embed_text(ai, doc["text"])
        if not embedding:
            print(f"Skipped {doc['key']}: empty embedding response")
            continue
        vectors.append(
            {
                "key": f"{KEY_PREFIX}{doc['key']}",
                "data": {"float32": embedding},
            }
        )

    print(f"Uploading {len(vectors)} vector(s) to S3 Vectors...")
    for index, batch in enumerate(batched(vectors, UPSERT_BATCH_SIZE), start=1):
        s3vectors.put_vectors(indexArn=index_arn, vectors=batch)
        print(f"Uploaded batch {index} ({len(batch)} vectors).")

    print("S3 Vector seeding complete.")


if __name__ == "__main__":
    main()
