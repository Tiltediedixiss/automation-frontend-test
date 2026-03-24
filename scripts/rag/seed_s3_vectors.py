from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv
from google import genai


ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_PATH = ROOT / "generated" / "graphrag-documents.json"
RAG_MANIFEST_PATH = ROOT / "generated" / "rag-manifest.json"
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


def doc_content_hash(doc: dict) -> str:
    """Stable hash for change detection; only key + type + text matter for embedding."""
    canonical = json.dumps(
        {"key": doc["key"], "type": doc["type"], "text": doc["text"]},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _manifest_s3_uri() -> str | None:
    return os.getenv("RAG_MANIFEST_S3_URI", "").strip() or None


def _parse_s3_uri(uri: str) -> tuple[str, str] | None:
    from urllib.parse import urlparse
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        return None
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    return (bucket, key) if bucket and key else None


def load_manifest(aws_region: str) -> dict[str, str]:
    """Load manifest of key -> content hash (prefixed keys). From S3 if RAG_MANIFEST_S3_URI set, else local file."""
    s3_uri = _manifest_s3_uri()
    if s3_uri:
        parsed = _parse_s3_uri(s3_uri)
        if parsed:
            bucket, key = parsed
            try:
                s3 = boto3.client("s3", region_name=aws_region)
                resp = s3.get_object(Bucket=bucket, Key=key)
                data = json.loads(resp["Body"].read().decode("utf-8"))
                return data.get("documents", {})
            except Exception as e:
                print(f"Could not load manifest from S3 ({s3_uri}): {e}, using empty.")
                return {}
    if not RAG_MANIFEST_PATH.exists():
        return {}
    data = json.loads(RAG_MANIFEST_PATH.read_text(encoding="utf-8"))
    return data.get("documents", {})


def save_manifest(wanted: dict[str, str], aws_region: str) -> None:
    """Persist manifest after a successful incremental run. To S3 if RAG_MANIFEST_S3_URI set."""
    payload = {
        "source": str(DOCUMENTS_PATH.relative_to(ROOT)),
        "documents": wanted,
    }
    body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    s3_uri = _manifest_s3_uri()
    if s3_uri:
        parsed = _parse_s3_uri(s3_uri)
        if parsed:
            bucket, key = parsed
            try:
                s3 = boto3.client("s3", region_name=aws_region)
                s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"), ContentType="application/json")
                print(f"Saved manifest to s3://{bucket}/{key}")
                return
            except Exception as e:
                print(f"Could not save manifest to S3 ({s3_uri}): {e}, saving locally.")
    RAG_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAG_MANIFEST_PATH.write_text(body, encoding="utf-8")


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

    full_seed = "--full" in sys.argv

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

    if full_seed:
        to_embed = documents
        print(f"Full seed: embedding all {len(documents)} document(s).")
    else:
        manifest = load_manifest(aws_region)
        if manifest:
            # Normal incremental: use manifest hashes to skip unchanged docs
            to_embed = []
            for doc in documents:
                prefixed = f"{KEY_PREFIX}{doc['key']}"
                h = doc_content_hash(doc)
                if prefixed not in manifest or manifest[prefixed] != h:
                    to_embed.append(doc)
            print(f"Incremental: {len(to_embed)} new/changed, {len(documents) - len(to_embed)} unchanged (skip).")
        else:
            # No manifest (e.g. first time or manifest in S3 missing): use index as source of truth.
            # Only embed docs whose key is NOT already in the index (avoids re-embedding everything).
            existing_index_keys = set(
                k for k in list_all_existing_keys(s3vectors, index_arn)
                if k.startswith(KEY_PREFIX)
            )
            to_embed = [doc for doc in documents if f"{KEY_PREFIX}{doc['key']}" not in existing_index_keys]
            print(
                f"No manifest: {len(existing_index_keys)} keys already in index, "
                f"{len(to_embed)} to embed, {len(documents) - len(to_embed)} skipped (already in index)."
            )

    cleanup_stale_vectors(s3vectors, index_arn, wanted_keys)

    vectors = []
    for doc in to_embed:
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

    if vectors:
        print(f"Uploading {len(vectors)} vector(s) to S3 Vectors...")
        for index, batch in enumerate(batched(vectors, UPSERT_BATCH_SIZE), start=1):
            s3vectors.put_vectors(indexArn=index_arn, vectors=batch)
            print(f"Uploaded batch {index} ({len(batch)} vectors).")
    else:
        print("No new or changed vectors to upload.")

    wanted_hashes = {f"{KEY_PREFIX}{doc['key']}": doc_content_hash(doc) for doc in documents}
    save_manifest(wanted_hashes, aws_region)
    print("S3 Vector seeding complete.")


if __name__ == "__main__":
    main()
