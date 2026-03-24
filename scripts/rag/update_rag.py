#!/usr/bin/env python3
"""
Run RAG update: build documents from our_components.json, then incremental seed to S3 Vectors.

Usage:
  # From repo root (uses our_components.json in repo root)
  python -m scripts.rag.update_rag

  # With components file from another path (e.g. frontend repo)
  python -m scripts.rag.update_rag /path/to/frontend/our_components.json

  # Full re-seed (embed everything, ignore manifest)
  python -m scripts.rag.update_rag --full
  python -m scripts.rag.update_rag /path/to/our_components.json --full

Requires: .env with GOOGLE_API_KEY, AWS_S3_VECTOR_INDEX_ARN, AWS_REGION; AWS credentials.
Optional: RAG_MANIFEST_S3_URI=s3://bucket/key to store manifest in S3 for CI.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "scripts" / "rag" / "build_rag_documents.py"
SEED_SCRIPT = ROOT / "scripts" / "rag" / "seed_s3_vectors.py"


def main() -> None:
    argv = sys.argv[1:]
    full = "--full" in argv
    if full:
        argv = [a for a in argv if a != "--full"]

    components_path = argv[0] if argv else None

    # 1) Build graphrag-documents.json from our_components.json
    build_cmd = [sys.executable, str(BUILD_SCRIPT)]
    if components_path:
        build_cmd.append(components_path)
    print("Running:", " ".join(build_cmd))
    r = subprocess.run(build_cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

    # 2) Incremental seed (or full if --full)
    seed_cmd = [sys.executable, str(SEED_SCRIPT)]
    if full:
        seed_cmd.append("--full")
    print("Running:", " ".join(seed_cmd))
    r = subprocess.run(seed_cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

    print("RAG update finished.")


if __name__ == "__main__":
    main()
