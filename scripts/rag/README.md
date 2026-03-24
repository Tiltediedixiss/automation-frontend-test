# GraphRAG: build and seed S3 Vectors

## Incremental updates (recommended)

When `our_components.json` changes, you only need to embed and seed **new or changed** documents. Unchanged documents are skipped to save time and embedding cost.

- **First run** (or after `--full`): embeds all documents and writes a manifest (key → content hash).
- **Next runs**: loads manifest, diffs current documents by hash, embeds only new/changed, deletes vectors for removed keys, updates manifest.

### Commands

From repo root:

```bash
# 1) Build graphrag-documents.json from our_components.json (in this repo)
python -m scripts.rag.build_rag_documents

# 2) Incremental seed to S3 Vectors (only new/changed)
python -m scripts.rag.seed_s3_vectors

# Or one shot: build + incremental seed
python -m scripts.rag.update_rag
```

With a components file from another path (e.g. frontend repo):

```bash
python -m scripts.rag.update_rag /path/to/frontend/our_components.json
```

Full re-seed (ignore manifest, embed everything):

```bash
python -m scripts.rag.seed_s3_vectors --full
python -m scripts.rag.update_rag --full
```

### Manifest storage

- **Local (default)**: `generated/rag-manifest.json`. Use when you always run from the same machine or same CI workspace and can persist the file.
- **S3 (CI / multi-runner)**: set `RAG_MANIFEST_S3_URI=s3://your-bucket/rag-manifest.json`. The script will load/save the manifest from S3 so incremental works across runs and machines. The IAM user or role must have `s3:GetObject` and `s3:PutObject` on that object.

## Triggering from the frontend repo (TypeScript)

Your frontend repo can trigger an RAG update whenever `our_components.json` is updated in two ways.

### Option A: GitHub Action in frontend repo

1. In the **frontend** repo, add a workflow that runs when `our_components.json` (or the script that generates it) changes.
2. Either:
   - **Check out this automation repo** and run Python there (requires Python + deps + AWS/Google secrets in the frontend repo), or
   - **Call a webhook** that runs the update (e.g. a small endpoint on Render/ECS that runs `update_rag` and is secured by a secret).

Example (frontend repo) — run update in automation repo:

```yaml
# .github/workflows/update-rag.yml
name: Update RAG
on:
  push:
    paths:
      - 'our_components.json'   # or path that generates it
jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          repository: YOUR_ORG/automation-frontend-test
          path: automation
          token: ${{ secrets.GH_PAT }}
      - uses: actions/checkout@v4
        with:
          path: frontend
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r automation/requirements.txt
      - run: python -m scripts.rag.update_rag ../frontend/our_components.json
        working-directory: automation
        env:
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
          AWS_S3_VECTOR_INDEX_ARN: ${{ secrets.AWS_S3_VECTOR_INDEX_ARN }}
          AWS_REGION: ${{ secrets.AWS_REGION }}
          RAG_MANIFEST_S3_URI: ${{ secrets.RAG_MANIFEST_S3_URI }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

Store the manifest in S3 (`RAG_MANIFEST_S3_URI`) so the next run is incremental.

### Option B: Webhook from frontend

1. Add an endpoint in this repo (e.g. `POST /update-rag`) that:
   - Verifies a shared secret (e.g. header or body).
   - Optionally accepts `our_components.json` in the request or a URL to fetch it.
   - Runs `build_rag_documents` + `seed_s3_vectors` (e.g. in a background task or subprocess).
2. From the frontend repo, after generating or committing `our_components.json`, call that endpoint (e.g. from GitHub Action with `curl` or from your deploy pipeline).

This keeps AWS/Google credentials only in the automation service, not in the frontend repo.

## Environment

- `GOOGLE_API_KEY` – used for embeddings (Gemini).
- `AWS_S3_VECTOR_INDEX_ARN` – S3 Vectors index ARN.
- `AWS_REGION` – e.g. `eu-central-1`.
- `RAG_MANIFEST_S3_URI` (optional) – e.g. `s3://front-automation/rag-manifest.json` for CI/manifest persistence.
