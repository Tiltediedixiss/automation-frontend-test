"""
pipeline.py
-----------
Core pipeline: S3 key in → result dict out.

All functions are copied verbatim from test_twelvelabs.py.
The only changes:
  - no local file writes (those were only for debugging)
  - video is downloaded from S3 instead of read from disk
  - returns a plain dict instead of writing .md / .json files
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import boto3
from dotenv import load_dotenv
from twelvelabs import ResponseFormat, TwelveLabs

load_dotenv()

# ---------------------------------------------------------------------------
# Config  (from environment — same names as your .env)
# ---------------------------------------------------------------------------
TWELVELABS_KEY    = os.getenv("TWELVELABS_API_KEY", "").strip().strip("\"'")
GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY", "").strip().strip("\"'")
INDEX_ARN         = os.getenv("AWS_S3_VECTOR_INDEX_ARN", "").strip().strip("\"'")
AWS_REGION        = os.getenv("AWS_REGION", "eu-central-1")
GRAPHRAG_DOCS     = Path(__file__).resolve().parent / "generated" / "graphrag-documents.json"
INDEX_NAME_PREFIX = "cursor-feature-requests"
MAX_UPLOAD_MB     = 200

if not TWELVELABS_KEY:
    raise RuntimeError("Missing TWELVELABS_API_KEY in environment/.env")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment/.env")
if not INDEX_ARN:
    raise RuntimeError("Missing AWS_S3_VECTOR_INDEX_ARN in environment/.env")

_s3     = boto3.client("s3", region_name=AWS_REGION)
client  = TwelveLabs(api_key=TWELVELABS_KEY)


# ---------------------------------------------------------------------------
# Helpers — identical to test_twelvelabs.py
# ---------------------------------------------------------------------------

def _get_duration_seconds(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr or out.stdout}")
    return float(out.stdout.strip())


def _compress_if_needed(path: str, max_mb: float = MAX_UPLOAD_MB) -> str:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb <= max_mb * 0.95:
        return path
    duration     = _get_duration_seconds(path)
    target_bytes = int((max_mb * 0.9) * 1024 * 1024)
    target_kbps  = int((target_bytes * 8) / duration / 1000)
    target_kbps  = max(400, min(target_kbps, 5000))
    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        rc = subprocess.run(
            [
                "ffmpeg", "-y", "-i", path,
                "-c:v", "libx264", "-b:v", f"{target_kbps}k",
                "-maxrate", f"{int(target_kbps * 1.2)}k",
                "-bufsize", f"{int(target_kbps * 2)}k",
                "-vf", "scale=-2:720",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                out_path,
            ],
            capture_output=True, text=True,
        )
        if rc.returncode != 0:
            os.unlink(out_path)
            raise RuntimeError(f"ffmpeg compress failed: {rc.stderr or rc.stdout}")
        new_mb = os.path.getsize(out_path) / (1024 * 1024)
        if new_mb > max_mb:
            os.unlink(out_path)
            raise RuntimeError(f"Compressed video still {new_mb:.1f} MB (max {max_mb} MB).")
        return out_path
    except Exception:
        try:
            os.unlink(out_path)
        except FileNotFoundError:
            pass
        raise


def _transcribe_with_google(video_path: str) -> str:
    from google import genai
    from google.genai import types

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        print("  Extracting audio to 16 kHz mono WAV...")
        rc = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-vn", wav_path],
            capture_output=True, text=True,
        )
        if rc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {rc.stderr or rc.stdout}")
        audio_bytes = Path(wav_path).read_bytes()
        ai = genai.Client(api_key=GOOGLE_API_KEY.strip())
        print("  Transcribing with Gemini...")
        response = ai.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                (
                    "Transcribe this audio verbatim. "
                    "The spoken language is Russian. "
                    "Return the transcript only in Russian Cyrillic. "
                    "Do not translate. Do not summarize. Do not infer missing content. "
                    "Preserve wording as spoken as closely as possible."
                ),
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            ],
        )
        return (response.text or "").strip()
    finally:
        try:
            os.unlink(wav_path)
        except FileNotFoundError:
            pass


def _create_fresh_index():
    index_name = f"{INDEX_NAME_PREFIX}-{int(time.time())}"
    print(f"--- Step 1: Creating fresh index: {index_name} ---")
    index = client.indexes.create(
        index_name=index_name,
        models=[
            {"model_name": "marengo3.0", "model_options": ["visual", "audio"]},
            {"model_name": "pegasus1.2", "model_options": ["visual", "audio"]},
        ],
        addons=["thumbnail"],
    )
    print(f"✅ Index created: {index.id}")
    return index


def _fetch_twelvelabs_artifacts(index_id: str, indexed_asset_id: str):
    details = client.indexes.indexed_assets.retrieve(
        index_id, indexed_asset_id, transcription=True,
    )
    tl_transcript = ""
    if details.transcription:
        tl_transcript = " ".join(
            item.value.strip()
            for item in details.transcription
            if item.value and item.value.strip()
        ).strip()

    thumbnail_urls: list[str] = []
    if details.hls and details.hls.thumbnail_urls:
        thumbnail_urls = list(details.hls.thumbnail_urls)

    return tl_transcript, thumbnail_urls


def _identify_relevant_moments(video_id: str, transcript: str) -> list[dict]:
    response = client.analyze(
        video_id=video_id,
        prompt=f"""You are analyzing a product feature-request video for Alims.

Goal: identify only the moments that are relevant to the PM's requested change.

The transcript below is the primary source of truth:
---
{transcript or "(no transcript available)"}
---

Return 1 to 5 key moments where the requested change is visually relevant on screen.
Only include moments that help developers understand the requested change.
Do not include generic dashboards or unrelated screens unless they are necessary context.
For each moment, provide:
- timestamp_sec: the best second in the video to capture a screenshot
- reason: short explanation of why this frame is useful
""",
        temperature=0,
        response_format=ResponseFormat(
            type="json_schema",
            json_schema={
                "type": "object",
                "properties": {
                    "moments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "timestamp_sec": {"type": "number"},
                                "reason":        {"type": "string"},
                            },
                            "required": ["timestamp_sec", "reason"],
                        },
                        "minItems": 1,
                    }
                },
                "required": ["moments"],
            },
        ),
        max_tokens=600,
    )
    raw = response.data or '{"moments":[]}'
    return json.loads(raw).get("moments") or []


def _extract_relevant_screenshots(video_path: str, moments: list[dict]) -> list[str]:
    out_dir = Path(tempfile.mkdtemp())
    paths: list[str] = []
    for idx, moment in enumerate(moments, start=1):
        timestamp   = float(moment.get("timestamp_sec", 0))
        output_path = str(out_dir / f"shot_{idx:02d}_{timestamp:.2f}s.jpg")
        rc = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", video_path,
             "-frames:v", "1", "-q:v", "2", output_path],
            capture_output=True, text=True,
        )
        if rc.returncode == 0 and os.path.exists(output_path):
            paths.append(output_path)
    return paths


def _extract_first_order_spec(video_id: str, transcript: str, tl_transcript: str) -> dict:
    response = client.analyze(
        video_id=video_id,
        prompt=f"""You are extracting a complete implementation spec from a product feature-request video for Alims.

Alims has three existing modules: teacher, student, principal. Do NOT re-describe their baseline behavior.
However, if the PM is requesting a change, fix, or addition TO any existing module, screen, or component — include it. Modifications to existing surfaces are in scope.

Your goal is HIGH RECALL. It is better to include something borderline than to miss a real request.

PRIMARY TRANSCRIPT — treat this as ground truth for what the PM said:
---
{transcript or "(no transcript available)"}
---

SECONDARY TRANSCRIPT from Twelve Labs — use only for screen/UI disambiguation:
---
{tl_transcript or "(not available)"}
---

Extract the following and be exhaustive:

- summary: A complete paragraph describing everything the PM wants changed or added. Do not shorten.
- search_query: A dense retrieval query covering all topics mentioned (components, routes, behaviors).
- requested_changes: Every distinct change, fix, or new behavior the PM requested — one item per action. Do not merge unrelated changes into one bullet. Do not omit any.
- affected_surfaces: Every route, screen, component, or module the PM mentions or that is clearly shown while they are describing a change.

Do not invent requirements. Do not omit requirements the PM stated.""",
        temperature=0.35,
        response_format=ResponseFormat(
            type="json_schema",
            json_schema={
                "type": "object",
                "properties": {
                    "summary":           {"type": "string"},
                    "search_query":      {"type": "string"},
                    "requested_changes": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "affected_surfaces": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "search_query", "requested_changes", "affected_surfaces"],
            },
        ),
        max_tokens=4096,
    )
    return json.loads(response.data or "{}")


def _retrieve_rag_context(spec: dict) -> dict:
    if not GRAPHRAG_DOCS.exists():
        raise FileNotFoundError(
            f"GraphRAG documents file not found: {GRAPHRAG_DOCS} (resolved from {Path(__file__).resolve()}). "
            "Ensure generated/graphrag-documents.json exists and is included in the Docker image."
        )
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from query_rag import GraphRAGRetriever, infer_region  # noqa: PLC0415

    aws_region = AWS_REGION or infer_region(INDEX_ARN)
    retriever  = GraphRAGRetriever(
        docs_path=GRAPHRAG_DOCS,
        index_arn=INDEX_ARN,
        aws_region=aws_region,
        google_api_key=GOOGLE_API_KEY,
    )
    query_text = "\n".join([
        spec.get("search_query", ""),
        spec.get("summary", ""),
        *spec.get("requested_changes", []),
        *spec.get("affected_surfaces", []),
    ]).strip()
    return retriever.retrieve(query_text, top_k=24, max_components=5, max_relations=6, max_bundles=2)


def _format_rag_context(result: dict) -> str:
    lines: list[str] = []

    primary_components = result.get("primary", {}).get("components", [])
    if primary_components:
        lines.append("Primary relevant components:")
        for item in primary_components[:5]:
            doc        = item["doc"]
            meta       = doc.get("metadata", {})
            text_lines = doc.get("text", "").splitlines()
            summary    = text_lines[5] if len(text_lines) > 5 else doc.get("text", "")
            lines.append(
                f"- {meta.get('sourcePath')} | featureArea={meta.get('featureArea')} | "
                f"kind={meta.get('componentKind')} | summary={summary}"
            )

    expanded_components = result.get("expanded", {}).get("components", [])
    if expanded_components:
        lines.append("\nExpanded neighboring components:")
        for doc in expanded_components[:4]:
            meta = doc.get("metadata", {})
            lines.append(
                f"- {meta.get('sourcePath')} | featureArea={meta.get('featureArea')} | "
                f"kind={meta.get('componentKind')}"
            )

    relations = result.get("expanded", {}).get("relations", [])
    if relations:
        lines.append("\nRelevant relations:")
        for doc in relations[:6]:
            meta = doc.get("metadata", {})
            lines.append(f"- {meta.get('sourcePath')} -> {meta.get('targetPath')}")

    bundles = (
        result.get("primary", {}).get("bundles", [])
        + result.get("expanded", {}).get("bundles", [])
    )
    seen: set[str] = set()
    bundle_lines: list[str] = []
    for item in bundles:
        doc = item["doc"] if isinstance(item, dict) and "doc" in item else item
        if doc["key"] in seen:
            continue
        seen.add(doc["key"])
        meta = doc.get("metadata", {})
        bundle_lines.append(
            f"- {doc['key']} | featureArea={meta.get('featureArea')} | "
            f"role={meta.get('role')} | members={meta.get('memberCount')}"
        )
    if bundle_lines:
        lines.append("\nFeature bundles:")
        lines.extend(bundle_lines[:2])

    return "\n".join(lines).strip()


def _extract_file_paths(rag_context: dict) -> list[str]:
    """
    Pull unique file paths from RAG results.
    sourcePath is like "src/components/student/folder/page.tsx#StudentFolder"
    — strip the #Symbol suffix to get the plain file path.
    """
    seen:  set[str]  = set()
    paths: list[str] = []
    all_components = (
        rag_context.get("primary", {}).get("components", [])
        + rag_context.get("expanded", {}).get("components", [])
    )
    for item in all_components:
        doc      = item.get("doc", item)
        raw      = doc.get("metadata", {}).get("sourcePath", "")
        cleaned  = raw.split("#")[0] if raw else ""
        if cleaned and cleaned not in seen:
            paths.append(cleaned)
            seen.add(cleaned)
    return paths


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(s3_bucket: str, s3_key: str, task_id: str = "") -> dict:
    """
    Download video from S3, run the full pipeline, return result dict.

    Returns:
        {
            "task_id":      str,
            "summary":      str,
            "files":        list[str],
            "instructions": str,
        }

    Raises on unrecoverable errors so the caller (API or CLI) can handle them.
    """
    label = task_id or s3_key
    video_path: str | None = None
    to_upload:  str | None = None

    try:
        # ── Download from S3 ────────────────────────────────────────────
        print(f"[{label}] Downloading s3://{s3_bucket}/{s3_key}")
        fd, video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        _s3.download_file(s3_bucket, s3_key, video_path)
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"[{label}] Downloaded: {size_mb:.1f} MB")

        # ── Compress if needed ──────────────────────────────────────────
        to_upload = _compress_if_needed(video_path)

        # ── Step 1: Create TwelveLabs index ────────────────────────────
        index = _create_fresh_index()

        # ── Step 2: Upload as asset ─────────────────────────────────────
        print(f"[{label}] Uploading to TwelveLabs...")
        with open(to_upload, "rb") as f:
            asset = client.assets.create(method="direct", file=f)
        print(f"[{label}] Asset: {asset.id}")

        if to_upload != video_path:
            try:
                os.unlink(to_upload)
                to_upload = None
            except FileNotFoundError:
                pass

        # ── Step 3: Index ───────────────────────────────────────────────
        indexed_asset = client.indexes.indexed_assets.create(
            index_id=index.id, asset_id=asset.id, enable_video_stream=True,
        )
        print(f"[{label}] Indexing (indexed_asset_id={indexed_asset.id})...")
        while True:
            indexed_asset = client.indexes.indexed_assets.retrieve(index.id, indexed_asset.id)
            print(f"[{label}]   status: {indexed_asset.status}")
            if indexed_asset.status == "ready":
                break
            if indexed_asset.status == "failed":
                raise RuntimeError("TwelveLabs indexing failed.")
            time.sleep(5)

        # ── Step 3.5: TwelveLabs artifacts ──────────────────────────────
        tl_transcript, thumbnail_urls = _fetch_twelvelabs_artifacts(index.id, indexed_asset.id)
        print(f"[{label}] TL transcript: {len(tl_transcript)} chars, thumbnails: {len(thumbnail_urls)}")

        # ── Step 4: Gemini transcription ────────────────────────────────
        print(f"\n[{label}] --- Step 4: Transcribing with Google Gemini ---")
        try:
            transcript = _transcribe_with_google(video_path)
            print(f"[{label}] Transcript: {len(transcript)} chars")
        except Exception as exc:
            print(f"[{label}] ⚠ Gemini transcription failed: {exc}")
            transcript = ""

        # ── Step 4.5: Screenshots ───────────────────────────────────────
        print(f"\n[{label}] --- Step 4.5: Relevant timestamps + screenshots ---")
        try:
            moments          = _identify_relevant_moments(indexed_asset.id, transcript)
            screenshot_paths = _extract_relevant_screenshots(video_path, moments)
            print(f"[{label}] Screenshots: {len(screenshot_paths)}")
        except Exception as exc:
            print(f"[{label}] ⚠ Screenshots failed: {exc}")
            moments, screenshot_paths = [], []

        # ── Step 4.6: First-order spec ──────────────────────────────────
        print(f"\n[{label}] --- Step 4.6: Extracting first-order spec ---")
        try:
            spec = _extract_first_order_spec(indexed_asset.id, transcript, tl_transcript)
            print(f"[{label}] Spec summary: {spec.get('summary', '')[:80]}...")
        except Exception as exc:
            print(f"[{label}] ⚠ Spec extraction failed: {exc}")
            spec = {
                "summary":           "",
                "search_query":      transcript,
                "requested_changes": [transcript] if transcript else [],
                "affected_surfaces": [],
            }

        # ── Step 4.7: GraphRAG ──────────────────────────────────────────
        print(f"\n[{label}] --- Step 4.7: Querying GraphRAG ---")
        try:
            rag_context      = _retrieve_rag_context(spec)
            rag_context_text = _format_rag_context(rag_context)
            resolved_files   = _extract_file_paths(rag_context)
            print(f"[{label}] RAG resolved {len(resolved_files)} file(s)")
        except Exception as exc:
            import traceback
            print(f"[{label}] ⚠ GraphRAG failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            rag_context, rag_context_text, resolved_files = {}, "(not available)", []

        # ── Step 5: Generate Cursor instructions ────────────────────────
        print(f"\n[{label}] --- Step 5: Generating Cursor instructions ---")
        pm_prompt = f"""Context: The product is Alims. It already has a ready UI/UX and these modules: teacher, student, principal. Do NOT describe or list these existing features.

Your job: Extract ONLY the new or changed requirements that the PM (product manager) is asking for in this feature-request video.

Source priority:
1. The FIRST-ORDER SPEC below is the primary source of truth for requested changes.
2. The GraphRAG context below tells you what is already implemented and which files/components are likely relevant.
3. The Google Gemini transcript is supporting evidence.
4. The video understanding from Twelve Labs is only for screen/module disambiguation.
5. Ignore any UI, flow, or module that is simply part of the already existing Alims product unless the PM explicitly asks to change it.

Filtering rules:
- Include only requested changes, bugs to fix, or new behaviors to implement.
- If something is visible in the video but not requested by the PM, omit it.
- If something is uncertain or only inferred from visuals, omit it.
- Do not restate baseline UI/UX, teacher/student/principal modules, or generic product descriptions.

FIRST-ORDER SPEC:
---
{json.dumps(spec, ensure_ascii=False, indent=2)}
---

GRAPHRAG CONTEXT:
---
{rag_context_text}
---

TRANSCRIPT of what the PM said (from Google Gemini STT):
---
{transcript or "(no transcript available)"}
---

OPTIONAL SUPPORTING CONTEXT from Twelve Labs:
Twelve Labs transcript:
---
{tl_transcript or "(not available)"}
---

Thumbnail URLs:
---
{chr(10).join(thumbnail_urls[:12]) if thumbnail_urls else "(not available)"}
---

Relevant screenshots extracted from the source video:
---
{chr(10).join(screenshot_paths) if screenshot_paths else "(not available)"}
---

Instructions: Write a detailed, Cursor-ready implementation brief for developers that contains:
- requested changes only
- likely existing files/components to inspect first (reference them as @filepath)
- implementation constraints inferred from current code
- a short acceptance checklist

Use concise bullet points. Do not include generic UI/UX or module descriptions. Output ONLY the brief text—no preamble or summary."""

        final_chunks: list[str] = []
        text_stream = client.analyze_stream(video_id=indexed_asset.id, prompt=pm_prompt)
        for chunk in text_stream:
            if chunk.event_type == "text_generation":
                print(chunk.text, end="", flush=True)
                final_chunks.append(chunk.text)
        print()  # newline after stream

        instructions = "".join(final_chunks).strip()

        if resolved_files:
            instructions += "\n\n## Files to open\n\n"
            instructions += "\n".join(f"@{p}" for p in resolved_files)

        print(f"\n[{label}] ✅ Done.")

        return {
            "task_id":      task_id,
            "summary":      spec.get("summary", ""),
            "files":        resolved_files,
            "instructions": instructions,
        }

    finally:
        for p in {video_path, to_upload} - {None}:
            try:
                os.unlink(p)  # type: ignore[arg-type]
            except FileNotFoundError:
                pass