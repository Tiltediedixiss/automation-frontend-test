import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from twelvelabs import ResponseFormat, TwelveLabs
from query_rag import GraphRAGRetriever, infer_region

# Load .env for local API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Initialize the client (replace with your actual API key)
API_KEY = os.environ.get("TWELVELABS_API_KEY")
client = TwelveLabs(api_key=API_KEY)

video_file_path = "test_easy.mp4"
MAX_UPLOAD_MB = 200  # Twelve Labs limit
GOOGLE_STT_LANGUAGE = "ru"
TRANSCRIPT_OUTPUT_PATH = "transcript_ru.txt"
TWELVELABS_TRANSCRIPT_OUTPUT_PATH = "twelvelabs_transcript.txt"
THUMBNAIL_URLS_OUTPUT_PATH = "thumbnail_urls.txt"
TIMESTAMPS_OUTPUT_PATH = "relevant_timestamps.json"
SCREENSHOTS_DIR = "relevant_screenshots"
FIRST_ORDER_SPEC_PATH = "first_order_spec.json"
RAG_CONTEXT_PATH = "rag_context.json"
FINAL_INSTRUCTIONS_PATH = "final_cursor_instructions.md"
INDEX_NAME_PREFIX = "cursor-feature-requests"
CREATE_FRESH_INDEX = False


def _get_duration_seconds(path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr or out.stdout}")
    return float(out.stdout.strip())


def _compress_if_needed(path: str, max_mb: float = MAX_UPLOAD_MB) -> str:
    """Return path to use: original if small enough, else a compressed temp file under max_mb."""
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb <= max_mb * 0.95:
        return path
    duration = _get_duration_seconds(path)
    target_bytes = int((max_mb * 0.9) * 1024 * 1024)
    target_kbps = int((target_bytes * 8) / duration / 1000)
    target_kbps = max(400, min(target_kbps, 5000))
    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        cmd = [
            "ffmpeg", "-y", "-i", path,
            "-c:v", "libx264", "-b:v", f"{target_kbps}k",
            "-maxrate", f"{int(target_kbps * 1.2)}k", "-bufsize", f"{int(target_kbps * 2)}k",
            "-vf", "scale=-2:720",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            out_path
        ]
        rc = subprocess.run(cmd, capture_output=True, text=True)
        if rc.returncode != 0:
            os.unlink(out_path)
            raise RuntimeError(f"ffmpeg compress failed: {rc.stderr or rc.stdout}")
        new_mb = os.path.getsize(out_path) / (1024 * 1024)
        if new_mb > max_mb:
            os.unlink(out_path)
            raise RuntimeError(f"Compressed video still {new_mb:.1f} MB (max {max_mb} MB). Try a shorter or lower-resolution source.")
        return out_path
    except Exception:
        try:
            os.unlink(out_path)
        except FileNotFoundError:
            pass
        raise


def _transcribe_with_google(video_path: str) -> str:
    """Transcribe video speech using Gemini with GOOGLE_API_KEY from .env."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError("Set GOOGLE_API_KEY in .env to your Google AI Studio API key.")
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("Install: pip install google-genai")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        print("  Extracting audio to 16 kHz mono WAV...")
        rc = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                "-vn", wav_path
            ],
            capture_output=True,
            text=True,
        )
        if rc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {rc.stderr or rc.stdout}")
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        client_stt = genai.Client(api_key=api_key.strip())
        print("  Transcribing with Gemini...")
        response = client_stt.models.generate_content(
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


def _create_or_get_index():
    """Prefer a fresh hybrid index so model settings match the current script."""
    if CREATE_FRESH_INDEX:
        index_name = f"{INDEX_NAME_PREFIX}-{int(time.time())}"
        print("--- Step 1: Creating a Fresh Analysis Index ---")
        index = client.indexes.create(
            index_name=index_name,
            models=[
                {
                    "model_name": "marengo3.0",
                    "model_options": ["visual", "audio"],
                },
                {
                    "model_name": "pegasus1.2",
                    "model_options": ["visual", "audio"],
                },
            ],
            addons=["thumbnail"],
        )
        print(f"✅ Fresh index created: {index.id} ({index_name})")
        return index

    index_name = INDEX_NAME_PREFIX
    print("--- Step 1: Getting or Creating an Index ---")
    existing = None
    for idx in client.indexes.list():
        if idx.index_name == index_name:
            existing = idx
            break
    if existing:
        print(f"✅ Using existing index: {existing.id}")
        return existing
    index = client.indexes.create(
        index_name=index_name,
        models=[
            {
                "model_name": "marengo3.0",
                "model_options": ["visual", "audio"],
            },
            {
                "model_name": "pegasus1.2",
                "model_options": ["visual", "audio"],
            },
        ],
        addons=["thumbnail"],
    )
    print(f"✅ Index created: {index.id}")
    return index


def _fetch_twelvelabs_artifacts(index_id: str, indexed_asset_id: str):
    """Retrieve Twelve Labs transcription and thumbnail URLs for debugging/inspection."""
    details = client.indexes.indexed_assets.retrieve(
        index_id,
        indexed_asset_id,
        transcription=True,
    )

    tl_transcript = ""
    if details.transcription:
        tl_transcript = " ".join(
            item.value.strip() for item in details.transcription if item.value and item.value.strip()
        ).strip()
        if tl_transcript:
            with open(TWELVELABS_TRANSCRIPT_OUTPUT_PATH, "w", encoding="utf-8") as f:
                f.write(tl_transcript)

    thumbnail_urls = []
    if details.hls and details.hls.thumbnail_urls:
        thumbnail_urls = list(details.hls.thumbnail_urls)
        with open(THUMBNAIL_URLS_OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(thumbnail_urls))

    return tl_transcript, thumbnail_urls


def _identify_relevant_moments(video_id: str, transcript: str):
    """Ask Twelve Labs for the most relevant timestamps for the PM request."""
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
                                "reason": {"type": "string"},
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
    parsed = json.loads(raw)
    moments = parsed.get("moments") or []
    with open(TIMESTAMPS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"moments": moments}, f, ensure_ascii=False, indent=2)
    return moments


def _extract_first_order_spec(video_id: str, transcript: str, tl_transcript: str):
    """Use Twelve Labs to extract only the requested implementation deltas from the video."""
    response = client.analyze(
        video_id=video_id,
        prompt=f"""You are extracting a FIRST-ORDER implementation spec from a feature-request video for Alims.

The product already has existing UI/UX and modules. Ignore baseline product structure and extract only what the PM is asking to change or add.

Primary transcript (authoritative):
---
{transcript or "(no transcript available)"}
---

Secondary transcript from Twelve Labs (supporting only):
---
{tl_transcript or "(not available)"}
---

Return a compact structured spec with:
- summary: one short paragraph of the requested change
- search_query: one retrieval-friendly query for GraphRAG
- requested_changes: a list of concrete changes to implement
- affected_surfaces: routes, screens, modules, or components mentioned or visually implied with high confidence

Only include changes actually requested by the PM. Omit generic UI descriptions and existing product capabilities.""",
        temperature=0,
        response_format=ResponseFormat(
            type="json_schema",
            json_schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "search_query": {"type": "string"},
                    "requested_changes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "affected_surfaces": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 0,
                    },
                },
                "required": ["summary", "search_query", "requested_changes", "affected_surfaces"],
            },
        ),
        max_tokens=900,
    )
    raw = response.data or "{}"
    spec = json.loads(raw)
    with open(FIRST_ORDER_SPEC_PATH, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    return spec


def _build_rag_retriever() -> GraphRAGRetriever:
    google_api_key = os.environ.get("GOOGLE_API_KEY", "").strip().strip("\"'")
    index_arn = os.environ.get("AWS_S3_VECTOR_INDEX_ARN", "").strip().strip("\"'")
    if not google_api_key:
        raise RuntimeError("Set GOOGLE_API_KEY in .env to query GraphRAG.")
    if not index_arn:
        raise RuntimeError("Set AWS_S3_VECTOR_INDEX_ARN in .env to query GraphRAG.")
    aws_region = os.environ.get("AWS_REGION", "").strip() or infer_region(index_arn)
    if not aws_region:
        raise RuntimeError("Missing AWS_REGION and could not infer it from AWS_S3_VECTOR_INDEX_ARN.")
    return GraphRAGRetriever(
        docs_path=Path("generated/graphrag-documents.json"),
        index_arn=index_arn,
        aws_region=aws_region,
        google_api_key=google_api_key,
    )


def _retrieve_rag_context(spec: dict):
    retriever = _build_rag_retriever()
    query_text = "\n".join(
        [
            spec.get("search_query", ""),
            spec.get("summary", ""),
            *spec.get("requested_changes", []),
            *spec.get("affected_surfaces", []),
        ]
    ).strip()
    result = retriever.retrieve(
        query_text,
        top_k=24,
        max_components=5,
        max_relations=6,
        max_bundles=2,
    )
    with open(RAG_CONTEXT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def _format_rag_context(result: dict) -> str:
    lines = []

    primary_components = result.get("primary", {}).get("components", [])
    if primary_components:
        lines.append("Primary relevant components:")
        for item in primary_components[:5]:
            doc = item["doc"]
            meta = doc.get("metadata", {})
            lines.append(
                f"- {meta.get('sourcePath')} | featureArea={meta.get('featureArea')} | "
                f"kind={meta.get('componentKind')} | summary={doc.get('text', '').splitlines()[5] if len(doc.get('text', '').splitlines()) > 5 else doc.get('text', '')}"
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

    bundles = result.get("primary", {}).get("bundles", []) + result.get("expanded", {}).get("bundles", [])
    seen_bundle_keys = set()
    bundle_lines = []
    for item in bundles:
        doc = item["doc"] if isinstance(item, dict) and "doc" in item else item
        if doc["key"] in seen_bundle_keys:
            continue
        seen_bundle_keys.add(doc["key"])
        meta = doc.get("metadata", {})
        bundle_lines.append(
            f"- {doc['key']} | featureArea={meta.get('featureArea')} | role={meta.get('role')} | members={meta.get('memberCount')}"
        )
    if bundle_lines:
        lines.append("\nFeature bundles:")
        lines.extend(bundle_lines[:2])

    return "\n".join(lines).strip()


def _extract_relevant_screenshots(video_path: str, moments):
    """Extract full-resolution screenshots from the local video using ffmpeg."""
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    screenshot_paths = []
    for idx, moment in enumerate(moments, start=1):
        timestamp = float(moment.get("timestamp_sec", 0))
        # Seek to the requested moment and save one full-res frame.
        output_path = os.path.join(SCREENSHOTS_DIR, f"shot_{idx:02d}_{timestamp:.2f}s.jpg")
        rc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-q:v",
                "2",
                output_path,
            ],
            capture_output=True,
            text=True,
        )
        if rc.returncode == 0 and os.path.exists(output_path):
            screenshot_paths.append(output_path)
    return screenshot_paths


def test_twelvelabs_pipeline():
    if not os.path.exists(video_file_path):
        print(f"Error: Please place '{video_file_path}' in this directory.")
        return

    # Check file size before any API calls; compress if over limit
    size_mb = os.path.getsize(video_file_path) / (1024 * 1024)
    print(f"Video file: {size_mb:.1f} MB (upload limit: {MAX_UPLOAD_MB} MB)")
    if size_mb > MAX_UPLOAD_MB:
        print("  Compressing to fit limit...")
        to_upload = _compress_if_needed(video_file_path)
        print(f"  Ready: {os.path.getsize(to_upload) / (1024*1024):.1f} MB")
    else:
        to_upload = video_file_path

    index = _create_or_get_index()

    print("\n--- Step 2: Uploading Video as an Asset ---")
    try:
        with open(to_upload, "rb") as f:
            asset = client.assets.create(
                method="direct",
                file=f
            )
    finally:
        if to_upload != video_file_path:
            try:
                os.unlink(to_upload)
            except FileNotFoundError:
                pass
    print(f"✅ Asset created: {asset.id}")

    print("\n--- Step 3: Indexing the Asset ---")
    indexed_asset = client.indexes.indexed_assets.create(
        index_id=index.id,
        asset_id=asset.id,
        enable_video_stream=True,
    )
    print(f"Indexing video (Indexed Asset ID: {indexed_asset.id})...")
    
    # Monitor the indexing status
    while True:
        indexed_asset = client.indexes.indexed_assets.retrieve(
            index.id,
            indexed_asset.id
        )
        print(f"  Status: {indexed_asset.status}")
        
        if indexed_asset.status == "ready":
            print("✅ Video indexing complete!")
            break
        elif indexed_asset.status == "failed":
            raise RuntimeError("Indexing failed. Check your file or API limits.")
            
        time.sleep(5)

    print("\n--- Step 3.5: Retrieving Twelve Labs artifacts ---")
    tl_transcript, thumbnail_urls = _fetch_twelvelabs_artifacts(index.id, indexed_asset.id)
    if tl_transcript:
        print(f"  Twelve Labs transcript saved to {TWELVELABS_TRANSCRIPT_OUTPUT_PATH}")
    else:
        print("  No Twelve Labs transcript returned")
    if thumbnail_urls:
        print(f"  Thumbnail URLs saved to {THUMBNAIL_URLS_OUTPUT_PATH}")
        print(f"  Retrieved {len(thumbnail_urls)} thumbnails")
    else:
        print("  No thumbnail URLs returned yet")

    print("\n--- Step 4: Transcribing speech with Google Gemini ---")
    try:
        transcript = _transcribe_with_google(video_file_path)
        if transcript:
            print(f"  Transcript length: {len(transcript)} chars")
            with open(TRANSCRIPT_OUTPUT_PATH, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"  Transcript saved to {TRANSCRIPT_OUTPUT_PATH}")
            print("\n========================================")
            print("RAW TRANSCRIPT")
            print("========================================")
            print(transcript)
            print("========================================")
        else:
            print("  (No speech detected)")
    except Exception as e:
        print(f"  ⚠ Gemini transcription failed: {e}")
        transcript = ""

    print("\n--- Step 4.5: Finding relevant timestamps and extracting screenshots ---")
    try:
        moments = _identify_relevant_moments(indexed_asset.id, transcript)
        if moments:
            print(f"  Relevant timestamps saved to {TIMESTAMPS_OUTPUT_PATH}")
            screenshot_paths = _extract_relevant_screenshots(video_file_path, moments)
            if screenshot_paths:
                print(f"  Extracted {len(screenshot_paths)} screenshots to {SCREENSHOTS_DIR}")
                for path in screenshot_paths:
                    print(f"  - {path}")
            else:
                print("  No screenshots were extracted")
        else:
            print("  No relevant moments returned by Twelve Labs")
            screenshot_paths = []
    except Exception as e:
        print(f"  ⚠ Could not extract relevant screenshots: {e}")
        moments = []
        screenshot_paths = []

    print("\n--- Step 4.6: Extracting first-order implementation spec ---")
    try:
        first_order_spec = _extract_first_order_spec(indexed_asset.id, transcript, tl_transcript)
        print(f"  First-order spec saved to {FIRST_ORDER_SPEC_PATH}")
    except Exception as e:
        print(f"  ⚠ Could not extract first-order spec: {e}")
        first_order_spec = {
            "summary": "",
            "search_query": transcript,
            "requested_changes": [transcript] if transcript else [],
            "affected_surfaces": [],
        }

    print("\n--- Step 4.7: Querying GraphRAG for existing implementation context ---")
    try:
        rag_context = _retrieve_rag_context(first_order_spec)
        rag_context_text = _format_rag_context(rag_context)
        print(f"  GraphRAG context saved to {RAG_CONTEXT_PATH}")
    except Exception as e:
        print(f"  ⚠ GraphRAG retrieval failed: {e}")
        rag_context = {}
        rag_context_text = "(not available)"

    print("\n--- Step 5: Generating Cursor instructions (Twelve Labs LLM + GraphRAG) ---")
    # Alims already has UI/UX and teacher/student/principal modules. Output ONLY delta requests.
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
{json.dumps(first_order_spec, ensure_ascii=False, indent=2)}
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

Relevant screenshots extracted from the local source video:
---
{chr(10).join(screenshot_paths) if screenshot_paths else "(not available)"}
---

Instructions: Write a detailed, Cursor-ready implementation brief for developers that contains:
- requested changes only
- likely existing files/components to inspect first
- implementation constraints inferred from current code
- a short acceptance checklist

Use concise bullet points. Do not include generic UI/UX or module descriptions. Output ONLY the brief text—no preamble or summary."""

    try:
        print("Generating prompt...")
        final_chunks = []
        text_stream = client.analyze_stream(
            video_id=indexed_asset.id,
            prompt=pm_prompt
        )
        
        print("\n========================================")
        print("🎯 CURSOR INSTRUCTIONS (PM requests only):")
        print("========================================")
        for text in text_stream:
            if text.event_type == "text_generation":
                # Print without line breaks to smoothly output the stream
                print(text.text, end="")
                final_chunks.append(text.text)
        print("\n========================================")
        with open(FINAL_INSTRUCTIONS_PATH, "w", encoding="utf-8") as f:
            f.write("".join(final_chunks).strip() + "\n")

        all_paths = []
        for item in rag_context.get("primary", {}).get("components", []) + rag_context.get("expanded", {}).get("components", []):
            doc = item.get("doc", item)
            source_path = doc.get("metadata", {}).get("sourcePath")
            if source_path:
                all_paths.append(source_path)
        if all_paths:
            with open(FINAL_INSTRUCTIONS_PATH, "a", encoding="utf-8") as f:
                f.write("\n\n## Files to open\n\n")
                f.write("\n".join(f"@{path}" for path in dict.fromkeys(all_paths)))
                f.write("\n")
        print(f"Saved final instructions to {FINAL_INSTRUCTIONS_PATH}")
    except Exception as e:
        print(f"\n❌ Error generating text: {e}")

if __name__ == "__main__":
    test_twelvelabs_pipeline()