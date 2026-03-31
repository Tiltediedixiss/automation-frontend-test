"""
pipeline.py
-----------
Core pipeline: S3 key in → result dict out.

Uses Gemini for both transcription AND video understanding (the PM shows
UI screens in the recording). No TwelveLabs — eliminates the 5+ min
indexing bottleneck while preserving visual analysis.

Step 5 uses 4 parallel focused Gemini calls (changes, files, constraints,
checklist) matching the original double-step extraction pattern.

Expected wall time: ~1-2 minutes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import boto3
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip().strip("\"'")
INDEX_ARN      = os.getenv("AWS_S3_VECTOR_INDEX_ARN", "").strip().strip("\"'")
AWS_REGION     = os.getenv("AWS_REGION", "eu-central-1")
GRAPHRAG_DOCS  = Path(__file__).resolve().parent / "generated" / "graphrag-documents.json"
MAX_UPLOAD_MB  = 200
PIPELINE_FAST_MODE = os.getenv("PIPELINE_FAST_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
MAX_SCREENSHOTS = 5
GEMINI_MODEL   = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 65536

if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment/.env")
if not INDEX_ARN:
    raise RuntimeError("Missing AWS_S3_VECTOR_INDEX_ARN in environment/.env")

_s3     = boto3.client("s3", region_name=AWS_REGION)
_gemini = genai.Client(api_key=GOOGLE_API_KEY.strip())


# ---------------------------------------------------------------------------
# Helpers
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


def _repair_json(text: str) -> dict:
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        last_brace = text.rfind("}")
        last_bracket = text.rfind("]")
        cut = max(last_brace, last_bracket) + 1
        if cut > 0:
            truncated = text[:cut]
            open_braces = truncated.count("{") - truncated.count("}")
            open_brackets = truncated.count("[") - truncated.count("]")
            truncated += "}" * max(0, open_braces) + "]" * max(0, open_brackets)
            return json.loads(truncated)
    except json.JSONDecodeError:
        pass
    timestamps = re.findall(r'"timestamp_sec"\s*:\s*([\d.]+)', text)
    if timestamps:
        return {"moments": [{"timestamp_sec": float(t), "reason": "extracted"} for t in timestamps]}
    return {}


def _gemini_text(prompt: str, temperature: float = 0.35, gemini_file: types.File | None = None) -> str:
    contents: list[Any] = []
    if gemini_file:
        contents.append(gemini_file)
    contents.append(prompt)
    response = _gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    return (response.text or "").strip()


# ---------------------------------------------------------------------------
# Phase 1: Upload + Transcription
# ---------------------------------------------------------------------------

def _upload_video_to_gemini(video_path: str) -> types.File:
    print("  Uploading video to Gemini Files API...")
    uploaded = _gemini.files.upload(file=video_path)
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = _gemini.files.get(name=uploaded.name)
    if uploaded.state.name == "FAILED":
        raise RuntimeError(f"Gemini file upload failed: {uploaded.state}")
    print(f"  Gemini file ready: {uploaded.name}")
    return uploaded


def _transcribe_with_google(video_path: str) -> str:
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
        print("  Transcribing with Gemini...")
        response = _gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                (
                    "Transcribe this audio verbatim. "
                    "The spoken language is English. "
                    "Return the transcript in English. "
                    "Do not translate. Do not summarize. Do not infer missing content. "
                    "Preserve wording as spoken as closely as possible."
                ),
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        return (response.text or "").strip()
    finally:
        try:
            os.unlink(wav_path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Phase 2: Spec + Moments (video understanding)
# ---------------------------------------------------------------------------

def _identify_relevant_moments(gemini_file: types.File, transcript: str) -> list[dict]:
    for attempt in range(2):
        response = _gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                gemini_file,
                f"""You are analyzing a product feature-request screen recording for Alims.
The PM is showing UI screens and explaining what they want changed.

TRANSCRIPT (ground truth for what PM said):
---
{transcript or "(no transcript available)"}
---

Watch the video carefully. Identify 1 to 5 key moments where the PM is showing
a screen or UI element relevant to their requested change.

For each moment provide:
- timestamp_sec: the second in the video to capture a screenshot
- reason: what the PM is showing/pointing at
- screen_description: describe what's visible on screen (UI elements, page, data shown)

Return JSON object with key "moments" containing array of objects.
Keep string values short (under 100 chars) to avoid truncation.""",
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
            ),
        )
        text = (response.text or "").strip()
        parsed = _repair_json(text)
        moments = parsed.get("moments", [])
        if moments:
            return moments
    return []


def _extract_screenshots(video_path: str, moments: list[dict]) -> list[str]:
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


def _extract_first_order_spec(gemini_file: types.File, transcript: str) -> dict:
    response = _gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            gemini_file,
            f"""You are extracting a complete implementation spec from a product feature-request video for Alims.

Alims has three existing modules: teacher, student, principal. Do NOT re-describe their baseline behavior.
However, if the PM is requesting a change, fix, or addition TO any existing module, screen, or component — include it.

Your goal is HIGH RECALL. It is better to include something borderline than to miss a real request.

TRANSCRIPT — treat this as ground truth for what the PM said:
---
{transcript or "(no transcript available)"}
---

Watch the video carefully. The PM is showing screens and pointing at UI elements.
Combine what you SEE on screen with what the PM SAYS to extract:

- summary: 2-3 sentences MAX. High-level overview of the feature scope — what module, what the PM wants. Do NOT list every individual change here. This is just an executive summary for a developer to quickly understand the scope.
- search_query: A dense retrieval query covering all topics mentioned (components, routes, behaviors).
- requested_changes: Every distinct change, fix, or new behavior the PM requested — one item per action. Be specific and actionable. Do NOT include timestamps or video references. Do not merge unrelated changes. Do not omit any.
- affected_surfaces: Every route, screen, component, or module the PM mentions or shows on screen.
- visual_context: Describe what the PM showed on screen that adds context beyond the transcript (UI state, data visible, navigation path taken). Do NOT repeat the requested_changes here.

Do not invent requirements. Do not omit requirements the PM stated or demonstrated visually.

Return JSON with keys: summary, search_query, requested_changes (string[]), affected_surfaces (string[]), visual_context (string).""",
        ],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
        ),
    )
    text = (response.text or "").strip()
    return _repair_json(text) if text else {}


# ---------------------------------------------------------------------------
# GraphRAG
# ---------------------------------------------------------------------------

def _retrieve_rag_context(spec: dict) -> dict:
    if not GRAPHRAG_DOCS.exists():
        raise FileNotFoundError(
            f"GraphRAG documents file not found: {GRAPHRAG_DOCS}. "
            "Ensure generated/graphrag-documents.json exists."
        )
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from query_rag import GraphRAGRetriever, infer_region

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
# Step 5: Cursor instructions — 4 parallel focused calls (original pattern)
# ---------------------------------------------------------------------------

CORE_RULES = """Context: The product is Alims. It already has a ready UI/UX and modules (teacher/student/principal).
Do NOT describe existing baseline UI/UX.
Extract only PM-requested changes.
If something is visible but not requested, omit it.
If uncertain/inferred, omit it.
Use concise bullet points."""


def _generate_changes_section(
    gemini_file: types.File,
    spec_json: str,
    rag_context_text: str,
    transcript: str,
) -> str:
    prompt = f"""{CORE_RULES}

FIRST-ORDER SPEC:
{spec_json}

GRAPHRAG CONTEXT (existing codebase — what is already implemented):
{rag_context_text or "(not available)"}

TRANSCRIPT (Gemini):
{transcript or "(no transcript available)"}

Watch the video again. Use both what you SEE on screen and the codebase context from GraphRAG.

Task: Output ONLY section:
## Requested changes

STRICT RULES:
- One bullet per distinct change. Be specific and actionable — a developer should know exactly what to build.
- Reference existing components from GRAPHRAG CONTEXT when the change modifies them (e.g. "In QuestionFeedback component, add...").
- Do NOT include timestamps (no "at 0:59", no "shown at 1:10").
- Do NOT include "(Visual: ...)" annotations.
- Do NOT describe what the PM showed — describe what the developer needs to IMPLEMENT.
- Do NOT repeat or rephrase the summary. Each bullet is a concrete implementation task.
- Be exhaustive — do not omit any requested change."""
    return _gemini_text(prompt, gemini_file=gemini_file)


def _generate_constraints_section(
    gemini_file: types.File,
    spec_json: str,
    rag_context_text: str,
    transcript: str,
) -> str:
    prompt = f"""{CORE_RULES}

FIRST-ORDER SPEC:
{spec_json}

GRAPHRAG CONTEXT (existing codebase — what is already implemented):
{rag_context_text or "(not available)"}

TRANSCRIPT (Gemini):
{transcript or "(no transcript available)"}

Watch the video again. Identify constraints by comparing what the PM showed on screen with what already exists in the codebase.

Task: Output ONLY section:
## Implementation constraints

STRICT RULES:
- Only technical constraints, edge cases, gotchas, patterns to follow, and things that could go wrong.
- Reference specific existing components/patterns from GraphRAG that must be reused or followed.
- Do NOT repeat the requested changes — the developer already has those. This section is about HOW to implement, not WHAT to implement.
- Do NOT include timestamps or video references.
- Max 10 bullets."""
    return _gemini_text(prompt, gemini_file=gemini_file)


def _generate_checklist_section(
    gemini_file: types.File,
    spec_json: str,
    transcript: str,
) -> str:
    prompt = f"""{CORE_RULES}

FIRST-ORDER SPEC:
{spec_json}

TRANSCRIPT:
{transcript or "(no transcript available)"}

Watch the video again. Base each checklist item on what the PM demonstrated.

Task: Output ONLY section:
## Acceptance checklist

STRICT RULES:
- Each bullet is a testable verification step: "Navigate to X, do Y, verify Z happens."
- Focus on user-visible behavior and expected outcomes.
- Do NOT repeat the requested changes — the developer already has those. This section is about HOW TO TEST, not what to build.
- Do NOT include timestamps (no "at 0:59", no "shown at 1:10").
- Do NOT describe what the PM showed — describe what the tester should verify.
- Max 10 bullets."""
    return _gemini_text(prompt, gemini_file=gemini_file)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(s3_bucket: str, s3_key: str, task_id: str = "") -> dict:
    label = task_id or s3_key
    video_path: str | None = None
    to_upload:  str | None = None
    gemini_file: types.File | None = None

    try:
        print(f"[{label}] Downloading s3://{s3_bucket}/{s3_key}")
        fd, video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        _s3.download_file(s3_bucket, s3_key, video_path)
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"[{label}] Downloaded: {size_mb:.1f} MB")

        to_upload = _compress_if_needed(video_path)
        final_video = to_upload

        # ── Phase 1: Upload to Gemini + transcribe (parallel) ───────────
        print(f"\n[{label}] --- Phase 1: Gemini upload + transcription (parallel) ---")
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_upload     = pool.submit(_upload_video_to_gemini, final_video)
            fut_transcript = pool.submit(_transcribe_with_google, final_video)

            gemini_file = fut_upload.result()
            transcript  = fut_transcript.result()
            print(f"[{label}] Transcript: {len(transcript)} chars")

        # ── Phase 2: Spec + moments (parallel, both use gemini video) ───
        print(f"\n[{label}] --- Phase 2: Spec + moments extraction (parallel) ---")
        moments: list[dict] = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_spec = pool.submit(_extract_first_order_spec, gemini_file, transcript)
            fut_moments = None
            if not PIPELINE_FAST_MODE:
                fut_moments = pool.submit(_identify_relevant_moments, gemini_file, transcript)

            try:
                spec = fut_spec.result()
                print(f"[{label}] Spec summary: {spec.get('summary', '')[:80]}...")
            except Exception as exc:
                print(f"[{label}] Spec extraction failed: {exc}")
                spec = {
                    "summary": "",
                    "search_query": transcript,
                    "requested_changes": [transcript] if transcript else [],
                    "affected_surfaces": [],
                }

            if fut_moments:
                try:
                    moments = fut_moments.result()
                    print(f"[{label}] Moments: {len(moments)}")
                except Exception as exc:
                    print(f"[{label}] Moments failed (non-fatal): {exc}")

        # ── Screenshots ─────────────────────────────────────────────────
        screenshot_paths: list[str] = []
        if moments:
            print(f"\n[{label}] --- Extracting screenshots ---")
            screenshot_paths = _extract_screenshots(final_video, moments)
            print(f"[{label}] Screenshots: {len(screenshot_paths)}")

        # ── Phase 3: GraphRAG ───────────────────────────────────────────
        print(f"\n[{label}] --- Phase 3: Querying GraphRAG ---")
        try:
            rag_context      = _retrieve_rag_context(spec)
            rag_context_text = _format_rag_context(rag_context)
            resolved_files   = _extract_file_paths(rag_context)
            print(f"[{label}] RAG resolved {len(resolved_files)} file(s)")
        except Exception as exc:
            import traceback
            print(f"[{label}] GraphRAG failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            rag_context, rag_context_text, resolved_files = {}, "(not available)", []

        # ── Phase 4: Cursor instructions — 3 focused parallel calls ─────
        print(f"\n[{label}] --- Phase 4: Generating Cursor instructions (3 parallel sections) ---")
        spec_json = json.dumps(spec, ensure_ascii=False, indent=2)

        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_changes = pool.submit(
                _generate_changes_section,
                gemini_file, spec_json, rag_context_text, transcript,
            )
            fut_constraints = pool.submit(
                _generate_constraints_section,
                gemini_file, spec_json, rag_context_text, transcript,
            )
            fut_checklist = pool.submit(
                _generate_checklist_section,
                gemini_file, spec_json, transcript,
            )

            sections: dict[str, str] = {}
            for name, fut in [
                ("changes", fut_changes),
                ("constraints", fut_constraints),
                ("checklist", fut_checklist),
            ]:
                try:
                    sections[name] = fut.result()
                    print(f"[{label}] Section '{name}': {len(sections[name])} chars")
                except Exception as exc:
                    print(f"[{label}] Section '{name}' failed: {exc}")
                    sections[name] = ""

        parts = [
            sections.get("changes", ""),
            sections.get("constraints", ""),
            sections.get("checklist", ""),
        ]

        if resolved_files:
            files_section = "## Files to open\n\n" + "\n".join(f"@{p}" for p in resolved_files)
            parts.append(files_section)

        instructions = "\n\n".join(
            part.strip() for part in parts if part and part.strip()
        ).strip()

        if not instructions:
            raise RuntimeError("Step 5 instruction generation produced empty output.")

        print(f"\n[{label}] Done.")

        return {
            "task_id":      task_id,
            "summary":      spec.get("summary", ""),
            "files":        resolved_files,
            "instructions": instructions,
        }

    finally:
        if gemini_file:
            try:
                _gemini.files.delete(name=gemini_file.name)
            except Exception:
                pass
        for p in {video_path, to_upload} - {None}:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass