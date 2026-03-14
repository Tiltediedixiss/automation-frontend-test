import subprocess
import os
import tempfile


video_file = "test.mp4"


def _get_google_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not key or not key.strip():
        raise RuntimeError("Set GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS in .env with your Google API key.")
    return key.strip()


def transcribe_with_google(video_path: str, language_code: str = "ru-RU") -> str:
    """Extract audio from video and transcribe with Gemini using API key from .env."""
    api_key = _get_google_api_key()
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("Install: pip install google-genai")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        print("Extracting audio to 16 kHz mono WAV...")
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

        client = genai.Client(api_key=api_key)
        print("Transcribing with Gemini...")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                f"Transcribe this audio to plain text. Language: {language_code}. Output only the transcript, no commentary.",
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            ],
        )
        return (response.text or "").strip()
    finally:
        try:
            os.unlink(wav_path)
        except FileNotFoundError:
            pass


def test_pipeline():
    if not os.path.exists(video_file):
        print(f"Error: Please place a test video named '{video_file}' in this directory.")
        return

    print("--- Step 1: Extracting unique frames with videostil ---")
    try:
        use_shell = os.name == "nt"
        if use_shell:
            cmd = f'npx videostil "{video_file}" --fps 5 --threshold 0.1 --dedup-method pixelmatch --no-serve'
        else:
            cmd = [
                "npx", "videostil", video_file,
                "--fps", "5",
                "--threshold", "0.1",
                "--dedup-method", "pixelmatch",
                "--no-serve"
            ]
        subprocess.run(cmd, check=True, shell=use_shell)
        print("✅ videostil complete. Check your ~/.videostil/ folder (or the local output directory) for the unique PNGs.")
    except subprocess.CalledProcessError:
        print("❌ Error running videostil. Ensure Node.js and ffmpeg are installed.")
        return

    print("\n--- Step 2: Transcribing audio with Gemini (ru-RU) ---")
    try:
        text = transcribe_with_google(video_file, language_code="ru-RU")
        print("\n✅ Transcription Complete:\n")
        print("========================================")
        print(text or "(no speech detected)")
        print("========================================")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_pipeline()
