"""
Process Clips with Vertical Resizing and Captions
--------------------------------------------------
1. Transcribe full video with ClipsAI (WhisperX)
2. Find optimal clips (60-120s)
3. Trim clips with FFmpeg
4. Resize to 9:16 (TikTok/Shorts) using ClipsAI smart crop
5. Transcribe each clip individually for accurate SRT timing
6. Burn captions at bottom using FFmpeg (TikTok style)
"""
import warnings
warnings.filterwarnings('ignore')

from clipsai import ClipFinder, Transcriber, resize
import os
import subprocess
import shutil
from pathlib import Path

# =============================================================================
# SETTINGS
# =============================================================================
VIDEO_PATH = Path(r"G:\pelosi-trades-youtube - Copy.mp4")
OUTPUT_DIR = Path(r"C:\dscodingpython\clips_output")

MIN_DURATION = 60.0   # seconds
MAX_DURATION = 120.0  # seconds

# Option to skip captions/transcription (just export vertical clips)
SKIP_CAPTIONS = True

# Hugging Face token for Pyannote (speaker diarization in resize)
# Set HF_TOKEN in your .env file
HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["HF_TOKEN"] = HF_TOKEN

# Subtitle style (TikTok-style, positioned at bottom)
# MarginV=120 pushes it up from very bottom to avoid TikTok UI overlap
SUBTITLE_STYLE = (
    "FontName=Arial Black,"
    "FontSize=18,"
    "PrimaryColour=&H00FFFFFF&,"
    "OutlineColour=&H00000000&,"
    "BackColour=&H80000000&,"
    "BorderStyle=4,"
    "Outline=2,"
    "Shadow=0,"
    "MarginV=120,"
    "Alignment=2,"
    "Bold=1"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def transcribe_and_find_clips():
    """Transcribe full video and find optimal clips in the 60-120s range."""
    print("=" * 60)
    print("STEP 1: Transcribing full video...")
    print("=" * 60)
    
    transcriber = Transcriber()
    transcription = transcriber.transcribe(audio_file_path=str(VIDEO_PATH))
    
    print("STEP 2: Finding optimal clips...")
    finder = ClipFinder()
    clips = finder.find_clips(transcription=transcription)
    
    # Filter by duration
    filtered = []
    for c in clips:
        duration = c.end_time - c.start_time
        if MIN_DURATION <= duration <= MAX_DURATION:
            filtered.append((c.start_time, c.end_time, duration))
    
    print(f"Found {len(clips)} total clips, {len(filtered)} in {MIN_DURATION}-{MAX_DURATION}s range")
    return filtered


def trim_clip(start_s: float, end_s: float, idx: int) -> Path:
    """Trim a segment from the source video using FFmpeg."""
    out_path = OUTPUT_DIR / f"clip_{idx:02d}_raw.mp4"
    print(f"  Trimming {start_s:.1f}s - {end_s:.1f}s...")
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-to", str(end_s),
        "-i", str(VIDEO_PATH),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(out_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def resize_to_vertical(input_path: Path, idx: int) -> Path:
    """
    Resize video to 9:16 (vertical) using FFmpeg center-crop.
    ClipsAI smart crop disabled due to mediapipe compatibility issues.
    """
    print(f"  Resizing to 9:16 vertical (FFmpeg center-crop)...")
    return resize_ffmpeg_fallback(input_path, idx)


def resize_ffmpeg_fallback(input_path: Path, idx: int) -> Path:
    """Fallback: Simple center crop to 9:16 using FFmpeg."""
    output_path = OUTPUT_DIR / f"clip_{idx:02d}_vertical.mp4"
    
    # Get video dimensions first
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(input_path)
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    w, h = map(int, result.stdout.strip().split(','))
    
    # Calculate 9:16 crop (keep height, crop width)
    target_w = int(h * 9 / 16)
    x_offset = (w - target_w) // 2
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"crop={target_w}:{h}:{x_offset}:0",
        "-c:a", "copy",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        str(output_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def _load_faster_whisper_model():
    """Load faster-whisper preferring GPU; fall back to CPU and return model or None."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("    [faster-whisper not installed]")
        return None

    try:
        print("    [Loading Whisper model on GPU...]")
        return WhisperModel("base", device="cuda", compute_type="float16")
    except Exception as e:
        print(f"    [GPU failed: {e}, trying CPU...]")
        try:
            return WhisperModel("base", device="cpu", compute_type="int8")
        except Exception as e2:
            print(f"    [CPU also failed: {e2}]")
            return None


def _transcribe_with_clipsai(video_path: Path) -> list:
    """Fallback transcription using ClipsAI transcriber."""
    transcriber = Transcriber()
    transcription = transcriber.transcribe(audio_file_path=str(video_path))
    segments = []
    if hasattr(transcription, "segments"):
        for seg in transcription.segments:
            segments.append({
                "start": seg.start_time,
                "end": seg.end_time,
                "text": getattr(seg, "text", str(seg)).strip()
            })
    return segments


def transcribe_clip_for_srt(video_path: Path) -> list:
    """
    Transcribe a single clip to get word-level timestamps for SRT.
    Returns list of segments: [{'start': float, 'end': float, 'text': str}, ...]
    """
    print(f"  Transcribing clip for subtitles...")

    model = _load_faster_whisper_model()
    if model is None:
        print("  ⚠ faster-whisper not available, using ClipsAI transcriber...")
        return _transcribe_with_clipsai(video_path)

    try:
        print("    [Running transcription...]")
        segments_out = []
        segments, info = model.transcribe(str(video_path), word_timestamps=False)
        print(f"    [Detected language: {info.language}, processing segments...]")
        for segment in segments:
            segments_out.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
            })
            print(f"    [{segment.start:.1f}s-{segment.end:.1f}s] {segment.text.strip()[:50]}...")
        print(f"    [Done: {len(segments_out)} segments]")
        return segments_out
    except Exception as e:
        print(f"  ⚠ faster-whisper failed: {e}. Falling back to ClipsAI.")
        return _transcribe_with_clipsai(video_path)


def generate_srt(segments: list, output_path: Path) -> Path:
    """Generate SRT subtitle file from segments."""
    srt_path = output_path.with_suffix('.srt')
    
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start_ts = format_srt_timestamp(seg['start'])
            end_ts = format_srt_timestamp(seg['end'])
            text = seg['text'].strip()
            
            f.write(f"{i}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{text}\n\n")
    
    return srt_path


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def burn_subtitles(video_path: Path, srt_path: Path, idx: int) -> Path:
    """
    Burn subtitles into video using FFmpeg.
    Subtitles positioned at bottom with TikTok-style formatting.
    """
    output_path = OUTPUT_DIR / f"clip_{idx:02d}_final.mp4"
    print(f"  Burning subtitles (bottom position)...")
    
    # Escape path for FFmpeg subtitles filter (Windows paths need special handling)
    srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"subtitles='{srt_escaped}':force_style='{SUBTITLE_STYLE}'",
        "-c:a", "copy",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"  ⚠ Subtitle burn failed: {e.stderr.decode()[:200]}")
        # Return video without subtitles
        shutil.copy(video_path, output_path)
        return output_path


def process_single_clip(idx: int, start_s: float, end_s: float) -> Path:
    """Process a single clip through the full pipeline."""
    print(f"\n{'='*60}")
    print(f"PROCESSING CLIP {idx}: {start_s:.1f}s - {end_s:.1f}s ({end_s-start_s:.1f}s)")
    print(f"{'='*60}")
    
    # Step 1: Trim
    trimmed = trim_clip(start_s, end_s, idx)
    
    # Step 2: Resize to 9:16 vertical
    vertical = resize_to_vertical(trimmed, idx)
    

    if SKIP_CAPTIONS:
        print("  [SKIP_CAPTIONS=True] Exporting vertical clip only, no subtitles.")
        final = vertical
    else:
        # Step 3: Transcribe for SRT
        segments = transcribe_clip_for_srt(vertical)
        if segments:
            # Step 4: Generate SRT
            srt_path = generate_srt(segments, vertical)
            print(f"  Generated SRT with {len(segments)} segments")
            # Step 5: Burn subtitles
            final = burn_subtitles(vertical, srt_path, idx)
        else:
            print(f"  ⚠ No segments found, skipping subtitles")
            final = vertical

    # Cleanup intermediate files
    if trimmed.exists() and trimmed != final:
        trimmed.unlink()
    if vertical.exists() and vertical != final:
        vertical.unlink()

    print(f"  ✓ Done: {final.name}")
    return final


def main():
    """Main pipeline: Find clips → Trim → Resize → Caption."""
    print("\n" + "=" * 60)
    print("TIKTOK/SHORTS CLIP PROCESSOR")
    print("=" * 60)
    print(f"Source: {VIDEO_PATH}")
    print(f"Duration filter: {MIN_DURATION}-{MAX_DURATION}s")
    print()
    
    ensure_output_dir()
    
    # Find clips
    clips = transcribe_and_find_clips()
    
    if not clips:
        print("\n❌ No clips found in the specified duration range.")
        return
    
    # Process each clip
    exported = []
    for idx, (start_s, end_s, duration) in enumerate(clips, start=1):
        try:
            final = process_single_clip(idx, start_s, end_s)
            exported.append(final)
        except Exception as e:
            print(f"  ❌ Failed to process clip {idx}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Processed {len(exported)}/{len(clips)} clips successfully:\n")
    for path in exported:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {path.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
