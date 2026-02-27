#!/usr/bin/env python3
"""AI-Powered Local Video Editor

A production-ready CLI tool for automated video editing optimized for RTX 3070+ GPUs.
Processes 10-min video in 3-10 minutes using Gemini API for intelligent editing decisions.

Features:
- Silence removal via auto-editor integration
- Retake detection using fuzzy transcript matching
- Face-tracking dynamic zoom (requires OpenCV)
- AI-driven B-roll insertion with Pexels/Pixabay API
- YouTube chapter generation
- Subtitle embedding

Usage:
    python ai_video_editor.py input.mp4 output.mp4 --mode auto --subtitles --chapters

Requirements:
    pip install faster-whisper opencv-python-headless numpy==1.26.4 tqdm requests python-dotenv
    FFmpeg binary must be installed and in PATH
    Set GEMINI_API_KEY in environment or .env file
"""
from __future__ import annotations

import argparse
import atexit
import difflib
import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Audio enhancement (lazy imports to avoid startup delay if not used)
# pip install noisereduce pydub scipy pedalboard
try:
    import noisereduce as nr
    from scipy.io import wavfile
    from pydub import AudioSegment, effects as pydub_effects
    AUDIO_ENHANCE_AVAILABLE = True
except ImportError:
    AUDIO_ENHANCE_AVAILABLE = False

# Spotify's Pedalboard for professional audio processing (EQ, compression, limiting)
# pip install pedalboard
try:
    from pedalboard import Pedalboard, Compressor, Gain, LowShelfFilter, HighShelfFilter, Limiter
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False

# Load environment variables using project-relative fallbacks
_PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_SEARCH_PATHS = [
    _PROJECT_ROOT / ".env",          # Repo-local .env
    _PROJECT_ROOT.parent / ".env",    # Parent directory .env
]

_env_loaded = False
for env_path in _ENV_SEARCH_PATHS:
    if env_path.exists():
        load_dotenv(env_path, override=True)  # override=True ensures .env takes priority
        _env_loaded = True
        break

if not _env_loaded:
    # Fall back to default dotenv search (HOME, cwd, etc.)
    load_dotenv(override=True)

# Lazy import for faster-whisper (heavy CUDA initialization)
WhisperModel = None

# Lazy import for Wav2Vec2 (acoustic stutter detection)
Wav2Vec2ForCTC = None
Wav2Vec2Processor = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ai_video_editor.log", mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"  # Flash is fast/cheap and sufficient for retake detection  # Gemini 2.5 Pro model

# B-roll API configuration
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PEXELS_API_URL = "https://api.pexels.com/videos/search"
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
PIXABAY_API_URL = "https://pixabay.com/api/videos/"

# Debug: Log API key status at startup
print(f"[ENV DEBUG] PEXELS_API_KEY loaded: {'Yes' if PEXELS_API_KEY else 'No'}")
print(f"[ENV DEBUG] PIXABAY_API_KEY loaded: {'Yes' if PIXABAY_API_KEY else 'No'}")
print(f"[ENV DEBUG] GEMINI_API_KEY loaded: {'Yes' if GEMINI_API_KEY else 'No'}")

# Retake detection phrases (expanded) - triggers immediate retake detection
RETAKE_PHRASES = [
    "sorry", "let's try again", "take two", "take 2", "one more time",
    "start over", "my bad", "oops", "wait", "hold on", "actually no",
    "let me start over", "let me try that again", "hang on", "scratch that",
    "never mind", "no no no", "that's not right", "let me rephrase",
    "i'll try again", "let me do that again", "that was bad", "that was wrong",
    "let me say that again", "again", "one sec", "no wait", "no no",
    "actually wait", "hold up", "let me", "i mean", "what i meant",
]

# Filler words for detection (expanded) - includes stutters common for non-native speakers
FILLER_WORDS = [
    "um", "uh", "like", "you know", "basically", "actually", "so", "right",
    "kind of", "sort of", "i mean", "you see", "well", "anyway", "literally",
    "ah", "eh", "er", "hmm", "mm", "hm",  # Hesitation sounds
]

# Stutter patterns - AGGRESSIVE detection for all types of stutters
# Catches: "th-the", "the, the", "the... the", "I was I was going" etc.
STUTTER_PATTERNS = [
    r'\b(\w{1,4})-\1',  # Hyphenated repeats: "th-the", "wh-what", "I-I"
    r'\b(\w+)(\s*[,.…]*\s*)\1\b',  # Word repeats with ANY punctuation: "the, the", "the... the"
    r'\b(\w+\s+\w+)\s+\1\b',  # 2-word phrase repeats: "I was I was going"
    r'\b(\w+\s+\w+\s+\w+)\s+\1\b',  # 3-word phrase repeats
]

# LLM prompt template for B-roll analysis
LLM_PROMPT_TEMPLATE = """You are an AGGRESSIVE video editor. Your job is to REMOVE bad segments ruthlessly.

Segments to analyze (numbered 1 to N):
{segments_text}

REMOVE segments that have ANY of these issues:

WITHIN A SINGLE SEGMENT (look for repeated phrases inside the text):
1. SELF-CORRECTIONS: "and that even left and that even left" - said twice, REMOVE the segment.
2. RESTARTS: "to where is our money to our to where is" - stumbled and restarted, REMOVE.
3. REPEATED PHRASES: "we're not just we're we're not just" - false start within segment, REMOVE.
4. MUMBLES: "their mother their history" - misspoke then corrected, REMOVE the segment.

ACROSS ADJACENT SEGMENTS:
5. RETAKES: Segment N and N+1 say the same idea? Remove the worse one.
6. FALSE STARTS: "I've been weeks digging" then "I've spent weeks digging" - REMOVE the first (incomplete).
7. INCOMPLETE: Segment ends mid-thought, next segment restarts - REMOVE the incomplete one.

GENERAL ISSUES:
8. FILLER WORDS: Lots of um/uh/like/you know/so/ah/er? REMOVE IT.
9. STUTTERS: "the the", "I I", "th-the" patterns - REMOVE.
10. WEAK: Uncertain, hesitant, low energy, mumbling - REMOVE.

CRITICAL: If you see the SAME PHRASE twice within a segment's text, that segment has a retake and should be REMOVED.

The speaker is non-native English - look for hesitation patterns and self-corrections.
You MUST remove segments with internal repetition. Don't be afraid to cut.

Respond with ONLY this JSON format (nothing else):
{{"remove_indices": [1, 3], "broll_suggestions": {{}}}}

The remove_indices array should contain segment numbers (1-based) to DELETE.
If all segments are perfect, use: {{"remove_indices": [], "broll_suggestions": {{}}}}"""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EditorConfig:
    """Configuration container for all editor settings."""
    input_path: Path
    output_path: Path
    mode: str = "auto"
    whisper_model: str = "crisper"  # CrisperWhisper for verbatim transcription with stutters/fillers
    gemini_model: str = GEMINI_DEFAULT_MODEL
    gemini_api_key: Optional[str] = None
    device: str = "cuda"
    similarity_threshold: float = 0.75
    silence_duration: float = 0.7
    pexels_api_key: Optional[str] = None
    pixabay_api_key: Optional[str] = None
    local_broll_dir: Path = field(default_factory=lambda: Path("./cc0_clips"))
    chapters: bool = False
    subtitles: bool = False
    pre_cut_silence: bool = False  # New flag for auto-editor pre-pass
    fix_drift: bool = False # New flag for VFR fix
    audio_delay_ms: int = 0  # Audio delay in milliseconds (positive = delay audio, for OBS sync fix)
    script_path: Optional[Path] = None  # Path to the original script for guided editing
    skip_llm_cuts: bool = True  # Skip LLM for CUT decisions (use only algorithmic). LLM still used for B-roll if enabled.
    use_wav2vec2: bool = True  # Use Wav2Vec2 for acoustic stutter detection (second pass)
    enhance_audio: bool = True  # Apply noise reduction and normalization to audio
    temp_dir: Optional[Path] = None
    
    def __post_init__(self):
        self.input_path = Path(self.input_path)
        self.output_path = Path(self.output_path)
        self.local_broll_dir = Path(self.local_broll_dir)
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        # Create temp directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ai_video_editor_"))
        atexit.register(self._cleanup_temp)
        logger.info(f"Temp directory: {self.temp_dir}")
    
    def _cleanup_temp(self):
        """Clean up temporary files on exit."""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info("Cleaned up temp files")
            except Exception as e:
                logger.warning(f"Failed to clean temp dir: {e}")


@dataclass
class TranscriptSegment:
    """A segment of transcribed audio with timing."""
    start: float
    end: float
    text: str
    words: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class EditDecision:
    """An edit decision for a segment."""
    start: float
    end: float
    keep: bool
    reason: str = ""
    broll_keywords: Optional[List[str]] = None


# =============================================================================
# TRANSCRIPTION (faster-whisper / CrisperWhisper)
# =============================================================================

# CrisperWhisper model ID for verbatim transcription with stutters/fillers
CRISPER_WHISPER_MODEL = "nyrahealth/faster_CrisperWhisper"

def load_whisper_model(model_name: str, device: str):
    """Load Whisper model with caching.
    
    Supports:
    - Standard faster-whisper models: "large-v3", "medium", etc.
    - CrisperWhisper for verbatim transcription: "crisper" or full model ID
    
    CrisperWhisper is recommended for retake detection as it transcribes
    stutters, false starts, and fillers that standard Whisper omits.
    """
    global WhisperModel
    if WhisperModel is None:
        from faster_whisper import WhisperModel as WM
        WhisperModel = WM
    
    # Handle CrisperWhisper shorthand
    if model_name.lower() == "crisper":
        model_name = CRISPER_WHISPER_MODEL
        logger.info("Using CrisperWhisper for verbatim transcription (includes stutters, fillers)")
    
    compute_type = "float16" if device == "cuda" else "int8"
    logger.info(f"Loading Whisper model '{model_name}' on {device} ({compute_type})...")
    
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )


# =============================================================================
# WAV2VEC2 ACOUSTIC STUTTER DETECTION
# =============================================================================

def load_wav2vec2_model(device: str = "cuda"):
    """Load Wav2Vec2 CTC model for acoustic stutter detection.
    
    Wav2Vec2 is "dumber" than Whisper - it transcribes literally what it hears
    without language model smoothing. This makes it better at detecting stutters
    that Whisper would normalize away.
    
    Example:
        Whisper hears "I- I- I want" and outputs "I want"
        Wav2Vec2 hears "I- I- I want" and outputs "I I I WANT"
    """
    global Wav2Vec2ForCTC, Wav2Vec2Processor
    
    if Wav2Vec2ForCTC is None:
        try:
            from transformers import Wav2Vec2ForCTC as W2V2, Wav2Vec2Processor as W2V2P
            Wav2Vec2ForCTC = W2V2
            Wav2Vec2Processor = W2V2P
        except ImportError:
            logger.warning("transformers not installed, Wav2Vec2 stutter detection disabled")
            return None, None
    
    model_id = "facebook/wav2vec2-large-960h-lv60-self"
    logger.info(f"Loading Wav2Vec2 model '{model_id}' for acoustic stutter detection...")
    
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    
    if device == "cuda":
        import torch
        if torch.cuda.is_available():
            model = model.to("cuda")
    
    return processor, model


def detect_acoustic_stutters_for_segment(
    audio_path: Path, 
    segment_start: float, 
    segment_end: float,
    whisper_text: str,
    processor, 
    model, 
    device: str = "cuda"
) -> Tuple[List[Tuple[float, float, str]], float]:
    """Use Wav2Vec2 to detect acoustic stutters in a specific segment.
    
    Compares Wav2Vec2's literal transcription to Whisper's cleaned text
    to find stutters that Whisper normalized away.
    
    Args:
        audio_path: Path to audio file
        segment_start: Start time in seconds
        segment_end: End time in seconds  
        whisper_text: The cleaned text from Whisper for comparison
        processor: Wav2Vec2 processor
        model: Wav2Vec2 model
        device: "cuda" or "cpu"
    
    Returns:
        Tuple of:
        - List of (start_time, end_time, stutter_text) for detected stutters
        - Suggested new start time (if stutter at beginning, trim it)
    """
    if processor is None or model is None:
        return [], segment_start

    # Ensure whisper_text is always a usable string
    whisper_text = whisper_text or ""
    
    import torch
    import torchaudio
    
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Resample to 16kHz if needed (Wav2Vec2 requirement)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)
        
        # Extract just this segment
        start_sample = int(segment_start * 16000)
        end_sample = int(segment_end * 16000)
        segment_waveform = waveform[start_sample:end_sample]
        
        if len(segment_waveform) < 1600:  # Skip very short segments (< 0.1s)
            return [], segment_start
        
        # Get transcription from Wav2Vec2
        inputs = processor(segment_waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        
        if device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(inputs['input_values']).logits
        
        # Decode without language model (literal transcription)
        predicted_ids = torch.argmax(logits, dim=-1)
        wav2vec_text = (processor.batch_decode(predicted_ids)[0] or "").upper()
        
        # Compare Wav2Vec2 to Whisper
        wav2vec_words = wav2vec_text.split()
        whisper_words = whisper_text.upper().split()
        
        stutters = []
        new_start = segment_start
        segment_duration = segment_end - segment_start
        
        # Count words in each
        wav2vec_word_count = len(wav2vec_words)
        whisper_word_count = len(whisper_words)
        
        # If Wav2Vec2 has MORE words than Whisper, there are likely stutters
        if wav2vec_word_count > whisper_word_count:
            # Find consecutive repeated words (stutters)
            i = 0
            while i < len(wav2vec_words) - 1:
                repeat_count = 1
                while i + repeat_count < len(wav2vec_words) and wav2vec_words[i] == wav2vec_words[i + repeat_count]:
                    repeat_count += 1
                
                if repeat_count >= 2:
                    # Check if this repeat is NOT in Whisper text (i.e., Whisper cleaned it)
                    repeated_word = wav2vec_words[i]
                    whisper_count = whisper_words.count(repeated_word)
                    
                    # If Whisper has fewer instances, this is a stutter Whisper removed
                    if whisper_count < repeat_count:
                        # Estimate timing based on position in Wav2Vec2 output
                        word_duration = segment_duration / max(wav2vec_word_count, 1)
                        stutter_start = segment_start + i * word_duration
                        stutter_end = stutter_start + (repeat_count - 1) * word_duration  # Keep one instance
                        stutter_text = ' '.join([repeated_word] * repeat_count)
                        stutters.append((stutter_start, stutter_end, stutter_text))
                        
                        # If stutter is at the very beginning, suggest trimming start
                        if i == 0:
                            new_start = segment_start + (repeat_count - 1) * word_duration
                            logger.debug(f"Wav2Vec2: Stutter at start, suggest trim {segment_start:.2f}s -> {new_start:.2f}s")
                        
                        logger.debug(f"Wav2Vec2: Stutter '{stutter_text}' at {stutter_start:.2f}s (Whisper has {whisper_count}x, Wav2Vec2 has {repeat_count}x)")
                    
                    i += repeat_count
                else:
                    i += 1
        
        return stutters, new_start
        
    except Exception as e:
        logger.warning(f"Wav2Vec2 segment detection failed: {e}")
        return [], segment_start


def detect_acoustic_stutters(audio_path: Path, processor, model, device: str = "cuda") -> List[Tuple[float, float, str]]:
    """Use Wav2Vec2 to detect acoustic stutters across the entire audio.
    
    Returns list of (start_time, end_time, text) for detected stutters.
    
    This is a "second opinion" pass - Wav2Vec2 transcribes literally what it hears,
    so repeated sounds like "I I I" will appear in the output even if Whisper
    smoothed them to "I".
    """
    if processor is None or model is None:
        return []
    
    import torch
    import torchaudio
    
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Resample to 16kHz if needed (Wav2Vec2 requirement)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)
        
        # Process in chunks (30 second windows) for memory efficiency
        chunk_duration = 30  # seconds
        chunk_samples = chunk_duration * 16000
        stutters = []
        
        for chunk_start in range(0, len(waveform), chunk_samples):
            chunk_end = min(chunk_start + chunk_samples, len(waveform))
            chunk = waveform[chunk_start:chunk_end]
            
            if len(chunk) < 1600:  # Skip very short chunks (< 0.1s)
                continue
            
            # Get transcription
            inputs = processor(chunk.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            
            if device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = model(inputs['input_values']).logits
            
            # Decode without language model (literal transcription)
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            
            # Find stutters in the transcription (repeated words/sounds)
            chunk_time_offset = chunk_start / 16000
            words = transcription.upper().split()
            
            # Detect consecutive repeated words
            i = 0
            while i < len(words) - 1:
                repeat_count = 1
                while i + repeat_count < len(words) and words[i] == words[i + repeat_count]:
                    repeat_count += 1
                
                if repeat_count >= 2:
                    # Estimate timing (rough approximation)
                    word_duration = (chunk_end - chunk_start) / 16000 / max(len(words), 1)
                    start_time = chunk_time_offset + i * word_duration
                    end_time = start_time + repeat_count * word_duration
                    stutter_text = ' '.join([words[i]] * repeat_count)
                    stutters.append((start_time, end_time, stutter_text))
                    logger.debug(f"Wav2Vec2 detected stutter at {start_time:.1f}s: '{stutter_text}'")
                    i += repeat_count
                else:
                    i += 1
        
        if stutters:
            logger.info(f"Wav2Vec2 detected {len(stutters)} acoustic stutters")
        
        return stutters
        
    except Exception as e:
        logger.warning(f"Wav2Vec2 stutter detection failed: {e}")
        return []


def apply_wav2vec_stutters(segments: List['TranscriptSegment'], stutters: List[Tuple[float, float, str]]) -> List['TranscriptSegment']:
    """Apply Wav2Vec2-detected stutters to segments by marking overlapping words for removal.
    
    For each stutter detected by Wav2Vec2, find the overlapping words in the segments
    and mark them with low probability so they get filtered out by word-level trimming.
    
    Args:
        segments: List of TranscriptSegment from Whisper
        stutters: List of (start_time, end_time, text) from Wav2Vec2
    
    Returns:
        Modified segments with stutter words marked for removal
    """
    if not stutters:
        return segments
    
    stutter_words_marked = 0
    
    for seg in segments:
        if not seg.words:
            continue
        
        for stutter_start, stutter_end, stutter_text in stutters:
            # Check if this stutter overlaps with this segment
            if stutter_end < seg.start or stutter_start > seg.end:
                continue
            
            # Find words that overlap with the stutter time range
            for word in seg.words:
                word_start = word.get('start', 0)
                word_end = word.get('end', 0)
                
                # Check for overlap
                if word_start < stutter_end and word_end > stutter_start:
                    # This word overlaps with a detected stutter
                    # Mark it with low probability so it gets filtered
                    original_prob = word.get('probability', 1.0)
                    if original_prob > 0.3:  # Don't re-mark already-marked words
                        word['probability'] = 0.1
                        word['wav2vec_stutter'] = True
                        stutter_words_marked += 1
                        logger.debug(f"Marked word '{word.get('word', '')}' at {word_start:.2f}s as stutter (Wav2Vec2)")
    
    if stutter_words_marked > 0:
        logger.info(f"Wav2Vec2: Marked {stutter_words_marked} words as stutters for removal")
    
    return segments


def fix_video_drift(input_path: Path, config: EditorConfig) -> Path:
    """Fix audio/video sync issues from OBS recordings and convert to CFR.
    
    OBS recordings often have audio drift due to CPU/GPU load during recording.
    This function applies audio delay and forces CFR for proper sync.
    Uses NVENC if available for speed.
    """
    output_path = config.temp_dir / "fixed_sync.mp4"
    
    # Get source video fps to preserve it
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)
    ]
    try:
        fps_str = subprocess.check_output(probe_cmd, text=True).strip()
        # fps_str is like "60/1" or "30000/1001"
        if "/" in fps_str:
            num, den = map(int, fps_str.split("/"))
            source_fps = num / den
        else:
            source_fps = float(fps_str)
        # Round to common values
        if source_fps > 55:
            target_fps = 60
        elif source_fps > 25:
            target_fps = 30
        else:
            target_fps = 24
    except Exception:
        target_fps = 30  # Default fallback
    
    # Build audio filter chain
    audio_filters = []
    
    # Apply audio delay if specified (for OBS sync fix)
    if config.audio_delay_ms > 0:
        # adelay delays audio - format is "delay_left|delay_right" in milliseconds
        audio_filters.append(f"adelay={config.audio_delay_ms}|{config.audio_delay_ms}")
        logger.info(f"Applying audio delay: {config.audio_delay_ms}ms")
    
    # Always add async resampling to handle any remaining drift
    audio_filters.append("aresample=async=1000:first_pts=0")
    
    audio_filter_str = ",".join(audio_filters)
    
    logger.info(f"Fixing A/V sync (delay={config.audio_delay_ms}ms, CFR {target_fps}fps): {input_path}")
    
    # Build ffmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", audio_filter_str,
        "-c:a", "aac", "-ar", "48000", "-b:a", "192k",
        "-r", str(target_fps),  # Force constant frame rate
        "-fps_mode", "cfr",
        "-max_muxing_queue_size", "4096",
        "-avoid_negative_ts", "make_zero",
    ]
    
    if config.device == "cuda":
        # Hardware accelerated encoding
        cmd.extend([
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-cq", "20",
        ])
    else:
        # CPU encoding
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
        ])
        
    cmd.append(str(output_path))
    
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"A/V sync fix complete: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to fix A/V sync: {e}")
        return input_path  # Fallback to original


def clean_whisper_artifacts(text: str) -> str:
    """Clean up Whisper transcription artifacts like repeated word endings.
    
    Whisper sometimes outputs artifacts like:
    - "button, tton" -> "button"
    - "framework, work" -> "framework"  
    - "decisions, sions" -> "decisions"
    - "the the" -> "the"
    
    Args:
        text: Raw transcription text
        
    Returns:
        Cleaned text with artifacts removed
    """
    import re
    
    # Pattern 1: Word followed by comma/space and partial repeat of its ending
    # e.g., "button, tton" or "framework, work" or "decisions, sions"
    # Match: word + comma/space + suffix that matches end of word (2+ chars)
    def remove_partial_repeats(match):
        return match.group(1)  # Return just the first word
    


def script_guided_word_cleanup(
    segments: List['TranscriptSegment'],
    script_text: str
) -> List['TranscriptSegment']:
    """Clean words using the script as the AUTHORITATIVE reference.
    
    For segments that MATCH the script:
    - Remove ANY word that is not in the matching script portion
    - "He Hell Hello World" with script "Hello World" → remove "He Hell"
    - This is STRICT enforcement of script
    
    For segments that are IMPROVISED (don't match script):
    - Only remove obvious stutters (immediate repeats)
    - Keep the improvised content intact
    
    Args:
        segments: List of transcript segments with word-level timing
        script_text: The original script text
        
    Returns:
        List of cleaned segments
    """
    import re
    from difflib import SequenceMatcher
    
    # Normalize script into word list for comparison
    script_words_raw = re.sub(r'[^\w\s]', ' ', script_text.lower()).split()
    script_words = [w for w in script_words_raw if len(w) > 0]
    
    cleaned_segments = []
    words_removed_scripted = 0
    words_removed_stutter = 0
    
    for seg in segments:
        if not seg.words:
            cleaned_segments.append(seg)
            continue
        
        # Get segment words normalized
        seg_word_texts = []
        for w in seg.words:
            if w and isinstance(w, dict) and 'word' in w:
                seg_word_texts.append(w['word'].strip().lower().replace('-', '').rstrip(',.!?'))
            else:
                seg_word_texts.append('')
        
        # Find the best matching portion of the script for this segment
        best_match_start = -1
        best_match_score = 0
        best_match_length = 0
        
        # Sliding window search - OPTIMIZED
        # Instead of O(N*M) sliding window, use SequenceMatcher on larger chunks
        # This is much faster for long scripts
        
        # Create a search window in the script around where we expect to be
        # (This assumes segments are roughly in order, but allows for jumping around)
        # For now, we'll just search the whole script but use a faster method
        
        seg_str = ' '.join(seg_word_texts)
        
        # Quick check: if segment is very short, skip expensive search
        if len(seg_word_texts) < 3:
            cleaned_segments.append(seg)
            continue
            
        # Use SequenceMatcher to find the best block match in the entire script
        # This is much faster than manual sliding window
        # We convert script to string for this search
        # (Note: this is an approximation but good enough for locating the region)
        
        # To make it feasible, we search in a window of +/- 1000 words from "current position"
        # But we don't track current position yet. 
        # Let's just use a simplified approach:
        # If the segment has high overlap with ANY part of the script
        
        # Optimization: Only check if we have a reasonable match
        # We can use difflib's get_matching_blocks but on words
        
        matcher = SequenceMatcher(None, seg_word_texts, script_words)
        match = matcher.find_longest_match(0, len(seg_word_texts), 0, len(script_words))
        
        # If the longest match covers most of the segment, it's scripted
        is_scripted = False
        if match.size > 0:
            coverage = match.size / len(seg_word_texts)
            if coverage > 0.4: # If 40% of words match a contiguous block
                is_scripted = True
                best_match_start = match.b
                best_match_length = match.size
                script_window = script_words[match.b : match.b + match.size]
        
        if is_scripted:
            # SCRIPTED SEGMENT: Enforce strict script compliance
            script_window_set = set(script_window)
            
            # Build expected sequence from script
            # Keep only words that appear in the script window, in order
            valid_words = []
            script_idx = 0  # Track position in script window
            
            for word_idx, w in enumerate(seg.words):
                if not w or not isinstance(w, dict) or 'word' not in w:
                    continue
                
                word_text = w['word'].strip().lower().replace('-', '').rstrip(',.!?')
                
                # Check if this word matches the next expected script word
                word_in_script = word_text in script_window_set
                
                # Check if it's the next word in sequence (or close)
                is_next_in_sequence = False
                if script_idx < len(script_window):
                    # Allow looking ahead a bit for flexibility
                    for look_ahead in range(min(3, len(script_window) - script_idx)):
                        if script_window[script_idx + look_ahead] == word_text:
                            is_next_in_sequence = True
                            script_idx = script_idx + look_ahead + 1
                            break
                
                if word_in_script and is_next_in_sequence:
                    valid_words.append(w)
                else:
                    # This word is NOT in the script - it's a stutter/false start
                    words_removed_scripted += 1
                    logger.debug(f"Script enforcement removed: '{word_text}' (not in script sequence)")
            
            if valid_words:
                # Reconstruct segment
                first_w = valid_words[0]
                last_w = valid_words[-1]
                if first_w.get('start') is not None and last_w.get('end') is not None:
                    new_start = first_w['start']
                    new_end = last_w['end']
                    new_text = ' '.join(w.get('word', '') for w in valid_words).strip()
                    seg = TranscriptSegment(
                        start=new_start,
                        end=new_end,
                        text=new_text,
                        words=valid_words
                    )
                    if len(valid_words) < len([w for w in seg.words if w]):
                        logger.info(f"Script enforced: kept {len(valid_words)} words, removed stutters")
        else:
            # IMPROVISED SEGMENT: Only remove stutters and filler words
            # NO script enforcement - keep the improvised content intact
            valid_words = []
            i = 0
            
            # Filler words to remove (CrisperWhisper markers + common fillers)
            filler_words = {'[uh]', '[um]', 'uh', 'um', 'uhh', 'umm', 'uhm', 'er', 'err', 'ah', 'ahh'}
            
            while i < len(seg.words):
                curr_w = seg.words[i]
                if not curr_w or not isinstance(curr_w, dict) or 'word' not in curr_w:
                    i += 1
                    continue
                
                word_text = curr_w['word'].strip().lower().replace('-', '').rstrip(',.')
                is_stutter = False
                
                # Remove filler words
                if word_text in filler_words:
                    is_stutter = True
                    words_removed_stutter += 1
                    logger.debug(f"Filler removed: '{word_text}'")
                
                # Check for immediate repeat (the the -> the)
                if not is_stutter and i < len(seg.words) - 1:
                    next_w = seg.words[i + 1]
                    if next_w and isinstance(next_w, dict) and 'word' in next_w:
                        next_text = next_w['word'].strip().lower().replace('-', '').rstrip(',.')
                        if word_text == next_text and len(word_text) > 1:
                            is_stutter = True
                            words_removed_stutter += 1
                
                # Check for partial word stutter (th -> the)
                if not is_stutter and i < len(seg.words) - 1:
                    next_w = seg.words[i + 1]
                    if next_w and isinstance(next_w, dict) and 'word' in next_w:
                        next_text = next_w['word'].strip().lower().replace('-', '').rstrip(',.')
                        if (next_text.startswith(word_text) and 
                            len(word_text) < len(next_text) and 
                            len(word_text) <= 3):
                            is_stutter = True
                            words_removed_stutter += 1
                
                if not is_stutter:
                    valid_words.append(curr_w)
                i += 1
            
            if valid_words and len(valid_words) < len([w for w in seg.words if w]):
                first_w = valid_words[0]
                last_w = valid_words[-1]
                if first_w.get('start') is not None and last_w.get('end') is not None:
                    new_start = first_w['start']
                    new_end = last_w['end']
                    new_text = ' '.join(w.get('word', '') for w in valid_words).strip()
                    seg = TranscriptSegment(
                        start=new_start,
                        end=new_end,
                        text=new_text,
                        words=valid_words
                    )
        
        if seg.words:  # Only add if segment still has content
            cleaned_segments.append(seg)
    
    if words_removed_scripted > 0 or words_removed_stutter > 0:
        logger.info(f"Script-guided cleanup: {words_removed_scripted} words removed (script enforcement), {words_removed_stutter} stutters removed (improvised)")
    
    return cleaned_segments


def detect_word_level_retakes(segment: 'TranscriptSegment') -> Optional[Tuple[float, float]]:
    """Detect in-segment retakes and return the trimmed time range.
    
    Looks for repeated phrases within a segment's word list and returns
    the time range that keeps only the LAST (best) version.
    
    Examples:
    - "and that even left and that even left a scar" -> trim to "and that even left a scar"
    - "to where is our money to our to where is our money" -> trim to "to where is our money"
    
    Returns:
        Tuple of (new_start, new_end) if retake detected, None otherwise
    """
    if not segment.words or len(segment.words) < 4:
        return None
    
    # Defensive: filter out malformed word entries
    valid_word_objs = [w for w in segment.words if w and isinstance(w, dict) and 'word' in w]
    if len(valid_word_objs) < 4:
        return None
    
    words = [w['word'].strip().lower() for w in valid_word_objs]
    
    # Look for repeated sequences of 2-5 words
    for seq_len in [5, 4, 3, 2]:
        for i in range(len(words) - seq_len * 2 + 1):
            seq1 = words[i:i + seq_len]
            # Look for this sequence appearing again later
            for j in range(i + seq_len, len(words) - seq_len + 1):
                seq2 = words[j:j + seq_len]
                if seq1 == seq2:
                    # Found a repeat! Keep from the second occurrence onwards
                    # But include some words before if they're part of the sentence
                    new_start_idx = j
                    
                    # If the repeat is at the start of a natural phrase, keep it
                    # Otherwise back up to include connector words
                    if new_start_idx > 0 and words[new_start_idx - 1] in ['and', 'but', 'so', 'or', 'the', 'a']:
                        new_start_idx -= 1
                    
                    # Safely get start time from valid word objects
                    if new_start_idx >= len(valid_word_objs):
                        continue
                    start_word = valid_word_objs[new_start_idx]
                    if not start_word.get('start'):
                        continue
                    new_start = start_word['start']
                    new_end = segment.end
                    
                    # Calculate what percentage of content we'd keep
                    kept_duration = new_end - new_start
                    original_duration = segment.end - segment.start
                    kept_words = len(words) - new_start_idx
                    
                    # Only trim if:
                    # 1. Removing at least 0.3 seconds (was 0.5)
                    # 2. Keeping at least 40% of original words (was 70% - too conservative)
                    # 3. The repeated sequence is substantial (sequence itself is short)
                    trim_amount = new_start - segment.start
                    if trim_amount > 0.3 and kept_words >= len(words) * 0.4:
                        logger.debug(f"Word-level retake detected: trimming {segment.start:.1f}s-{new_start:.1f}s")
                        return (new_start, new_end)
                    elif trim_amount > 0.3:
                        # Log skipped trim for debugging
                        logger.debug(f"Word-level retake SKIPPED (would keep only {kept_words}/{len(words)} words): {segment.text[:50]}...")
    
    return None


def trim_segment_retakes(segments: List['TranscriptSegment']) -> List['TranscriptSegment']:
    """Apply word-level retake trimming to all segments.
    
    Three-pass approach:
    1. Detect phrase-level retakes (existing logic)  
    2. Clean repeated word sequences like "their, their, their" -> "their"
    3. Filter garbage words (very short + low probability)
    
    Returns new list of segments with in-segment retakes trimmed.
    """
    trimmed = []
    trimmed_count = 0
    stutter_words_removed = 0
    
    for seg in segments:
        # Initialize words_to_remove for each segment (used by Pass 2 and Pass 3)
        words_to_remove = set()
        
        # Pass 1: Detect phrase-level retakes
        result = detect_word_level_retakes(seg)
        if result:
            new_start, new_end = result
            # Create trimmed segment - with defensive checks
            if seg.words:
                new_words = [w for w in seg.words if w and isinstance(w, dict) and w.get('start') is not None and w['start'] >= new_start]
                if new_words:
                    new_text = ' '.join(w.get('word', '') for w in new_words)
                    seg = TranscriptSegment(
                        start=new_start,
                        end=new_end,
                        text=new_text.strip(),
                        words=new_words
                    )
                    trimmed_count += 1
        
        # Pass 2: Detect and remove TRUE PHRASE REPEATS (e.g., "I've been thinking I've been thinking")
        # STRICT: Only remove when EXACT same phrase appears IMMEDIATELY CONSECUTIVELY
        # Do NOT remove common phrases like "a billion" that might appear multiple times in different contexts
        if seg.words:
            # Helper function to safely get word text
            def safe_word_text(w):
                if not w or not isinstance(w, dict) or 'word' not in w:
                    return ''
                return w['word'].strip().lower().replace('-', '').rstrip(',.')
            
            # Check for 2-4 word phrase repeats - must be IMMEDIATELY consecutive
            for phrase_len in [4, 3, 2]:  # Start with longer phrases
                i = 0
                while i <= len(seg.words) - phrase_len * 2:
                    # Get phrase at position i
                    phrase1_words = [safe_word_text(seg.words[j])
                                     for j in range(i, i + phrase_len)]
                    phrase1 = ' '.join(phrase1_words)
                    # Get phrase IMMEDIATELY after (at position i + phrase_len)
                    phrase2_words = [safe_word_text(seg.words[j])
                                     for j in range(i + phrase_len, i + phrase_len * 2)]
                    phrase2 = ' '.join(phrase2_words)
                    
                    # STRICT MATCHING:
                    # 1. Exact match
                    # 2. Minimum phrase length (avoid matching "a a", "the the" here - that's Pass 3)
                    # 3. NOT a common phrase that might repeat legitimately
                    common_phrases = {'a', 'the', 'and', 'or', 'is', 'are', 'was', 'were', 'to', 'for', 'of', 'in', 'on',
                                      'it', 'that', 'this', 'with', 'as', 'at', 'by', 'from', 'be', 'an'}
                    # Skip if first word is a common word and phrase is only 2 words
                    if phrase_len == 2 and phrase1_words[0] in common_phrases:
                        i += 1
                        continue
                    
                    if phrase1 == phrase2 and len(phrase1.replace(' ', '')) > 5:  # Require longer match
                        # Found a TRUE phrase repeat! Mark the FIRST occurrence for removal
                        for j in range(i, i + phrase_len):
                            if j not in words_to_remove:
                                words_to_remove.add(j)
                                stutter_words_removed += 1
                        logger.info(f"Phrase repeat removed: '{phrase1}' at word indices {i}-{i+phrase_len-1}")
                        i += phrase_len * 2  # Skip past BOTH occurrences
                    else:
                        i += 1
        
        # Pass 3: Clean single repeated words and garbage words
        if seg.words:
            valid_words = []
            i = 0
            while i < len(seg.words):
                # Skip words already marked for removal by phrase detection
                if i in words_to_remove:
                    i += 1
                    continue
                    
                curr_w = seg.words[i]
                # Defensive check for malformed word entries
                if not curr_w or not isinstance(curr_w, dict) or 'word' not in curr_w:
                    i += 1
                    continue
                
                word_text = curr_w['word'].strip().lower().replace('-', '').rstrip(',')
                word_duration = (curr_w.get('end', 0) or 0) - (curr_w.get('start', 0) or 0)
                word_prob = curr_w.get('probability') or 1.0  # Handle None probability
                
                is_stutter = False
                
                # Check for repeated word (current word same as next, and next not marked for removal)
                if i < len(seg.words) - 1 and (i + 1) not in words_to_remove:
                    next_w = seg.words[i+1]
                    # Defensive check for next word
                    if next_w and isinstance(next_w, dict) and 'word' in next_w:
                        next_text = next_w['word'].strip().lower().replace('-', '').rstrip(',')
                        
                        # DETECT STUTTER PAIR:
                        # 1. Exact match (the, the) or (their, their)
                        is_repeat = word_text == next_text and len(word_text) > 1
                        # 2. Partial match - current is prefix of next (th -> the)
                        is_partial = (next_text.startswith(word_text) and 
                                      len(word_text) < len(next_text) and 
                                      len(word_text) <= 3)
                        
                        if is_repeat or is_partial:
                            is_stutter = True
                            stutter_words_removed += 1
                            logger.debug(f"Stutter word removed: '{word_text}' (repeat of next word)")
                
                # Check for garbage word (very short + low probability)
                if not is_stutter and word_duration < 0.15 and word_prob < 0.4:
                    is_stutter = True
                    stutter_words_removed += 1
                    logger.debug(f"Garbage word removed: '{word_text}' (dur={word_duration:.2f}s, prob={word_prob:.2f})")
                
                if not is_stutter:
                    valid_words.append(curr_w)
                i += 1
            
            if not valid_words:
                # Segment was entirely stutters - skip it
                continue
            
            if len(valid_words) < len(seg.words):
                # Reconstruct segment with filtered words - with defensive checks
                first_w = valid_words[0]
                last_w = valid_words[-1]
                if first_w.get('start') is not None and last_w.get('end') is not None:
                    new_start = first_w['start']
                    new_end = last_w['end']
                    new_text = ' '.join(w.get('word', '') for w in valid_words).strip()
                    seg = TranscriptSegment(
                        start=new_start,
                        end=new_end,
                        text=new_text,
                        words=valid_words
                    )
        
        trimmed.append(seg)
    
    if trimmed_count > 0 or stutter_words_removed > 0:
        logger.info(f"Word-level trimming: {trimmed_count} segments had retakes, {stutter_words_removed} stutter words removed")
    
    return trimmed


def _get_transcription_cache_path(original_video_path: Path) -> Path:
    """Generate a cache file path based on original video file.
    
    Uses a hash of video path + file size + modification time to detect changes.
    Cache is stored next to the ORIGINAL video file (not temp files).
    """
    # Create a unique identifier based on video path, size, and mtime
    video_stat = original_video_path.stat()
    cache_key = f"{original_video_path.absolute()}|{video_stat.st_size}|{video_stat.st_mtime}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    
    # Store cache next to the ORIGINAL video file for persistence
    cache_filename = f".{original_video_path.stem}_{cache_hash}.transcription_cache.json"
    return original_video_path.parent / cache_filename


def _load_transcription_cache(cache_path: Path) -> Optional[List[TranscriptSegment]]:
    """Load cached transcription if it exists and is valid."""
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct TranscriptSegment objects
        segments = []
        for seg_data in data.get('segments', []):
            segments.append(TranscriptSegment(
                start=seg_data['start'],
                end=seg_data['end'],
                text=seg_data.get('text') or "",
                words=seg_data.get('words') or [],
            ))
        
        logger.info(f"✓ Loaded {len(segments)} segments from transcription cache")
        return segments
    except Exception as e:
        logger.warning(f"Failed to load transcription cache: {e}")
        return None


def _save_transcription_cache(cache_path: Path, segments: List[TranscriptSegment]) -> None:
    """Save transcription to cache file."""
    try:
        data = {
            'segments': [
                {
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text,
                    'words': seg.words,
                }
                for seg in segments
            ]
        }
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"✓ Saved transcription cache to {cache_path.name}")
    except Exception as e:
        logger.warning(f"Failed to save transcription cache: {e}")


def _normalize_segments(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """Ensure TranscriptSegment fields are well-formed (no None text/words)."""
    for seg in segments:
        if seg.text is None:
            seg.text = ""
        # Ensure words is a list for downstream processing
        if seg.words is None:
            seg.words = []
        elif not isinstance(seg.words, list):
            seg.words = list(seg.words)
        # Remove any malformed word entries
        cleaned_words = []
        for word in seg.words:
            if isinstance(word, dict) and 'word' in word:
                cleaned_words.append(word)
        seg.words = cleaned_words
    return segments


def transcribe_video(video_path: Path, config: EditorConfig) -> List[TranscriptSegment]:
    """Transcribe video using faster-whisper with word-level timestamps.
    
    Uses caching to avoid re-transcribing the same video. Cache is based on 
    the ORIGINAL input video (config.input_path), so it persists across runs
    even when processing temp files like silence-cut versions.
    
    Args:
        video_path: Path to video file (may be temp file)
        config: Editor configuration (contains original input_path)
        
    Returns:
        List of TranscriptSegment with word-level timing
    """
    # Use ORIGINAL video path for cache (not the temp silence-cut file)
    # This makes the cache persistent across runs on the same video
    original_path = config.input_path
    cache_path = _get_transcription_cache_path(original_path)
    cached_segments = _load_transcription_cache(cache_path)
    if cached_segments is not None:
        return _normalize_segments(cached_segments)
    
    logger.info(f"Transcribing: {video_path} (no cache found, this may take a few minutes...)")
    
    model = load_whisper_model(config.whisper_model, config.device)
    
    # Extract audio to temp file for faster processing
    # Audio enhancement (noise reduction + normalization) improves transcription accuracy
    audio_path = config.temp_dir / "audio.wav"
    extract_audio(video_path, audio_path, enhance=config.enhance_audio, temp_dir=config.temp_dir)
    
    # Transcribe with word timestamps
    # VAD tuned for natural speech - less aggressive to preserve breathing room
    segments_iter, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language="en",
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,  # Don't cut silences shorter than 500ms
            speech_pad_ms=200,            # 200ms padding around speech
        ),
        condition_on_previous_text=False,
        temperature=0.0,
    )
    
    logger.info(f"Detected language: {info.language} (prob: {info.language_probability:.2f})")
    
    segments = []
    for seg in tqdm(segments_iter, desc="Transcribing", unit="seg"):
        words = []
        if seg.words:
            words = [
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                for w in seg.words
            ]
        
        # Clean up Whisper artifacts (repeated word endings like "button, tton")
        cleaned_text = clean_whisper_artifacts(seg.text.strip())
        
        segments.append(TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=cleaned_text,
            words=words,
        ))
    
    logger.info(f"Transcribed {len(segments)} segments, {sum(s.duration for s in segments):.1f}s total")
    
    # Cache the transcription for future runs
    _save_transcription_cache(cache_path, segments)
    
    return _normalize_segments(segments)


def extract_audio(video_path: Path, output_path: Path, enhance: bool = True, temp_dir: Optional[Path] = None) -> None:
    """Extract audio from video, optionally clean noise and normalize volume.
    
    Args:
        video_path: Input video file
        output_path: Output audio file (WAV)
        enhance: If True, apply noise reduction and normalization
        temp_dir: Temp directory for intermediate files
    """
    # Use a temp directory if provided, otherwise use output's parent
    work_dir = temp_dir or output_path.parent
    
    # 1. Extract Raw Audio via FFmpeg
    # Use 44100Hz for better quality during processing, will be resampled if needed
    temp_raw = work_dir / "raw_audio_extract.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(temp_raw)
    ]
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found for audio extraction: {video_path}")
        
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else "No stderr output"
        logger.error(f"FFmpeg audio extraction failed: {error_msg}")
        raise

    if not enhance or not AUDIO_ENHANCE_AVAILABLE:
        # No enhancement, just move the file
        if temp_raw != output_path:
            shutil.move(str(temp_raw), str(output_path))
        return
    
    logger.info("Enhancing audio (noise reduction + normalization)...")
    
    # 2. Spectral Noise Reduction (noisereduce)
    # Good for: Fans, AC, consistent room hum
    # prop_decrease=0.75: Only remove 75% of noise to avoid "underwater robot" voice
    try:
        rate, data = wavfile.read(str(temp_raw))
        
        # Convert to float for processing if needed
        if data.dtype == np.int16:
            data_float = data.astype(np.float32) / 32768.0
        else:
            data_float = data.astype(np.float32)
        
        reduced_noise = nr.reduce_noise(
            y=data_float, 
            sr=rate, 
            prop_decrease=0.75,  # Keep 25% of noise to sound natural
            stationary=True,     # Optimize for constant noise (fan/AC)
            n_std_thresh_stationary=1.5,
        )
        
        # Convert back to int16
        reduced_noise_int = (reduced_noise * 32768.0).astype(np.int16)
        
        temp_denoised = work_dir / "denoised_audio.wav"
        wavfile.write(str(temp_denoised), rate, reduced_noise_int)
        
        # Cleanup raw
        temp_raw.unlink(missing_ok=True)
        logger.debug("Noise reduction complete")
        
    except Exception as e:
        logger.warning(f"Noise reduction failed: {e}. Using raw audio.")
        temp_denoised = temp_raw

    # 3. Pedalboard Enhancement (professional voice processing)
    # Applies: EQ for warmth/clarity, compression for even dynamics, limiting for peaks
    if PEDALBOARD_AVAILABLE:
        try:
            logger.debug("Applying pedalboard voice enhancement...")
            rate_pd, data_pd = wavfile.read(str(temp_denoised))
            
            # Convert to float32 for pedalboard
            if data_pd.dtype == np.int16:
                audio_float = data_pd.astype(np.float32) / 32768.0
            else:
                audio_float = data_pd.astype(np.float32)
            
            # Reshape for pedalboard: (channels, samples)
            if audio_float.ndim == 1:
                audio_float = audio_float.reshape(1, -1)  # Mono
            elif audio_float.ndim == 2 and audio_float.shape[0] > audio_float.shape[1]:
                audio_float = audio_float.T  # (samples, channels) -> (channels, samples)
            
            # Build professional voice processing chain
            board = Pedalboard([
                # Low shelf boost for warmth (subtle, like a good mic preamp)
                LowShelfFilter(cutoff_frequency_hz=200, gain_db=2.0),
                # High shelf for voice clarity/presence
                HighShelfFilter(cutoff_frequency_hz=3000, gain_db=1.5),
                # Gentle compression to even out volume (podcast/YouTube style)
                Compressor(
                    threshold_db=-20,
                    ratio=3.0,
                    attack_ms=10,
                    release_ms=100,
                ),
                # Makeup gain after compression
                Gain(gain_db=3.0),
                # Limiter to prevent clipping
                Limiter(threshold_db=-1.0, release_ms=100),
            ])
            
            # Process audio
            enhanced = board(audio_float, rate_pd)
            
            # Transpose back if needed: (channels, samples) -> (samples,) for mono
            if enhanced.ndim == 2:
                if enhanced.shape[0] == 1:
                    enhanced = enhanced.squeeze(0)  # Remove channel dim for mono
                else:
                    enhanced = enhanced.T  # (channels, samples) -> (samples, channels)
            
            # Convert back to int16
            enhanced_int = np.clip(enhanced * 32768.0, -32768, 32767).astype(np.int16)
            
            temp_enhanced = work_dir / "enhanced_audio.wav"
            wavfile.write(str(temp_enhanced), rate_pd, enhanced_int)
            temp_denoised.unlink(missing_ok=True)
            temp_denoised = temp_enhanced
            logger.debug("Pedalboard voice enhancement complete")
            
        except Exception as e:
            logger.warning(f"Pedalboard enhancement failed: {e}. Continuing with denoised audio.")
    
    # 4. Volume Normalization (pydub)
    # Ensures voice is at standard YouTube loudness (-1 dB Peak)
    try:
        audio = AudioSegment.from_wav(str(temp_denoised))
        
        # Normalize to -1 dBFS (standard safe peak for YouTube)
        normalized = pydub_effects.normalize(audio, headroom=1.0)
        
        normalized.export(str(output_path), format="wav")
        temp_denoised.unlink(missing_ok=True)
        logger.debug("Volume normalization complete")
        
    except Exception as e:
        logger.warning(f"Normalization failed: {e}. Using denoised audio.")
        # Fallback: just move the file
        if temp_denoised.exists() and temp_denoised != output_path:
            shutil.move(str(temp_denoised), str(output_path))


# =============================================================================
# MODE 1: SILENCE REMOVAL (auto-editor)
# =============================================================================

def cut_silence_auto_editor(config: EditorConfig) -> Path:
    """Remove silence using auto-editor CLI.
    
    Args:
        config: Editor configuration
        
    Returns:
        Path to the cut video file
    """
    logger.info("Running auto-editor for silence removal...")
    
    output_path = config.temp_dir / f"cut_{config.input_path.name}"
    
    cmd = [
        "auto-editor",
        str(config.input_path),
        "--silent-speed", "9999",
        "--video-speed", "1",
        "--output", str(output_path),
        "--no-open",
    ]
    
    try:
        # Check if auto-editor is installed
        subprocess.run(["auto-editor", "--help"], capture_output=True, check=True)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("auto-editor completed successfully")
        if result.stdout:
            logger.debug(f"stdout: {result.stdout}")
        
        # Find output file (auto-editor may modify name)
        if output_path.exists():
            return output_path
        
        # Check for alternative output names
        for f in config.temp_dir.glob(f"cut_*{config.input_path.suffix}"):
            return f
        
        raise FileNotFoundError("auto-editor output not found")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"auto-editor failed: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("auto-editor not found. Install with: pip install auto-editor")
        raise


def cut_silence_ffmpeg(input_path: Path, config: EditorConfig) -> Path:
    """Remove silence using ffmpeg silencedetect filter.
    
    Args:
        input_path: Input video path
        config: Editor configuration
        
    Returns:
        Path to video with silence removed
    """
    logger.info("Detecting and removing silence with ffmpeg...")
    output_path = config.temp_dir / "silence_cut.mp4"
    
    # 1. Extract audio only for faster silence detection (no video decoding needed)
    # This is MUCH faster than analyzing video+audio together
    audio_temp = config.temp_dir / "temp_audio_for_silence.wav"
    logger.info("Extracting audio for fast silence analysis...")
    extract_cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # Fast uncompressed audio
        "-ar", "16000",  # Lower sample rate for faster processing
        "-ac", "1",  # Mono
        str(audio_temp)
    ]
    try:
        subprocess.run(extract_cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Audio extraction failed, falling back to full video: {e}")
        audio_temp = input_path  # Fallback to original
    
    # 2. Detect silence on audio (or original if extraction failed)
    # CONSERVATIVE settings to only cut LONG silences, not natural pauses:
    # - noise=-40dB: Only true silence (quieter threshold)
    # - d=1.5s: Only silences longer than 1.5 seconds (keeps natural sentence pauses)
    noise_floor = -40  # More conservative - only true silence
    min_silence = max(1.5, config.silence_duration)  # At least 1.5s - preserve natural pauses
    cmd_detect = [
        "ffmpeg", "-i", str(audio_temp),
        "-af", f"silencedetect=noise={noise_floor}dB:d={min_silence}",
        "-f", "null", "-"
    ]
    
    try:
        result = subprocess.run(cmd_detect, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Silence detection failed: {e}")
        if audio_temp != input_path:
            audio_temp.unlink(missing_ok=True)
        return input_path
    
    # Cleanup temp audio
    if audio_temp != input_path:
        audio_temp.unlink(missing_ok=True)

    # Parse silence timestamps
    silence_starts = []
    silence_ends = []
    for line in result.stderr.splitlines():
        if "silence_start" in line:
            silence_starts.append(float(line.split("silence_start: ")[1]))
        elif "silence_end" in line:
            silence_ends.append(float(line.split("silence_end: ")[1].split(" ")[0]))
            
    if not silence_starts:
        logger.info("No silence detected.")
        return input_path
        
    # Create keep segments with PADDING to preserve natural flow
    # If silence is [s1, e1], [s2, e2]...
    # Keep segments are [0, s1+pad], [e1-pad, s2+pad], [e2-pad, end]
    # Padding leaves a small silence at start/end of each segment for natural breathing room
    
    # IMPORTANT: Use generous padding to preserve word beginnings and endings
    # 350ms ensures we don't cut into speech - natural pauses between sentences
    SILENCE_PADDING = 0.35  # 350ms padding - prevents cutting word beginnings/endings
    
    keep_segments = []
    current_pos = 0.0
    
    for start, end in zip(silence_starts, silence_ends):
        # Add padding: extend INTO the silence (away from speech)
        # padded_start = end of keep segment (push INTO the silence gap)
        # padded_end = start of next keep segment (start earlier FROM the silence gap)
        padded_start = start + SILENCE_PADDING  # End keep segment a bit INTO the silence
        padded_end = end - SILENCE_PADDING  # Start next keep segment a bit BEFORE silence ends
        
        if padded_start > current_pos:
            keep_segments.append((current_pos, padded_start))
        current_pos = max(padded_end, current_pos)  # Don't go backwards
        
    # Add final segment
    # We don't know exact duration, but we can just use a large number or check duration
    # Better to check duration
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)
    ]
    total_duration = float(subprocess.check_output(duration_cmd, text=True).strip())
    
    if current_pos < total_duration:
        keep_segments.append((current_pos, total_duration))
        
    logger.info(f"Found {len(silence_starts)} silence gaps (>1.5s). Keeping {len(keep_segments)} segments.")
    
    # Create concat file
    concat_file = config.temp_dir / "silence_concat.txt"
    segment_files = []
    
    codec = "h264_nvenc" if config.device == "cuda" else "libx264"
    
    for i, (start, end) in enumerate(tqdm(keep_segments, desc="Cutting silence")):
        duration = end - start
        if duration < 0.1:
            continue  # Skip tiny segments

        seg_path = config.temp_dir / f"silence_seg_{i:04d}.mp4"
        segment_files.append(seg_path)
        
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-i", str(input_path),
            "-t", str(duration),
            "-c:v", codec, "-preset", "fast", "-cq", "19",
            "-c:a", "aac",
            "-avoid_negative_ts", "make_zero",
            str(seg_path)
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cut segment {i}: {e.stderr.decode()}")
            raise
        
    if not segment_files:
        logger.warning("Silence cutting produced no segments; returning original video.")
        return input_path

    with open(concat_file, "w") as f:
        for seg_path in segment_files:
            f.write(f"file '{seg_path.as_posix()}'\n")
            
    # Concatenate with re-encode to guarantee monotonic timestamps
    concat_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
    ]

    if codec == "h264_nvenc":
        concat_cmd += [
            "-c:v", codec,
            "-preset", "fast",
            "-cq", "19",
        ]
    else:
        concat_cmd += [
            "-c:v", codec,
            "-preset", "fast",
            "-crf", "20",
        ]

    concat_cmd += [
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    try:
        subprocess.run(concat_cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Silence concat failed: {e.stderr.decode()}")
        raise
    
    # Cleanup
    for seg_path in segment_files:
        seg_path.unlink(missing_ok=True)
    concat_file.unlink(missing_ok=True)
    
    return output_path


# =============================================================================
# MODE 2: RETAKE DETECTION (Script-Guided + Fuzzy Matching)
# =============================================================================

def load_script(script_path: Path) -> List[str]:
    """Load script file and split into sentences."""
    with open(script_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into sentences (handle multiple punctuation types)
    import re
    # Split on sentence-ending punctuation followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Clean up and filter empty
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    logger.info(f"Loaded script with {len(sentences)} sentences")
    return sentences


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove extra spaces, basic cleanup)."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    return text


def word_level_similarity(text1: str, text2: str, max_word_errors: int = 2) -> float:
    """Calculate similarity allowing for Whisper transcription errors (1-2 words wrong).
    
    Uses word-level comparison to be more forgiving of individual word errors.
    E.g., "Mt. Gox" transcribed as "mind main goss" still matches if context is right.
    """
    words1 = normalize_text(text1).split()
    words2 = normalize_text(text2).split()
    
    if not words1 or not words2:
        return 0.0
    
    # Use difflib on words instead of characters
    matcher = difflib.SequenceMatcher(None, words1, words2)
    
    # Get matching blocks
    matches = matcher.get_matching_blocks()
    matched_words = sum(block.size for block in matches)
    
    # Calculate ratio based on matched words
    total_words = max(len(words1), len(words2))
    word_ratio = matched_words / total_words if total_words > 0 else 0
    
    # Also get character-level ratio for short segments
    char_ratio = difflib.SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()
    
    # Use the better of the two
    return max(word_ratio, char_ratio)


def find_script_match(segment_text: str, script_sentences: List[str], threshold: float = 0.6) -> Tuple[Optional[str], float]:
    """Find the best matching script sentence for a transcript segment.
    
    Uses word-level matching to handle Whisper transcription errors.
    
    Returns:
        Tuple of (matched_sentence, similarity_score) or (None, 0) if no match
    """
    norm_segment = normalize_text(segment_text)
    if len(norm_segment) < 10:
        return None, 0.0
    
    best_match = None
    best_score = 0.0
    
    for sentence in script_sentences:
        norm_sentence = normalize_text(sentence)
        
        # Use word-level similarity (more forgiving of Whisper errors)
        score = word_level_similarity(segment_text, sentence)
        
        # Also check if segment is a substantial substring (partial match for long sentences)
        segment_words = norm_segment.split()
        sentence_words = norm_sentence.split()
        
        # Check for overlapping word sequences (handles partial sentence matches)
        if len(segment_words) >= 5:
            # Look for 5+ consecutive matching words
            for i in range(len(sentence_words) - 4):
                window = ' '.join(sentence_words[i:i+5])
                if window in norm_segment:
                    score = max(score, 0.75)
                    break
        
        if score > best_score:
            best_score = score
            best_match = sentence
    
    if best_score >= threshold:
        return best_match, best_score
    return None, 0.0


def is_incomplete_sentence(text: str) -> bool:
    """Detect if a sentence is incomplete (cuts off mid-thought).
    
    Examples:
    - "This history of failure and rebirth is why the landscape..." -> True
    - "It isn't a massive list of coins or complex trading tools." -> False (complete)
    """
    text = text.strip()
    
    # Ends with ellipsis or incomplete markers
    if text.endswith('...') or text.endswith('..'):
        return True
    
    # Very short and doesn't end with punctuation
    words = text.split()
    if len(words) < 5 and not text[-1] in '.!?':
        return True
    
    # Ends with articles/prepositions (incomplete thought)
    incomplete_endings = ['the', 'a', 'an', 'to', 'of', 'for', 'in', 'on', 'with', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'that', 'which']
    last_word = words[-1].lower().rstrip('.,!?') if words else ''
    if last_word in incomplete_endings and len(words) > 3:
        return True
    
    return False


def has_internal_stutter(text: str) -> bool:
    """Detect stutters/retakes WITHIN a single segment.
    
    Examples:
    - "actually Bybit and some other, actually Bybit and some other" -> True
    - "that pr that promises" -> True
    - "the the button" -> True
    - "I think I think we should" -> True
    - "[UH] [UH] the" -> True (CrisperWhisper filler markers)
    """
    # Remove CrisperWhisper filler markers for word analysis
    # but detect excessive fillers
    filler_count = text.count('[UH]') + text.count('[UM]')
    clean_text = text.replace('[UH]', '').replace('[UM]', '')
    
    words = clean_text.lower().split()
    if len(words) < 4:
        # Short segment with multiple fillers = likely a stutter/false start
        if filler_count >= 2 and len(words) < 6:
            return True
        return False
    
    # Check for immediate word repeats (the the, I I)
    for i in range(len(words) - 1):
        if words[i] == words[i+1] and len(words[i]) > 1:
            return True
    
    # Check for 2-4 word phrase repeats (more common stutter pattern)
    for phrase_len in [4, 3, 2]:
        for i in range(len(words) - phrase_len * 2 + 1):
            phrase = ' '.join(words[i:i+phrase_len])
            # Look for this exact phrase later in the text
            rest_start = i + phrase_len
            rest_words = words[rest_start:]
            rest = ' '.join(rest_words)
            if phrase in rest:
                # Found a repeat - this is a stutter
                return True
    
    # Check for "word1 word2 ... word1 word2" pattern with gap
    for i in range(len(words) - 1):
        pair = (words[i], words[i+1])
        for j in range(i + 2, len(words) - 1):
            if (words[j], words[j+1]) == pair:
                return True
    
    return False


def analyze_with_script(
    segments: List[TranscriptSegment],
    script_sentences: List[str],
    match_threshold: float = 0.5,
) -> List[EditDecision]:
    """Analyze segments using the original script as reference.
    
    PHILOSOPHY: 
    - Script is the REFERENCE for what should be in the video
    - For SCRIPTED lines: Keep only the LAST (best) take, cut earlier attempts
    - For IMPROVISATION: Keep ALL (valuable extra content)
    - Word-level stutters ("He Hell Hello") are cleaned by script_guided_word_cleanup()
    
    Algorithm:
    1. Match each segment to script sentences
    2. Group segments by which script sentence they match
    3. For each script sentence with multiple takes: keep ONLY the LAST take
    4. Segments that don't match any script = improvisation = KEEP ALL
    
    Returns:
        List of EditDecision
    """
    logger.info(f"Script-guided analysis: {len(segments)} segments, {len(script_sentences)} script sentences")
    
    # Initialize all decisions as KEEP (will update for bad takes)
    decisions = []
    for seg in segments:
        decisions.append(EditDecision(
            start=seg.start,
            end=seg.end,
            keep=True,
            reason="Analysis pending"
        ))
    
    # Map segments to script sentences
    # script_matches[script_idx] = list of (segment_idx, score, segment)
    script_matches = {}
    improvised_count = 0
    
    for seg_idx, seg in enumerate(segments):
        match_sentence, score = find_script_match(seg.text, script_sentences, match_threshold)
        
        if match_sentence and score >= match_threshold:
            # Find which script sentence index this matches
            try:
                script_idx = script_sentences.index(match_sentence)
                if script_idx not in script_matches:
                    script_matches[script_idx] = []
                script_matches[script_idx].append((seg_idx, score, seg))
                logger.debug(f"Segment {seg_idx} matches script line {script_idx+1} (score={score:.0%})")
            except ValueError:
                # Shouldn't happen, but treat as improvisation
                decisions[seg_idx].reason = "Improvised (match error)"
                improvised_count += 1
        else:
            # No script match = IMPROVISATION - always keep
            decisions[seg_idx].reason = f"Improvised content ({len(seg.text.split())} words)"
            improvised_count += 1
    
    # Now filter scripted parts - keep only LAST take for each script line
    total_cuts = 0
    
    for script_idx, match_list in script_matches.items():
        script_preview = script_sentences[script_idx][:50] + "..." if len(script_sentences[script_idx]) > 50 else script_sentences[script_idx]
        
        if len(match_list) == 1:
            # Only one take for this script line - perfect!
            seg_idx, score, seg = match_list[0]
            decisions[seg_idx].reason = f"Script line {script_idx+1} ({score:.0%})"
            logger.debug(f"Script line {script_idx+1}: Single take at segment {seg_idx}")
        else:
            # MULTIPLE TAKES for the same script line!
            # Strategy: Keep the LAST take (usually the correct one after retries)
            
            # Sort by segment index (temporal order)
            match_list.sort(key=lambda x: x[0])
            
            # The LAST one is the keeper
            best_seg_idx, best_score, best_seg = match_list[-1]
            decisions[best_seg_idx].reason = f"Best take - Script line {script_idx+1} ({best_score:.0%})"
            
            # Mark all PREVIOUS attempts as CUT
            logger.info(f"Script line {script_idx+1}: {len(match_list)} takes found, keeping LAST (segment {best_seg_idx})")
            logger.info(f"  Script: \"{script_preview}\"")
            
            for seg_idx, score, seg in match_list[:-1]:
                decisions[seg_idx].keep = False
                decisions[seg_idx].reason = f"Bad take - Script line {script_idx+1} (retake found later)"
                total_cuts += 1
                text_preview = seg.text[:60] + "..." if len(seg.text) > 60 else seg.text
                logger.info(f"  CUT segment {seg_idx}: \"{text_preview}\"")
    
    # Summary
    script_lines_covered = len(script_matches)
    logger.info(f"Script-guided analysis complete:")
    logger.info(f"  - {script_lines_covered} script lines matched")
    logger.info(f"  - {improvised_count} improvised segments (ALL KEPT)")
    logger.info(f"  - {total_cuts} bad takes CUT (earlier attempts of scripted lines)")
    
    return decisions


def analyze_retakes_fuzzy(
    segments: List[TranscriptSegment],
    threshold: float = 0.75,
) -> List[EditDecision]:
    """Detect retakes using fuzzy text matching and keep only the best take.
    
    CONSERVATIVE threshold (0.75) - prefer keeping content over cutting.
    Unique mentions (FTX, specific numbers, etc.) are protected.
    Returns an EditDecision for every segment so downstream logic can
    zero out entire clusters of repeated takes.
    """
    logger.info(f"Analyzing retakes with threshold={threshold}")
    
    if not segments:
        return []
    
    decisions = [
        EditDecision(start=s.start, end=s.end, keep=True, reason="Default")
        for s in segments
    ]
    
    i = 0
    groups_marked = 0
    filler_removed = 0
    
    while i < len(segments) - 1:
        group_indices = [i]
        j = i + 1
        while j < len(segments) and is_retake_pair(segments[j - 1], segments[j]):
            group_indices.append(j)
            j += 1
        
        if len(group_indices) > 1:
            groups_marked += 1
            group_size = len(group_indices)
            best_idx = max(
                group_indices, 
                key=lambda idx: segment_quality(
                    segments[idx], 
                    position_in_group=group_indices.index(idx),
                    group_size=group_size
                )
            )
            
            # DETAILED LOGGING: Show what's being marked as retake cluster
            logger.info(f"=== RETAKE CLUSTER {groups_marked} (segments {group_indices}) ===")
            for pos, idx in enumerate(group_indices):
                seg = segments[idx]
                quality = segment_quality(seg, position_in_group=pos, group_size=group_size)
                is_best = "★ KEEP" if idx == best_idx else "✗ CUT"
                text_preview = f"\"{seg.text[:80]}...\"" if len(seg.text) > 80 else f"\"{seg.text}\""
                logger.info(f"  [{is_best}] Seg {idx} ({seg.start:.1f}s-{seg.end:.1f}s) Q={quality:.1f}: {text_preview}")
            
            for idx in group_indices:
                if idx == best_idx:
                    decisions[idx].reason = "Best take in retake cluster"
                    continue
                decisions[idx].keep = False
                decisions[idx].reason = "Retake trimmed (inferior to best take in cluster)"
        
        i = group_indices[-1] + 1
    
    # Second pass: Remove filler-heavy segments (>25% filler words)
    for i, seg in enumerate(segments):
        if decisions[i].keep:  # Only check segments not already marked for removal
            filler_ratio = calculate_filler_ratio(seg.text)
            if filler_ratio > 0.25:
                decisions[i].keep = False
                decisions[i].reason = f"Filler-heavy ({filler_ratio:.0%} filler words)"
                filler_removed += 1
    
    # NOTE: We do NOT cut segments just because they contain stutters.
    # Stuttered words should be cleaned at word-level, not by removing whole segments.
    # The word-level trimming in trim_segment_retakes handles this.
    
    logger.info(f"Detected {groups_marked} retake clusters, {filler_removed} filler-heavy segments")
    return decisions


def is_false_start(a: TranscriptSegment, b: TranscriptSegment) -> bool:
    """Detect if segment A is a false start that B corrects/continues.
    
    Patterns:
    - A ends with incomplete phrase, B starts with same words
    - A: "they were" B: "they were building" -> A is false start
    - A: "that pr" B: "that promises" -> A is false start (cut-off word)
    - A: "I've been weeks" B: "I've spent weeks" -> A is false start (grammar fix)
    
    IMPORTANT: Don't mark substantial content (>15 words) as false starts.
    """
    a_words = a.text.lower().split()
    b_words = b.text.lower().split()
    
    if len(a_words) < 2 or len(b_words) < 3:
        return False
    
    # SAFEGUARD: Never mark a segment with >15 words as a false start
    # Substantial content should not be removed as a "false start"
    if len(a_words) > 15:
        return False
    
    # Check if A's last 2-4 words match B's first 2-4 words (restart pattern)
    for match_len in [4, 3, 2]:
        if len(a_words) >= match_len and len(b_words) >= match_len:
            a_end = a_words[-match_len:]
            b_start = b_words[:match_len]
            if a_end == b_start:
                return True
    
    # Check if A ends with incomplete word that B completes
    # e.g., A ends with "pr" and B starts with "promises"
    if len(a_words[-1]) <= 4:  # Short last word might be cut off
        last_word = a_words[-1].rstrip(',-.')
        if len(b_words) > 0 and b_words[0].startswith(last_word) and len(b_words[0]) > len(last_word):
            return True
    
    # Check if A is much shorter and B starts similarly (incomplete thought restated)
    if len(a_words) < len(b_words) * 0.6:  # A is less than 60% of B's length
        # Check if first 2 words match
        if len(a_words) >= 2 and len(b_words) >= 2:
            if a_words[:2] == b_words[:2]:
                return True
    
    return False


# NOTE: PROTECTED_ENTITIES removed - no longer needed since fuzzy retake cutting is disabled.
# All content is kept; only word-level stutters are removed by trim_segment_retakes().


def is_retake_pair(a: TranscriptSegment, b: TranscriptSegment, threshold: float = 0.75) -> bool:
    """Detect if two segments are retakes of each other.
    
    NOTE: This function is no longer used since fuzzy retake cutting is disabled.
    Kept for reference but analyze_retakes_fuzzy returns all-keep immediately.
    """
    if len(a.text) < 10 or len(b.text) < 10:
        return False
    
    # Check for false start pattern first
    if is_false_start(a, b):
        return True
    
    similarity = difflib.SequenceMatcher(
        None,
        a.text.lower(),
        b.text.lower(),
    ).ratio()
    has_phrase = any(
        phrase in a.text.lower() or phrase in b.text.lower()
        for phrase in RETAKE_PHRASES
    )
    
    # SAFEGUARD: Require higher similarity (90%) for longer segments
    # Short segments (< 30 chars) can use the configured threshold
    # Longer segments need to be MORE similar to be considered retakes
    adjusted_threshold = threshold
    if len(a.text) > 30 and len(b.text) > 30:
        adjusted_threshold = max(threshold, 0.90)  # At least 90% for substantial text
    
    return similarity > adjusted_threshold or has_phrase


def segment_quality(seg: TranscriptSegment, position_in_group: int = 0, group_size: int = 1) -> float:
    """Calculate quality score for a segment to pick the best take.
    
    Factors (in order of importance):
    1. Completeness: Does it end with proper punctuation? Full thought?
    2. Fluency: Fewer fillers, stutters, false starts
    3. Length: More words = more complete (but not always better)
    4. Pacing: Natural speaking rate (not too fast, not too slow)
    5. Position: Later takes often improve (you correct yourself)
    
    CrisperWhisper markers: [UH], [UM] count as fillers
    """
    text = seg.text.strip()
    words = text.split()
    num_words = len(words)
    duration = max(seg.duration, 0.5)
    
    # Base score from word count (more complete = better)
    base_score = num_words * 2.0
    
    # COMPLETENESS BONUS: Ends with punctuation = complete thought
    if text and text[-1] in '.!?':
        base_score += 15.0
    elif text and text[-1] in ',;:':
        base_score += 5.0  # Partial credit
    
    # FLUENCY PENALTY: Fillers and stutters hurt quality
    filler_ratio = calculate_filler_ratio(text)
    fluency_penalty = filler_ratio * 30.0  # Heavy penalty for fillers
    
    # CrisperWhisper filler markers
    crisper_fillers = text.count('[UH]') + text.count('[UM]')
    fluency_penalty += crisper_fillers * 5.0
    
    # INTERNAL STUTTER PENALTY: Repeated phrases within segment
    if has_internal_stutter(text):
        fluency_penalty += 20.0
    
    # PACING SCORE: Natural speaking rate is ~2.5-3.5 words/sec
    words_per_sec = num_words / duration
    if 2.0 <= words_per_sec <= 4.0:
        pacing_score = 10.0  # Optimal pace
    elif 1.5 <= words_per_sec <= 5.0:
        pacing_score = 5.0   # Acceptable
    else:
        pacing_score = 0.0   # Too slow or too rushed
    
    # POSITION BONUS: Later takes in a retake cluster often improve
    # (you correct yourself, speak more clearly the 2nd/3rd time)
    position_bonus = 0.0
    if group_size > 1:
        # Give slight bonus to later positions (but not too much)
        position_bonus = (position_in_group / group_size) * 8.0
    
    # INCOMPLETE SENTENCE PENALTY
    if is_incomplete_sentence(text):
        base_score -= 15.0
    
    final_score = base_score - fluency_penalty + pacing_score + position_bonus
    return max(0.0, final_score)  # Never negative


def has_phrase_stutter(text: str) -> bool:
    """Detect if text contains repeated phrases indicating a stutter or self-correction.
    
    STRICT detection for TRUE stutters only:
    - "their, their" (word repeated immediately with comma)
    - "the the" (word repeated immediately)
    - "th-the" (hyphenated stutter)
    - "we wouldn't, we would" (restart pattern)
    - "or innovative or innovative" (phrase repeated immediately)
    
    Does NOT match common phrase reuse like:
    - "not your keys, not your coins" (different sentences)
    - "it's the... it's the" (far apart in text)
    
    Returns True if any IMMEDIATE repeated pattern is found.
    """
    import re
    text_lower = text.lower()
    
    # Pattern 1: Hyphenated repeats (th-the, wh-what, I-I)
    if re.search(r'\b(\w{1,4})-\1\w*\b', text_lower):
        logger.debug(f"Hyphenated stutter found in: {text_lower[:60]}")
        return True
    
    # Pattern 2: IMMEDIATE single word repeats with comma/space
    # "the, the" or "the the" or "their, their" - MUST be directly adjacent
    match = re.search(r'\b(\w+),?\s+\1\b', text_lower)
    if match:
        # Avoid false positives for common patterns like "is is" in "this is is the"
        word = match.group(1)
        # Only flag if it's a meaningful word (not just "a", "I" alone)
        if len(word) > 1 or word in ['i']:
            logger.debug(f"Word repeat stutter found: '{word}' in: {text_lower[:60]}")
            return True
    
    # Pattern 3: 2-3 word phrase IMMEDIATELY repeated (no gap allowed)
    # "a billion a billion" or "or innovative or innovative"
    if re.search(r'\b(\w+\s+\w+),?\s+\1\b', text_lower):
        logger.debug(f"2-word phrase repeat found in: {text_lower[:60]}")
        return True
    
    # Pattern 4: Self-correction pattern "we wouldn't, we would"
    # Word + something, same word + different continuation
    if re.search(r"\b(we|i|you|they|it)\s+\w+[',]?\s+\1\s+\w+\b", text_lower):
        # Check if it's actually a restart (words after are different or similar)
        pass  # This is too broad, skip it
    
    return False
    return False


def calculate_filler_ratio(text: str) -> float:
    """Calculate the ratio of filler words and stutters in text."""
    import re
    text_lower = text.lower()
    words = text_lower.split()
    if not words:
        return 0.0
    
    # 1. Count explicit Crisper markers (e.g. [UH], [UM])
    crisper_markers = {"[uh]", "[um]", "[ah]", "[erm]", "[mm]"}
    crisper_count = sum(1 for w in words if w in crisper_markers)
    
    # 2. Count standard fillers (avoiding double count with markers)
    # Remove crisper markers from text for the standard filler check
    clean_text = text_lower
    for marker in crisper_markers:
        clean_text = clean_text.replace(marker, "")
    
    standard_filler_count = 0
    # Sort fillers by length (descending) to match phrases like "you know" before "you"
    sorted_fillers = sorted(FILLER_WORDS, key=len, reverse=True)
    
    for filler in sorted_fillers:
        # Use word boundaries to avoid matching "so" in "some" or "er" in "better"
        # Escape filler to handle special chars if any
        pattern = r'\b' + re.escape(filler) + r'\b'
        matches = len(re.findall(pattern, clean_text))
        standard_filler_count += matches

    # 3. Count stutters
    # Use regex for repeated words and stutter patterns
    stutter_count = 0
    for pattern in STUTTER_PATTERNS:
        stutter_count += len(re.findall(pattern, text_lower))
    
    total_issues = crisper_count + standard_filler_count + stutter_count
    
    # Cap at 1.0 (100%) to avoid confusing >100% scores
    return min(1.0, total_issues / max(len(words), 1))

# =============================================================================
# MODE 3: DYNAMIC ZOOM (Face Tracking)
# =============================================================================

def apply_dynamic_zoom(
    video_path: Path,
    output_path: Path,
    config: EditorConfig,
) -> Path:
    """Apply face-tracking dynamic zoom to video using OpenCV + ffmpeg.
    
    Uses OpenCV for face detection and applies smooth zoom/pan
    to keep the face centered with a Ken Burns-style effect.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        config: Editor configuration
        
    Returns:
        Path to zoomed video
    """
    logger.info("Applying dynamic face-tracking zoom...")
    
    # Load face detection model
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Open video with OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Parameters
    zoom_factor = 1.3  # How much to zoom in
    smoothing_frames = 15  # Frames for running average
    
    # Track face positions
    face_positions: List[Optional[Tuple[int, int, int, int]]] = []
    
    logger.info("Pass 1: Detecting faces...")
    
    # First pass: detect faces in all frames
    for _ in tqdm(range(total_frames), desc="Face detection"):
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        
        if len(faces) > 0:
            # Take largest face
            largest = max(faces, key=lambda f: f[2] * f[3])
            face_positions.append(tuple(largest))
        else:
            face_positions.append(None)
    
    cap.release()
    
    # Interpolate missing face positions
    face_positions = interpolate_face_positions(face_positions)
    
    # Smooth face positions with running average
    smoothed_positions = smooth_positions(face_positions, smoothing_frames)
    
    logger.info("Pass 2: Applying zoom transformation...")
    
    # Reopen video for second pass
    cap = cv2.VideoCapture(str(video_path))
    
    # Create temporary video output (without audio)
    temp_video = config.temp_dir / "zoom_temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_video), fourcc, fps, (w, h))
    
    for idx in tqdm(range(total_frames), desc="Applying zoom"):
        ret, frame = cap.read()
        if not ret:
            break
        
        pos = smoothed_positions[idx] if idx < len(smoothed_positions) else None
        
        if pos is not None:
            fx, fy, fw, fh = pos
            
            # Calculate crop region centered on face
            face_center_x = fx + fw // 2
            face_center_y = fy + fh // 2
            
            # Crop dimensions (zoomed in)
            crop_w = int(w / zoom_factor)
            crop_h = int(h / zoom_factor)
            
            # Center crop on face with padding
            crop_x = max(0, min(w - crop_w, face_center_x - crop_w // 2))
            crop_y = max(0, min(h - crop_h, face_center_y - crop_h // 2))
            
            # Crop and resize back to original dimensions
            cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Merge zoomed video with original audio using ffmpeg
    codec = "h264_nvenc" if config.device == "cuda" else "libx264"
    
    logger.info(f"Merging with audio, codec: {codec}")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(temp_video),
        "-i", str(video_path),
        "-c:v", codec, "-preset", "fast",
        "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0?",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    
    # Cleanup
    temp_video.unlink(missing_ok=True)
    
    logger.info(f"Dynamic zoom complete: {output_path}")
    return output_path


def interpolate_face_positions(
    positions: List[Optional[Tuple[int, int, int, int]]],
) -> List[Optional[Tuple[int, int, int, int]]]:
    """Interpolate missing face positions between detected frames."""
    result = positions.copy()
    
    # Find first and last valid positions
    first_valid = next((i for i, p in enumerate(positions) if p), None)
    last_valid = next((i for i, p in enumerate(reversed(positions)) if p), None)
    
    if first_valid is None:
        return result  # No faces detected
    
    last_valid = len(positions) - 1 - last_valid
    
    # Fill gaps with linear interpolation
    prev_valid_idx = first_valid
    for i in range(first_valid, last_valid + 1):
        if positions[i] is not None:
            # Interpolate between prev_valid_idx and i
            if i > prev_valid_idx + 1:
                start_pos = positions[prev_valid_idx]
                end_pos = positions[i]
                for j in range(prev_valid_idx + 1, i):
                    t = (j - prev_valid_idx) / (i - prev_valid_idx)
                    result[j] = tuple(
                        int(start_pos[k] + t * (end_pos[k] - start_pos[k]))
                        for k in range(4)
                    )
            prev_valid_idx = i
    
    return result


def smooth_positions(
    positions: List[Optional[Tuple[int, int, int, int]]],
    window: int,
) -> List[Optional[Tuple[int, int, int, int]]]:
    """Apply running average smoothing to face positions."""
    result = []
    
    for i in range(len(positions)):
        start = max(0, i - window // 2)
        end = min(len(positions), i + window // 2 + 1)
        
        valid = [p for p in positions[start:end] if p is not None]
        
        if valid:
            avg = tuple(
                int(np.mean([p[k] for p in valid]))
                for k in range(4)
            )
            result.append(avg)
        else:
            result.append(None)
    
    return result


# =============================================================================
# MODE 4: B-ROLL INSERTION (LLM + Pexels)
# =============================================================================

def analyze_with_llm(
    segments: List[TranscriptSegment],
    config: EditorConfig,
) -> Tuple[List[EditDecision], Dict[float, List[str]]]:
    """Analyze segments with local LLM for edit decisions and B-roll suggestions.
    
    Args:
        segments: Transcript segments
        config: Editor configuration
        
    Returns:
        Tuple of (edit_decisions, broll_requests)
        broll_requests maps segment start time to list of keywords
    """
    logger.info(f"Analyzing {len(segments)} segments with Gemini ({config.gemini_model})...")
    
    decisions = []
    broll_requests = {}
    
    # Chunk segments into ~90 second groups to respect 50 RPD limit (40min video -> ~27 requests)
    chunked_segments = chunk_segments(segments, target_duration=90.0)
    
    for i, chunk in enumerate(tqdm(chunked_segments, desc="LLM Analysis")):
        # Rate limit delay: 15 RPM = 4s minimum, use 5s for safety margin
        if i > 0:
            time.sleep(5)

        # Format segments for prompt
        segment_lines = []
        for idx, seg in enumerate(chunk, 1):
            segment_lines.append(f"{idx}. [{seg.start:.1f}-{seg.end:.1f}] {seg.text}")
        
        segments_text = "\n".join(segment_lines)
        if len(segments_text.strip()) < 10:
            # Nothing meaningful in this chunk
            continue
        
        # Query Gemini
        try:
            response = query_gemini(
                LLM_PROMPT_TEMPLATE.format(segments_text=segments_text),
                config,
            )
            
            # Parse JSON response
            result = parse_llm_response(response)
            remove_indices = set(result.get("remove_indices", []))
            broll_map = result.get("broll_suggestions", {})
            
            # Apply decisions to segments in this chunk
            for idx, seg in enumerate(chunk, 1):
                keep = idx not in remove_indices
                
                # Get B-roll keywords if available (keys are strings in JSON)
                keywords = broll_map.get(str(idx))
                
                decisions.append(EditDecision(
                    start=seg.start,
                    end=seg.end,
                    keep=keep,
                    reason="LLM decision",
                    broll_keywords=keywords,
                ))
                
                if keep and keywords:
                    broll_requests[seg.start] = keywords
                
        except Exception as e:
            logger.warning(f"LLM analysis failed for chunk starting at {chunk[0].start:.1f}s: {e}")
            # Default to keeping on error
            for seg in chunk:
                decisions.append(EditDecision(
                    start=seg.start,
                    end=seg.end,
                    keep=True,
                    reason=f"LLM error (defaulted to keep): {e}",
                ))
    
    kept = sum(1 for d in decisions if d.keep)
    logger.info(f"LLM decisions: {kept}/{len(decisions)} segments kept")
    logger.info(f"B-roll requests: {len(broll_requests)} segments")
    
    return decisions, broll_requests


def chunk_segments(
    segments: List[TranscriptSegment],
    target_duration: float = 10.0,
) -> List[List[TranscriptSegment]]:
    """Chunk segments into groups of approximately target_duration seconds."""
    chunks = []
    current_chunk = []
    current_duration = 0.0
    
    for seg in segments:
        current_chunk.append(seg)
        current_duration += seg.duration
        
        if current_duration >= target_duration:
            chunks.append(current_chunk)
            current_chunk = []
            current_duration = 0.0
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def query_gemini(prompt: str, config: EditorConfig, retry_count: int = 0) -> str:
    """Query Gemini API for LLM response with exponential backoff.
    
    Args:
        prompt: The prompt to send
        config: Editor configuration with API key and model
        retry_count: Current retry attempt (for exponential backoff)
        
    Returns:
        Generated response text
    """
    MAX_RETRIES = 5
    
    api_key = config.gemini_api_key or GEMINI_API_KEY
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to .env file or pass via --gemini-api-key"
        )
    
    url = GEMINI_API_URL.format(model=config.gemini_model)
    
    try:
        response = requests.post(
            url,
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1500,
                },
            },
            timeout=60,
        )
        
        if response.status_code == 429:
            # Rate limit hit - exponential backoff
            if retry_count >= MAX_RETRIES:
                logger.error(f"Gemini rate limit: max retries ({MAX_RETRIES}) exceeded")
                return ""  # Return empty instead of crashing
            wait_time = min(60, 10 * (2 ** retry_count))  # 10s, 20s, 40s, 60s, 60s
            logger.warning(f"Gemini rate limit hit (429). Waiting {wait_time}s (retry {retry_count + 1}/{MAX_RETRIES})...")
            time.sleep(wait_time)
            return query_gemini(prompt, config, retry_count + 1)
            
        response.raise_for_status()
        
        data = response.json()
        candidates = data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return parts[0].get("text", "")
        return ""
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            if retry_count >= MAX_RETRIES:
                logger.error(f"Gemini rate limit: max retries ({MAX_RETRIES}) exceeded")
                return ""
            wait_time = min(60, 10 * (2 ** retry_count))
            logger.warning(f"Gemini rate limit hit (429). Waiting {wait_time}s (retry {retry_count + 1}/{MAX_RETRIES})...")
            time.sleep(wait_time)
            return query_gemini(prompt, config, retry_count + 1)
        raise RuntimeError(f"Gemini API error: {e}")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Gemini API. Check internet connection.")


def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, handling common formatting issues."""
    # Try to extract JSON from response (greedy match for outermost braces)
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to parse entire response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Default response if parsing fails
        return {"remove_indices": [], "broll_suggestions": {}}


def download_broll(
    keywords: List[str],
    config: EditorConfig,
) -> Optional[Path]:
    """Download B-roll from Pexels API or use local folder.
    
    Args:
        keywords: List of search keywords
        config: Editor configuration
        
    Returns:
        Path to downloaded video or None if unavailable
    """
    # Try local folder first
    if config.local_broll_dir.exists():
        local_match = find_local_broll(keywords, config.local_broll_dir)
        if local_match:
            logger.debug(f"Using local B-roll: {local_match}")
            return local_match
    
    # Try Pexels API first
    if config.pexels_api_key:
        try:
            result = download_from_pexels(keywords, config)
            if result:
                return result
        except Exception as e:
            logger.warning(f"Pexels download failed: {e}")
    
    # Fall back to Pixabay API
    if config.pixabay_api_key:
        try:
            result = download_from_pixabay(keywords, config)
            if result:
                return result
        except Exception as e:
            logger.warning(f"Pixabay download failed: {e}")
    
    return None


def find_local_broll(keywords: List[str], folder: Path) -> Optional[Path]:
    """Search local folder for matching B-roll."""
    video_extensions = {".mp4", ".mov", ".avi", ".webm"}
    
    for keyword in keywords:
        keyword_lower = keyword.lower().replace(" ", "_")
        
        for file in folder.iterdir():
            if file.suffix.lower() in video_extensions:
                if keyword_lower in file.stem.lower():
                    return file
    
    return None


def download_from_pexels(
    keywords: List[str],
    config: EditorConfig,
) -> Optional[Path]:
    """Download video from Pexels API."""
    headers = {"Authorization": config.pexels_api_key}
    
    for keyword in keywords:
        try:
            response = requests.get(
                PEXELS_API_URL,
                headers=headers,
                params={"query": keyword, "per_page": 1, "size": "medium"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("videos"):
                video = data["videos"][0]
                video_files = video.get("video_files", [])
                
                # Find medium quality file
                video_file = next(
                    (f for f in video_files if f.get("quality") == "sd"),
                    video_files[0] if video_files else None,
                )
                
                if video_file:
                    video_url = video_file["link"]
                    output_path = config.temp_dir / f"broll_{keyword.replace(' ', '_')}.mp4"
                    
                    # Download video (streamed)
                    with requests.get(video_url, stream=True, timeout=60) as r:
                        r.raise_for_status()
                        with open(output_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    
                    logger.info(f"Downloaded B-roll: {keyword}")
                    return output_path
                    
        except Exception as e:
            logger.debug(f"Pexels search failed for '{keyword}': {e}")
            continue
    
    return None


def download_from_pixabay(
    keywords: List[str],
    config: EditorConfig,
) -> Optional[Path]:
    """Download video from Pixabay API."""
    for keyword in keywords:
        try:
            response = requests.get(
                PIXABAY_API_URL,
                params={
                    "key": config.pixabay_api_key,
                    "q": keyword,
                    "per_page": 3,
                    "video_type": "film",
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("hits"):
                video = data["hits"][0]
                videos = video.get("videos", {})
                
                # Prefer medium, then small quality
                video_url = None
                for quality in ["medium", "small", "tiny"]:
                    if quality in videos and videos[quality].get("url"):
                        video_url = videos[quality]["url"]
                        break
                
                if video_url:
                    output_path = config.temp_dir / f"broll_{keyword.replace(' ', '_')}.mp4"
                    
                    with requests.get(video_url, stream=True, timeout=60) as r:
                        r.raise_for_status()
                        with open(output_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    
                    logger.info(f"Downloaded B-roll from Pixabay: {keyword}")
                    return output_path
                    
        except Exception as e:
            logger.debug(f"Pixabay search failed for '{keyword}': {e}")
            continue
    
    return None


# =============================================================================
# VIDEO COMPOSITING
# =============================================================================

def apply_cuts(
    video_path: Path,
    decisions: List[EditDecision],
    config: EditorConfig,
) -> Path:
    """Apply edit decisions to cut video using ffmpeg complex filter.
    
    OPTIMIZED: Uses single-pass encoding with trim+concat filter instead of
    extracting each segment separately. This is ~10x faster for videos with
    many segments (100+ segments processed in one ffmpeg call).
    
    Args:
        video_path: Input video path
        decisions: List of edit decisions
        config: Editor configuration
        
    Returns:
        Path to cut video
    """
    logger.info("Applying cuts to video...")
    
    # Get segments to keep
    raw_segments = [d for d in decisions if d.keep]
    
    if not raw_segments:
        logger.warning("No segments to keep! Returning original.")
        return video_path
    
    # Sort by start time
    raw_segments.sort(key=lambda x: x.start)
    
    # =========================================================
    # PADDING & MERGING - Prevents choppy "micro-cuts"
    # Adds breathing room and merges segments that are close together
    # =========================================================
    PADDING = 0.15  # 150ms padding (sweet spot: 100-250ms)
    
    keep_segments = []
    if raw_segments:
        # Start first segment with padding
        current_start = max(0, raw_segments[0].start - PADDING)
        current_end = raw_segments[0].end + PADDING
        current_reason = raw_segments[0].reason
        
        for next_seg in raw_segments[1:]:
            # Calculate next segment's padded range
            next_start = max(0, next_seg.start - PADDING)
            next_end = next_seg.end + PADDING
            
            # If padding causes overlap, MERGE them (prevents glitchy micro-cuts)
            if next_start <= current_end:
                current_end = max(current_end, next_end)
            else:
                # No overlap - finalize previous segment
                keep_segments.append(EditDecision(
                    start=current_start,
                    end=current_end,
                    keep=True,
                    reason=current_reason
                ))
                current_start = next_start
                current_end = next_end
                current_reason = next_seg.reason
        
        # Append final segment
        keep_segments.append(EditDecision(
            start=current_start,
            end=current_end,
            keep=True,
            reason=current_reason
        ))
    
    logger.info(f"Merged {len(raw_segments)} segments → {len(keep_segments)} smooth segments (150ms padding)")
    
    # Get video duration using ffprobe
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    try:
        video_duration = float(subprocess.check_output(duration_cmd, text=True).strip())
    except Exception:
        video_duration = float('inf')
    
    output_path = config.temp_dir / "cut_video.mp4"
    codec = "h264_nvenc" if config.device == "cuda" else "libx264"
    
    # For many segments, use complex filter (single-pass, much faster)
    # For very few segments (<5), the overhead isn't worth it
    if len(keep_segments) >= 5:
        return _apply_cuts_complex_filter(video_path, keep_segments, video_duration, output_path, codec, config)
    else:
        return _apply_cuts_simple(video_path, keep_segments, video_duration, output_path, codec, config)


def _apply_cuts_complex_filter(
    video_path: Path,
    keep_segments: List[EditDecision],
    video_duration: float,
    output_path: Path,
    codec: str,
    config: EditorConfig,
) -> Path:
    """Apply cuts using ffmpeg complex filter - FAST single-pass encoding.
    
    IMPORTANT: Uses careful A/V sync handling to prevent lipsync drift:
    - Each audio segment uses asetpts with proper timestamp reset
    - Async resampling applied to final audio to catch any remaining drift
    """
    logger.info(f"Using complex filter for {len(keep_segments)} segments (single-pass with crossfades)...")
    
    # Crossfade duration for smooth audio transitions (eliminates pops)
    # 50ms is industry standard for dialogue - short enough to not lose words,
    # long enough to hide the sudden change in background noise "texture"
    CROSSFADE_MS = 50  # 50ms crossfade - blends breath/room-tone between cuts
    CROSSFADE_SEC = CROSSFADE_MS / 1000.0
    
    # Build filter components
    video_trims = []
    audio_trims = []
    
    for i, seg in enumerate(keep_segments):
        start = max(0, seg.start)
        end = min(video_duration, seg.end)
        duration = end - start
        
        # GUARD: Skip segments shorter than 2x crossfade (would cause FFmpeg error)
        if duration < CROSSFADE_SEC * 3:
            logger.warning(f"Skipping segment {i} (too short: {duration:.3f}s < {CROSSFADE_SEC*3:.3f}s min)")
            continue
        
        # Video trim with 5ms fade in/out to smooth visual cuts
        fade_dur = 0.005
        video_trims.append(f"[0:v]trim={start}:{end},setpts=PTS-STARTPTS,fade=t=in:st=0:d={fade_dur},fade=t=out:st={duration-fade_dur}:d={fade_dur}[v{i}]")
        # Audio trim with micro-fade to prevent pops  
        audio_trims.append(f"[0:a]atrim={start}:{end},asetpts=PTS-STARTPTS,afade=t=in:st=0:d={CROSSFADE_SEC},afade=t=out:st={duration-CROSSFADE_SEC}:d={CROSSFADE_SEC}[a{i}]")
    
    # Build concat inputs - only include valid segments
    valid_indices = [i for i, seg in enumerate(keep_segments) 
                     if (min(video_duration, seg.end) - max(0, seg.start)) >= CROSSFADE_SEC * 3]
    concat_inputs = [f"[v{i}][a{i}]" for i in valid_indices]
    
    if not concat_inputs:
        logger.error("No valid segments after filtering!")
        return video_path
    
    # Build the full filter
    n_segments = len(concat_inputs)
    filter_complex = ";".join(video_trims + audio_trims)
    # Add async resampling AFTER concat to fix any accumulated A/V drift
    filter_complex += f";{''.join(concat_inputs)}concat=n={n_segments}:v=1:a=1[outv][outa_raw]"
    filter_complex += ";[outa_raw]aresample=async=1000:first_pts=0[outa]"
    
    # Single ffmpeg command to do everything
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", codec, "-preset", "p4" if "nvenc" in codec else "fast", "-cq", "19",
        "-c:a", "aac", "-ar", "48000",
        "-fps_mode", "cfr",
        "-avoid_negative_ts", "make_zero",
        str(output_path)
    ]
    
    logger.info(f"Running single-pass encoding...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        # Fallback to simple method if complex filter fails
        logger.info("Complex filter failed, falling back to segment-by-segment...")
        return _apply_cuts_simple(video_path, keep_segments, video_duration, output_path, codec, config)
    
    logger.info(f"Cut video saved: {output_path}")
    return output_path


def _apply_cuts_simple(
    video_path: Path,
    keep_segments: List[EditDecision],
    video_duration: float,
    output_path: Path,
    codec: str,
    config: EditorConfig,
) -> Path:
    """Apply cuts using segment extraction + concat (slower but more reliable)."""
    logger.info(f"Using segment extraction for {len(keep_segments)} segments...")
    
    concat_file = config.temp_dir / "concat_list.txt"
    segment_files = []
    
    for i, seg in enumerate(tqdm(keep_segments, desc="Extracting segments")):
        start = max(0, seg.start)
        end = min(video_duration, seg.end)
        duration = end - start
        
        segment_path = config.temp_dir / f"segment_{i:04d}.mp4"
        segment_files.append(segment_path)
        
        # Use hybrid seeking for speed + accuracy
        rough_start = max(0, start - 5.0)
        fine_start = start - rough_start
        
        cmd = [
            "ffmpeg", "-y", 
            "-ss", str(rough_start),
            "-i", str(video_path),
            "-ss", str(fine_start),
            "-t", str(duration), 
            "-c:v", codec, "-preset", "fast", "-cq", "19",
            "-c:a", "aac", "-ar", "48000",
            "-fps_mode", "cfr",
            "-af", "aresample=async=1:first_pts=0",
            "-avoid_negative_ts", "make_zero",
            str(segment_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
    
    # Write concat file
    with open(concat_file, "w") as f:
        for seg_path in segment_files:
            f.write(f"file '{seg_path.as_posix()}'\n")
    
    # Concatenate with copy (segments already encoded)
    concat_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy",  # Just copy, don't re-encode!
        "-avoid_negative_ts", "make_zero",
        str(output_path)
    ]
    
    subprocess.run(concat_cmd, capture_output=True, check=True)
    
    # Cleanup
    for seg_path in segment_files:
        seg_path.unlink(missing_ok=True)
    concat_file.unlink(missing_ok=True)
    
    logger.info(f"Cut video saved: {output_path}")
    return output_path


def composite_broll(
    base_video: Path,
    broll_requests: Dict[float, List[str]],
    config: EditorConfig,
) -> Path:
    """Composite B-roll overlays onto base video using ffmpeg.
    
    Args:
        base_video: Base video path
        broll_requests: Dict mapping timestamps to B-roll keywords
        config: Editor configuration
        
    Returns:
        Path to composited video
    """
    logger.info(f"Compositing {len(broll_requests)} B-roll segments...")
    
    if not broll_requests:
        return base_video
    
    # Get base video dimensions
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", str(base_video)
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    base_w, base_h = map(int, result.stdout.strip().split(","))
    
    # Full screen B-roll (scale to fit base video)
    # We use force_original_aspect_ratio=increase to fill screen, then crop
    
    # Process each B-roll overlay
    overlay_inputs = []
    filter_parts = []
    
    sorted_requests = sorted(broll_requests.items(), key=lambda x: x[0])
    
    for i, (timestamp, keywords) in enumerate(tqdm(sorted_requests, desc="Preparing B-roll")):
        if i % 5 == 0:
            logger.info(f"Preparing B-roll {i+1}/{len(sorted_requests)}")
        # Download/find B-roll
        broll_path = download_broll(keywords, config)
        
        if broll_path is None:
            continue
        
        # Apply Ken Burns effect via ffmpeg and prepare overlay
        processed_broll = apply_ken_burns_ffmpeg(broll_path, config, duration=5.0)
        if processed_broll is None:
            continue
        
        overlay_inputs.append(processed_broll)
        input_idx = len(overlay_inputs)  # 1-indexed since base is 0
        
        # Scale B-roll to fill screen and overlay
        # scale={base_w}:{base_h}:force_original_aspect_ratio=increase,crop={base_w}:{base_h}
        filter_parts.append(
            f"[{input_idx}:v]scale={base_w}:{base_h}:force_original_aspect_ratio=increase,crop={base_w}:{base_h}[broll{i}];"
            f"[tmp{i}][broll{i}]overlay=0:0:enable='between(t,{timestamp},{timestamp+5})'[tmp{i+1}]"
        )
    
    if not overlay_inputs:
        logger.warning("No B-roll clips available, returning base video")
        return base_video
    
    # Build ffmpeg command
    output_path = config.temp_dir / "broll_composite.mp4"
    codec = "h264_nvenc" if config.device == "cuda" else "libx264"
    
    # Build input list
    inputs = ["-i", str(base_video)]
    for broll_path in overlay_inputs:
        inputs.extend(["-i", str(broll_path)])
    
    # Build filter chain - start with base video as tmp0 and copy final output label explicitly
    filter_chain = f"[0:v]copy[tmp0];" + ";".join(filter_parts)
    final_label = f"tmp{len(overlay_inputs)}"
    filter_chain += f";[{final_label}]copy[outv]"
    
    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_chain,
        "-map", "[outv]", "-map", "0:a",
        "-c:v", codec, "-c:a", "aac", "-preset", "fast",
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        logger.info(f"B-roll composite saved: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"B-roll compositing failed: {e.stderr.decode() if e.stderr else e}")
        return base_video


def apply_ken_burns_ffmpeg(
    input_path: Path,
    config: EditorConfig,
    duration: float = 5.0,
) -> Optional[Path]:
    """Apply Ken Burns (slow pan + zoom) effect using ffmpeg zoompan filter.
    
    Args:
        input_path: Input video path
        config: Editor configuration  
        duration: Target duration in seconds
        
    Returns:
        Path to processed clip or None on failure
    """
    output_path = config.temp_dir / f"kb_{input_path.stem}.mp4"
    
    # Random zoom: start at 1.0-1.1, end at 1.1-1.2
    start_zoom = random.uniform(1.0, 1.1)
    end_zoom = random.uniform(1.1, 1.2)
    
    # zoompan filter: z is zoom level, d is duration in frames (assuming 30fps)
    fps = 30
    frames = int(duration * fps)
    
    # Expression for zoom: linear interpolation from start to end
    zoom_expr = f"'{start_zoom}+on/{frames}*{end_zoom - start_zoom}'"
    
    codec = "h264_nvenc" if config.device == "cuda" else "libx264"
    
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"zoompan=z={zoom_expr}:d={frames}:fps={fps}:s=1280x720",
        "-t", str(duration),
        "-c:v", codec, "-preset", "fast", "-an",
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        logger.warning(f"Ken Burns effect failed: {e}")
        return None


# =============================================================================
# ADDITIONAL FEATURES
# =============================================================================

def generate_chapters(segments: List[TranscriptSegment]) -> str:
    """Generate YouTube chapter timestamps from transcript.
    
    Uses silence gaps and semantic shifts to detect chapter boundaries.
    
    Args:
        segments: Transcript segments
        
    Returns:
        YouTube chapter format string
    """
    logger.info("Generating YouTube chapters...")
    
    chapters = [{"time": 0.0, "title": "Intro"}]
    
    # Detect chapter breaks by long pauses (>5 seconds between segments)
    for i in range(1, len(segments)):
        prev_end = segments[i - 1].end
        curr_start = segments[i].start
        
        gap = curr_start - prev_end
        
        if gap >= 5.0:
            # Use first few words as chapter title
            title_words = segments[i].text.split()[:5]
            title = " ".join(title_words)
            if len(title) > 30:
                title = title[:27] + "..."
            
            chapters.append({
                "time": curr_start,
                "title": title.strip().capitalize(),
            })
    
    # Format as YouTube chapters
    lines = []
    for ch in chapters:
        minutes = int(ch["time"] // 60)
        seconds = int(ch["time"] % 60)
        lines.append(f"{minutes:02d}:{seconds:02d} {ch['title']}")
    
    return "\n".join(lines)


def generate_subtitles(
    segments: List[TranscriptSegment],
    output_path: Path,
) -> Path:
    """Generate SRT subtitle file from transcript.
    
    Args:
        segments: Transcript segments
        output_path: Output SRT path
        
    Returns:
        Path to SRT file
    """
    logger.info("Generating subtitles...")
    
    srt_lines = []
    
    for i, seg in enumerate(segments, 1):
        start_time = format_srt_time(seg.start)
        end_time = format_srt_time(seg.end)
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(seg.text)
        srt_lines.append("")
    
    srt_content = "\n".join(srt_lines)
    output_path.write_text(srt_content, encoding="utf-8")
    
    logger.info(f"Subtitles saved: {output_path}")
    return output_path


def format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_subtitle_filter_arg(srt_path: Path) -> str:
    """Return a subtitles filter arg with Windows-safe escaping."""
    posix_path = srt_path.resolve().as_posix()
    escaped = posix_path.replace(":", "\\:").replace("'", "\\'")
    return f"subtitles='{escaped}'"


def embed_subtitles(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
) -> Path:
    """Embed subtitles into video using FFmpeg.
    
    Args:
        video_path: Input video
        srt_path: SRT subtitle file
        output_path: Output video path
        
    Returns:
        Path to video with embedded subtitles
    """
    logger.info("Embedding subtitles...")
    
    subtitle_filter = build_subtitle_filter_arg(srt_path)
    
    # Use nvenc if available for faster encoding
    # Subtitle burning requires re-encoding
    codec = "h264_nvenc" if "cuda" in str(subprocess.check_output(["ffmpeg", "-encoders"], text=True)) else "libx264"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", subtitle_filter,
        "-c:v", codec, "-preset", "fast",
        "-c:a", "copy",
        str(output_path),
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        logger.info(f"Subtitles embedded: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.warning(f"Subtitle embedding failed: {e.stderr.decode() if e.stderr else e}")
        return video_path


def ensure_segments_for_optional_outputs(
    segments: Optional[List[TranscriptSegment]],
    video_source: Path,
    config: EditorConfig,
) -> List[TranscriptSegment]:
    """Transcribe video if chapters/subtitles requested but segments missing."""
    if segments is not None:
        return segments
    logger.info("Transcribing video for requested chapters/subtitles...")
    return transcribe_video(video_source, config)


def apply_optional_outputs(
    segments: Optional[List[TranscriptSegment]],
    current_video: Path,
    config: EditorConfig,
) -> Path:
    """Generate chapters/subtitles for non-auto modes when requested."""
    final_video = current_video
    if config.chapters:
        if segments:
            chapters = generate_chapters(segments)
            chapters_path = config.output_path.with_suffix(".chapters.txt")
            chapters_path.write_text(chapters, encoding="utf-8")
            print(f"\nYouTube Chapters:\n{chapters}")
        else:
            logger.warning("Chapters requested but transcript unavailable for this mode.")
    if config.subtitles:
        if segments:
            srt_path = config.temp_dir / "subtitles.srt"
            generate_subtitles(segments, srt_path)
            subtitled = config.temp_dir / "subtitled.mp4"
            final_video = embed_subtitles(final_video, srt_path, subtitled)
        else:
            logger.warning("Subtitles requested but transcript unavailable for this mode.")
    return final_video


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline(config: EditorConfig) -> Path:
    """Run the full auto editing pipeline.
    
    Steps:
    1. Transcribe video
    2. Analyze with LLM
    3. Apply cuts
    4. Composite B-roll
    5. (Optional) Generate chapters
    6. (Optional) Embed subtitles
    7. Export final video
    
    Args:
        config: Editor configuration
        
    Returns:
        Path to final output video
    """
    logger.info("=" * 60)
    logger.info("STARTING FULL AUTO PIPELINE")
    logger.info("=" * 60)
    
    # Step 0: Pre-cut Silence (Optional)
    current_input = config.input_path
    
    # Always run silence cutting in auto mode unless explicitly disabled?
    # For now, we use the flag, but we default to using our new ffmpeg cutter if auto-editor fails
    # or if we want to enforce silence cutting.
    # Let's make silence cutting standard in auto mode using our new ffmpeg function
    
    logger.info("\n[0/6] Pre-cutting Silence...")
    if config.pre_cut_silence:
        # Try auto-editor first if requested
        try:
            current_input = cut_silence_auto_editor(config)
        except Exception as e:
            logger.warning(f"Auto-editor failed: {e}")
            current_input = cut_silence_ffmpeg(current_input, config)
    else:
        # Use our internal ffmpeg cutter by default for "auto" mode to fix "silence gaps"
        current_input = cut_silence_ffmpeg(current_input, config)

    # Step 1: Transcribe
    logger.info("\n[1/6] Transcription...")
    segments = transcribe_video(current_input, config)
    
    # Step 1.1: Wav2Vec2 Acoustic Stutter Detection (optional second pass)
    # This compares literal Wav2Vec2 output to cleaned Whisper to find stutters
    wav2vec_stutters = []
    if config.use_wav2vec2:
        logger.info("\n[1.1/6] Wav2Vec2 Acoustic Stutter Detection (Whisper ↔ Wav2Vec2 Diff)...")
        try:
            processor, wav2vec_model = load_wav2vec2_model(config.device)
            if processor is not None and wav2vec_model is not None:
                audio_path = config.temp_dir / "audio.wav"
                if not audio_path.exists():
                    extract_audio(current_input, audio_path, enhance=config.enhance_audio, temp_dir=config.temp_dir)
                
                # Per-segment stutter detection with diff logic
                total_stutters = 0
                segments_with_stutters = 0
                for seg in segments:
                    seg_stutters, new_start = detect_acoustic_stutters_for_segment(
                        audio_path, 
                        seg.start, 
                        seg.end, 
                        seg.text,
                        processor, 
                        wav2vec_model, 
                        config.device
                    )
                    
                    if seg_stutters:
                        segments_with_stutters += 1
                        total_stutters += len(seg_stutters)
                        wav2vec_stutters.extend(seg_stutters)
                        
                        # If stutter at beginning, trim segment start
                        if new_start > seg.start:
                            logger.debug(f"Wav2Vec2: Trimming segment start {seg.start:.2f}s -> {new_start:.2f}s")
                            seg.start = new_start
                
                if total_stutters > 0:
                    logger.info(f"Wav2Vec2: Found {total_stutters} stutters in {segments_with_stutters} segments")
                    # Also apply to word-level for additional cleanup
                    segments = apply_wav2vec_stutters(segments, wav2vec_stutters)
                else:
                    logger.info("Wav2Vec2: No additional stutters found (Whisper already got them)")
                    
                # Free Wav2Vec2 model from GPU memory to make room for LLM later
                del wav2vec_model, processor
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logger.warning(f"Wav2Vec2 stutter detection failed: {e}")
    
    # Step 1.25: Word-level retake trimming (trim false starts WITHIN segments)
    logger.info("\n[1.25/6] Word-Level Retake Trimming...")
    segments = trim_segment_retakes(segments)
    
    # Step 1.3: Script-guided word cleanup (if script provided)
    if config.script_path and config.script_path.exists():
        logger.info("\n[1.3/6] Script-Guided Word Cleanup...")
        with open(config.script_path, 'r', encoding='utf-8') as f:
            script_text = f.read()
        segments = script_guided_word_cleanup(segments, script_text)
    
    # Step 1.5: Script-Guided OR Fuzzy Retake Detection
    skip_llm_for_cuts = config.skip_llm_cuts  # User can force skip via --skip-llm flag
    
    if config.script_path and config.script_path.exists():
        logger.info("\n[1.5/6] Script-Guided Analysis (PRIMARY MODE)...")
        logger.info("Script provided - ALL CONTENT KEPT (script + improvisation)")
        script_sentences = load_script(config.script_path)
        script_decisions = analyze_with_script(segments, script_sentences, match_threshold=0.5)
        
        # Since analyze_with_script now returns all-keep, this will be empty
        bad_take_indices = {
            i for i, d in enumerate(script_decisions) 
            if not d.keep
        }
        fuzzy_decisions = script_decisions
        skip_llm_for_cuts = True  # Script is ground truth - no need for LLM cuts
        
        logger.info(f"Script analysis: {len(bad_take_indices)} cuts, {len(segments) - len(bad_take_indices)} kept")
    else:
        logger.info("\n[1.5/6] Fuzzy Retake Detection...")
        fuzzy_decisions = analyze_retakes_fuzzy(segments, config.similarity_threshold)
        
        # Identify segments to skip in LLM analysis (already marked as bad)
        bad_take_indices = {
            i for i, d in enumerate(fuzzy_decisions) 
            if not d.keep
        }
        
        if skip_llm_for_cuts:
            logger.info(f"Fuzzy matcher caught {len(bad_take_indices)} retakes. LLM for cuts SKIPPED (--skip-llm flag).")
        else:
            logger.info(f"Fuzzy matcher caught {len(bad_take_indices)} retakes. Sending {len(segments) - len(bad_take_indices)} segments to LLM.")
    
    # Filter segments for LLM (only keep ones that passed fuzzy/script check)
    segments_for_llm = [
        s for i, s in enumerate(segments) 
        if i not in bad_take_indices
    ]

    # Step 2: LLM Analysis for CUTS (SKIPPED when --skip-llm or script mode)
    # NOTE: Even when skipped for cuts, LLM will be used for B-roll suggestions if APIs are configured
    if skip_llm_for_cuts:
        logger.info("\n[2/6] LLM Cut Analysis... SKIPPED (algorithmic mode)")
        llm_decisions = []
        # Still try to get B-roll suggestions if APIs are available
        if config.pexels_api_key or config.pixabay_api_key or config.local_broll_dir.exists():
            logger.info("    (LLM will still be used for B-roll suggestions in step 4)")
            # We'll generate B-roll requests later in the pipeline
            broll_requests = []
        else:
            broll_requests = []
    else:
        logger.info("\n[2/6] LLM Analysis...")
        llm_decisions, broll_requests = analyze_with_llm(segments_for_llm, config)
    
    # Merge decisions
    # Start with all fuzzy decisions (mostly "keeps", some "drops")
    # Then update the "keeps" with LLM's decisions
    
    final_decisions = []
    llm_iter = iter(llm_decisions)
    
    # We need to map LLM decisions back to the original timeline
    # Since LLM chunks might not align 1:1 with segments, we rely on timestamps
    
    # Simpler merge strategy:
    # 1. Create a master list of decisions initialized to KEEP
    # 2. Apply Fuzzy cuts (set keep=False)
    # 3. Apply LLM cuts (set keep=False for ranges covered by LLM drop decisions)
    
    # Initialize all as KEEP
    merged_decisions = [
        EditDecision(s.start, s.end, keep=True, reason="Default")
        for s in segments
    ]
    
    # Apply Fuzzy cuts
    for i, d in enumerate(fuzzy_decisions):
        if not d.keep:
            merged_decisions[i].keep = False
            merged_decisions[i].reason = d.reason
            
    # Apply LLM cuts
    # LLM decisions cover time ranges. We need to find which segments fall into those ranges.
    llm_cuts = 0
    for llm_d in llm_decisions:
        if not llm_d.keep:
            # Find all segments fully contained in this LLM decision's time range
            for md in merged_decisions:
                # Allow small tolerance for float comparison
                if md.start >= llm_d.start - 0.1 and md.end <= llm_d.end + 0.1:
                    if md.keep:  # Only count if not already marked
                        llm_cuts += 1
                    md.keep = False
                    md.reason = f"{md.reason} + {llm_d.reason}"
        
        # Also merge B-roll keywords if kept
        if llm_d.keep and llm_d.broll_keywords:
            # Assign B-roll to the first segment in this range
            for md in merged_decisions:
                if md.start >= llm_d.start - 0.1:
                    md.broll_keywords = llm_d.broll_keywords
                    break
    
    # Summary logging
    total_segments = len(merged_decisions)
    kept_segments = sum(1 for d in merged_decisions if d.keep)
    cut_segments = total_segments - kept_segments
    original_duration = sum(s.duration for s in segments)
    kept_duration = sum(s.duration for i, s in enumerate(segments) if merged_decisions[i].keep)
    
    logger.info(f"Merge Summary: {kept_segments}/{total_segments} segments kept ({cut_segments} cut)")
    logger.info(f"Duration: {original_duration:.1f}s -> {kept_duration:.1f}s (cut {original_duration - kept_duration:.1f}s / {100*(1-kept_duration/original_duration):.1f}%)")
    logger.info(f"Cut breakdown: {len(bad_take_indices)} fuzzy retakes + {llm_cuts} LLM decisions")

    # Step 3: Apply Cuts
    logger.info("\n[3/6] Applying Cuts...")
    cut_video = apply_cuts(current_input, merged_decisions, config)
    
    # Step 4: B-roll Composite
    logger.info("\n[4/6] B-roll Composite...")
    if broll_requests and (config.pexels_api_key or config.pixabay_api_key or config.local_broll_dir.exists()):
        composited = composite_broll(cut_video, broll_requests, config)
    else:
        logger.info("Skipping B-roll (no API key or local folder)")
        composited = cut_video
    
    # Step 5: Chapters (optional)
    if config.chapters:
        logger.info("\n[5/6] Generating Chapters...")
        chapters = generate_chapters(segments)
        chapters_path = config.output_path.with_suffix(".chapters.txt")
        chapters_path.write_text(chapters, encoding="utf-8")
        print(f"\nYouTube Chapters:\n{chapters}")
    
    # Step 6: Subtitles (optional)
    current_output = composited
    if config.subtitles:
        logger.info("\n[6/6] Embedding Subtitles...")
        srt_path = config.temp_dir / "subtitles.srt"
        generate_subtitles(segments, srt_path)
        
        subtitled = config.temp_dir / "subtitled.mp4"
        current_output = embed_subtitles(current_output, srt_path, subtitled)
    
    # Final copy to output path
    logger.info("\nFinalizing output...")
    shutil.copy(current_output, config.output_path)
    
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE: {config.output_path}")
    logger.info("=" * 60)
    
    return config.output_path


# =============================================================================
# CLI
# =============================================================================

def setup_args() -> argparse.Namespace:
    """Setup command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Local Video Editor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full auto mode with B-roll
  python ai_video_editor.py raw_video.mp4 final.mp4 --broll-api-key YOUR_KEY --subtitles

  # Just detect retakes (no editing)
  python ai_video_editor.py input.mp4 output.mp4 --mode detect-retakes

  # Face-tracking zoom only
  python ai_video_editor.py talking_head.mp4 zoomed.mp4 --mode dynamic-zoom

  # Silence removal via auto-editor
  python ai_video_editor.py podcast.mp4 cut.mp4 --mode cut-silence
        """,
    )
    
    parser.add_argument(
        "input",
        type=Path,
        help="Input video path",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output video path",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "cut-silence", "detect-retakes", "dynamic-zoom", "broll"],
        default="auto",
        help="Processing mode (default: auto)",
    )
    parser.add_argument(
        "--whisper-model",
        default="crisper",
        help="Whisper model: 'crisper' (default, verbatim with stutters/fillers) or 'large-v3' for standard Whisper",
    )
    parser.add_argument(
        "--gemini-model",
        default=GEMINI_DEFAULT_MODEL,
        help=f"Gemini model for LLM analysis (default: {GEMINI_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--gemini-api-key",
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device (default: cuda)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Retake detection similarity threshold (default: 0.75)",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=0.7,
        help="Minimum silence duration to detect (default: 0.7)",
    )
    parser.add_argument(
        "--pexels-api-key",
        help="Pexels API key for B-roll downloads (or set PEXELS_API_KEY env var)",
    )
    parser.add_argument(
        "--pixabay-api-key",
        help="Pixabay API key for B-roll downloads (or set PIXABAY_API_KEY env var)",
    )
    parser.add_argument(
        "--local-broll-dir",
        type=Path,
        default=Path("./cc0_clips"),
        help="Local B-roll folder (default: ./cc0_clips)",
    )
    parser.add_argument(
        "--chapters",
        action="store_true",
        help="Generate YouTube chapters",
    )
    parser.add_argument(
        "--subtitles",
        action="store_true",
        help="Embed subtitles in output",
    )
    parser.add_argument(
        "--pre-cut-silence",
        action="store_true",
        help="Run auto-editor to remove silence before AI processing (requires auto-editor)",
    )
    parser.add_argument(
        "--fix-drift",
        action="store_true",
        help="Convert input to Constant Frame Rate (CFR) to fix lip-sync drift issues",
    )
    parser.add_argument(
        "--audio-delay",
        type=int,
        default=0,
        help="Audio delay in milliseconds to fix OBS sync issues (positive delays audio, e.g. 550 for typical OBS drift)",
    )
    parser.add_argument(
        "--script",
        type=str,
        default=None,
        help="Path to the original script file for guided editing (matches transcript to script to avoid cutting intended content)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_false",
        dest="skip_llm_cuts",
        help="Enable LLM for cut decisions (disabled by default). Uses Gemini to decide which segments to cut.",
    )
    parser.add_argument(
        "--no-wav2vec2",
        action="store_true",
        dest="no_wav2vec2",
        help="Disable Wav2Vec2 acoustic stutter detection (second pass). Enabled by default for better stutter detection.",
    )
    parser.add_argument(
        "--no-audio-enhance",
        action="store_true",
        dest="no_audio_enhance",
        help="Disable audio enhancement (noise reduction + normalization). Enabled by default for cleaner audio.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = setup_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create config
    config = EditorConfig(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        whisper_model=args.whisper_model,
        gemini_model=args.gemini_model,
        gemini_api_key=args.gemini_api_key,
        device=args.device,
        similarity_threshold=args.similarity_threshold,
        silence_duration=args.silence_duration,
        pexels_api_key=args.pexels_api_key or PEXELS_API_KEY,
        pixabay_api_key=args.pixabay_api_key or PIXABAY_API_KEY,
        local_broll_dir=args.local_broll_dir,
        chapters=args.chapters,
        subtitles=args.subtitles,
        pre_cut_silence=args.pre_cut_silence,
        fix_drift=args.fix_drift,
        audio_delay_ms=args.audio_delay,
        script_path=Path(args.script) if args.script else None,
        skip_llm_cuts=args.skip_llm_cuts,
        use_wav2vec2=not args.no_wav2vec2,
        enhance_audio=not args.no_audio_enhance,
    )
    
    logger.info(f"Input: {config.input_path}")
    logger.info(f"Output: {config.output_path}")
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Device: {config.device}")
    if config.script_path:
        logger.info(f"Script: {config.script_path}")
    
    # Create temp directory
    if config.temp_dir is None:
        config.temp_dir = Path(tempfile.mkdtemp(prefix="ai_video_editor_"))
    logger.info(f"Temp directory: {config.temp_dir}")

    # Pre-processing: Fix Drift
    current_input = config.input_path
    if config.fix_drift:
        current_input = fix_video_drift(current_input, config)
        # Update config to point to fixed file for subsequent steps
        # But keep original input_path for reference if needed
    
    segments: Optional[List[TranscriptSegment]] = None
    current_output: Optional[Path] = None
    mode_label = config.mode.replace("-", " ").title()
    
    try:
        if config.mode == "cut-silence":
            result = cut_silence_auto_editor(config)
            current_output = result
            
        elif config.mode == "detect-retakes":
            segments = transcribe_video(current_input, config)
            retakes = analyze_retakes_fuzzy(segments, config.similarity_threshold)
            
            # Output as JSON
            output_data = [
                {"start": r.start, "end": r.end, "reason": r.reason}
                for r in retakes
            ]
            print(json.dumps(output_data, indent=2))
            
            # Also save to file
            retakes_path = config.output_path.with_suffix(".retakes.json")
            retakes_path.write_text(json.dumps(output_data, indent=2))
            logger.info(f"Retakes saved: {retakes_path}")
            if config.chapters or config.subtitles:
                logger.warning("Chapters/subtitles are unavailable in detect-retakes mode (no video output).")
            
        elif config.mode == "dynamic-zoom":
            temp_zoom = config.temp_dir / f"dynamic_zoom_{config.input_path.name}"
            apply_dynamic_zoom(current_input, temp_zoom, config)
            current_output = temp_zoom
            
        elif config.mode == "broll":
            segments = transcribe_video(current_input, config)
            decisions, broll_requests = analyze_with_llm(segments, config)
            cut_video = apply_cuts(current_input, decisions, config)
            
            if broll_requests:
                result = composite_broll(cut_video, broll_requests, config)
            else:
                result = cut_video
            current_output = result
            
        elif config.mode == "auto":
            # Update config.input_path temporarily for the pipeline if drift fix was applied
            original_input = config.input_path
            config.input_path = current_input
            run_full_pipeline(config)
            config.input_path = original_input # Restore
            logger.info("Auto mode complete.")
        
        if current_output is not None:
            final_video = current_output
            if config.chapters or config.subtitles:
                segments = ensure_segments_for_optional_outputs(
                    segments,
                    current_output,
                    config,
                )
                final_video = apply_optional_outputs(segments, current_output, config)
            shutil.copy(final_video, config.output_path)
            logger.info(f"{mode_label} mode complete: {config.output_path}")
            
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
