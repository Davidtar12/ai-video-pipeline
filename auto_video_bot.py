#!/usr/bin/env python3
"""
AutoVideoBot - Faceless Vertical Video Generator

Generates high-energy 9:16 vertical videos automatically from audio/transcript.

Components:
- Ear (Transcribe): faster-whisper for word-level timestamps
- Brain (Search Queries): Gemini 2.5 Flash for visual search queries
- Eyes (Stock Media): Pexels/Pixabay for video, Brave for celebrity images
- Hands (Editor): FFmpeg-based video stitching

Usage:
    python auto_video_bot.py "path/to/audio.wav" --output "output.mp4"
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict
from collections import defaultdict
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageChops, ImageEnhance, ImageOps, ImageFilter
import numpy as np

# Import entity extractor
from script_entity_extractor import extract_entities_from_script, match_name_fuzzy, ScriptEntities

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# API Keys from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
COVERR_API_KEY = os.getenv("COVERR_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
WEBSHARE_PROXY_URL = os.getenv("WEBSHARE_PROXY_URL")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

# Gemini configuration
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# OpenRouter fallback (Xiaomi)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
XIAOMI_MODEL = "xiaomi/mimo-v2-flash:free"
OLMO_MODEL = "allenai/olmo-3.1-32b-think:free"  # Reasoning fallback when Gemini rate limited

# Video settings
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30
CLIP_DURATION_DEFAULT = 5  # seconds per clip if no timing info
MIN_CLIP_DURATION = 2.0
MAX_LLM_RETRIES = 2
LLM_RETRY_DELAY = 3
MAX_REPLACEMENTS = 100  # effectively unlimited - replace ALL bad clips
VISION_MAX_WORKERS = 3  # limit parallel vision checks
BLACK_FRAME_THRESHOLD = 0.1  # detect black frames lasting > 0.1 seconds

# Watermark/Logo settings (for branding without killing reach)
WATERMARK_LOGO_PATH = r"C:\Users\USERNAME\Downloads\YouTube\FinTara\Logo\ChatGPT_Image_Dec_6__2025__05_22_09_PM-removebg-preview.png"
WATERMARK_SIZE = 120  # pixels (diameter for circular logo)
WATERMARK_OPACITY = 0.6  # 60% opacity (50-70% recommended for subtlety)
WATERMARK_POSITION = "top-right-safe"  # top-right-safe = 15% from top (avoids search bar UI)
WATERMARK_PADDING = 30  # pixels from edge (safe zone)

# Company Logos Directory (for automatic logo usage when companies mentioned)
COMPANY_LOGOS_DIR = Path(r"C:\Users\USERNAME\Downloads\YouTube\Companies")

# Video cache settings (prevent repetition between videos)
VIDEO_CACHE_FILE = Path(r"C:\Users\USERNAME\Downloads\YouTube\.video_cache.json")
VIDEO_COOLDOWN_DAYS = 7  # Videos can be reused after 7 days

MEDIA_CACHE: dict = {}

def _load_video_cache() -> dict:
    """Load persistent video usage cache with timestamps."""
    if not VIDEO_CACHE_FILE.exists():
        return {}
    try:
        with open(VIDEO_CACHE_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Failed to load video cache: %s", e)
        return {}

def _save_video_cache(cache: dict) -> None:
    """Persist video usage cache to disk."""
    try:
        VIDEO_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(VIDEO_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        logger.warning("Failed to save video cache: %s", e)

# Global persistent cache loaded once per process
GLOBAL_VIDEO_CACHE: dict = _load_video_cache()

# Company name → logo filename mapping (case-insensitive)
COMPANY_LOGO_MAP = {
    "amazon": "amazon.png",
    "amd": "amd.jpg",
    "apple": "apple.jpg",
    "applovin": "applovin.png",
    "ark": "ark invest.png",
    "ark invest": "ark invest.png",
    "arkk": "ark invest.png",
    "beam therapeutics": "Beam Therapeutics.png",
    "beam": "Beam Therapeutics.png",
    "berkshire": "berkshire.jpg",
    "berkshire hathaway": "berkshire.jpg",
    "broadcom": "broadcom.jpg",
    "constellation": "constellation.png",
    "ge vernova": "ge vernova.png",
    "vernova": "ge vernova.png",
    "google": "google.png",
    "alphabet": "google.png",
    "groq": "groq.png",
    "insmed": "Insmed Inc.png",
    "intel": "intel.png",
    "lam research": "lam research.png",
    "lumentum": "Lumentum Holdings Inc.png",
    "lumentum holdings": "Lumentum Holdings Inc.png",
    "meta": "Meta.jpg",
    "facebook": "Meta.jpg",
    "microsoft": "microsoft.png",
    "micron": "micron.png",
    "micron technology": "micron.png",
    "mp materials": "Mp Materials Corp.png",
    "mp materials corp": "Mp Materials Corp.png",
    "nvidia": "Nvidia.jpg",
    "palantir": "palantir.jpg",
    "palantir technologies": "palantir.jpg",
    "robinhood": "robinhood.png",
    "roblox": "roblox.png",
    "roku": "roku.jpg",
    "sandisk": "sandisk.png",
    "seagate": "seagate.jpg",
    "seagate technology": "seagate.jpg",
    "sk hynix": "sk hynix.png",
    "talen": "Talen Energy.png",
    "talen energy": "Talen Energy.png",
    "bloom": "Bloom Energy.jpg",
    "bloom energy": "Bloom Energy.jpg",
    "tesla": "tesla.jpg",
    "tsmc": "tsmc.png",
    "vistra": "vistra.png",
    "warner bros": "warner bros.png",
    "warner brothers": "warner bros.png",
    "western digital": "Western Digital.jpg",
}

# Company → CEO mapping (for videos where company name is over-repeated)
# If company appears 3+ times in script, use CEO image for visual variety
COMPANY_CEO_MAP = {
    "apple": "Tim Cook",
    "tesla": "Elon Musk",
    "microsoft": "Satya Nadella",
    "google": "Sundar Pichai",
    "alphabet": "Sundar Pichai",
    "meta": "Mark Zuckerberg",
    "facebook": "Mark Zuckerberg",
    "nvidia": "Jensen Huang",
    "amazon": "Jeff Bezos",
    "berkshire": "Warren Buffett",
    "berkshire hathaway": "Warren Buffett",
    "intel": "Pat Gelsinger",
    "tsmc": "C.C. Wei",
    "ark": "Cathie Wood",
    "ark invest": "Cathie Wood",
}

# Company → Stock Ticker mapping (loaded from CSV file for easy updates)
COMPANY_TICKER_CSV = COMPANY_LOGOS_DIR / "company_tickers.csv"

def _load_company_tickers() -> dict:
    """Load company ticker mappings from CSV file."""
    ticker_map = {}
    if not COMPANY_TICKER_CSV.exists():
        logger.warning("Company tickers CSV not found: %s", COMPANY_TICKER_CSV)
        return ticker_map

    try:
        import csv
        with open(COMPANY_TICKER_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('name', '').strip().lower()
                ticker = row.get('ticker', '').strip().upper()
                if name and ticker:
                    ticker_map[name] = ticker
    except Exception as e:
        logger.warning("Failed to load company tickers: %s", e)
    
    return ticker_map

# Loaded tickers
COMPANY_TICKER_MAP = _load_company_tickers()

# Blocklist for unwanted imagery (checked in queries and vision verification)
IMAGE_BLOCKLIST = [
    "heaven", "sky", "clouds", "bonfire", "fire", "flames", "nature",
    "mountain", "mountains", "ocean", "sea", "beach", "forest", "trees",
    "sunset", "sunrise", "landscape", "waterfall", "river", "lake",
    "abstract", "artistic", "spiritual", "religious", "meditation"
]

# Vision models for clip verification
QWEN_VL_MODEL = "qwen/qwen-2.5-vl-7b-instruct:free"
NEMOTRON_VL_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"

# Proxy configuration
# Webshare proxy IPs for DNS fallback (if p.webshare.io is blocked)
WEBSHARE_PROXY_IPS = [
    "51.77.20.223", "141.95.202.227", "141.94.162.15", "54.38.13.221",
    "135.125.3.89", "141.95.173.161", "141.95.157.159", "54.38.13.176",
]


def _fix_proxy_dns(proxy_url: str) -> str:
    """Replace hostname with IP if DNS fails (keeps credentials intact)."""
    if not proxy_url:
        return ""
    import socket

    parsed = urlparse(proxy_url)
    hostname = parsed.hostname
    try:
        result = socket.getaddrinfo(hostname, parsed.port, socket.AF_INET, socket.SOCK_STREAM)
        ip = result[0][4][0]
        if ip and ip != "0.0.0.0":
            return proxy_url  # DNS works
    except socket.gaierror:
        pass

    if WEBSHARE_PROXY_IPS:
        fallback_ip = WEBSHARE_PROXY_IPS[0]
        logger.warning("Proxy DNS failed for %s, using fallback IP: %s", hostname, fallback_ip)
        netloc = f"{parsed.username}:{parsed.password}@{fallback_ip}:{parsed.port}"
        return f"{parsed.scheme}://{netloc}"
    return proxy_url


def _build_us_proxy_url(proxy_url: str) -> str:
    """Convert proxy URL to US-specific by adding country suffix."""
    if not proxy_url:
        return ""
    parsed = urlparse(proxy_url)
    if parsed.username and "-country-" not in parsed.username:
        new_username = f"{parsed.username}-country-US-session-{random.randint(1000, 9999)}"
        return proxy_url.replace(parsed.username, new_username)
    return proxy_url


WEBSHARE_PROXY_URL_FIXED = _fix_proxy_dns(WEBSHARE_PROXY_URL) if WEBSHARE_PROXY_URL else ""
WEBSHARE_PROXY_URL_US = _build_us_proxy_url(WEBSHARE_PROXY_URL_FIXED) if WEBSHARE_PROXY_URL_FIXED else ""
REQUESTS_PROXY_US = {"http": WEBSHARE_PROXY_URL_US, "https": WEBSHARE_PROXY_URL_US} if WEBSHARE_PROXY_URL_US else {}

# Request settings
REQUEST_TIMEOUT = 60

# ---------------------------------------------------------------------------
# Caption Post-Processing (convert spoken numbers to digits)
# ---------------------------------------------------------------------------

def post_process_caption_text(text: str, language: str = "en") -> str:
    """Convert spoken numbers to digits for cleaner captions.
    
    Examples:
        "five million percent" → "5,000,000%"
        "fifteen point one percent" → "15.1%"
        "zero point zero three percent" → "0.03%"
        "quince por ciento" → "15%"
    """
    import re
    
    # Number word mappings
    ones_en = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
        'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
        'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '000',
    }
    
    ones_es = {
        'cero': '0', 'uno': '1', 'dos': '2', 'tres': '3', 'cuatro': '4',
        'cinco': '5', 'seis': '6', 'siete': '7', 'ocho': '8', 'nueve': '9',
        'diez': '10', 'once': '11', 'doce': '12', 'trece': '13', 'catorce': '14',
        'quince': '15', 'dieciséis': '16', 'diecisiete': '17', 'dieciocho': '18',
        'diecinueve': '19', 'veinte': '20', 'treinta': '30', 'cuarenta': '40',
        'cincuenta': '50', 'sesenta': '60', 'setenta': '70', 'ochenta': '80',
        'noventa': '90', 'cien': '100', 'mil': '000',
    }
    
    result = text
    
    # Common spoken number patterns → digit conversions
    replacements_en = [
        # Percentages with decimals
        (r'(\d+)\s+point\s+(\d+)\s+percent', r'\1.\2%'),
        (r'zero\s+point\s+zero\s+(\w+)\s+percent', lambda m: f"0.0{ones_en.get(m.group(1), m.group(1))}%"),
        (r'zero\s+point\s+(\w+)\s+percent', lambda m: f"0.{ones_en.get(m.group(1), m.group(1))}%"),
        (r'(\w+)\s+point\s+(\w+)\s+percent', lambda m: f"{ones_en.get(m.group(1), m.group(1))}.{ones_en.get(m.group(2), m.group(2))}%"),
        # Large numbers
        (r'(\d+(?:\.\d+)?)\s+million\s+percent', r'\1M%'),
        (r'(\d+(?:\.\d+)?)\s+million\s+dollars', r'$\1M'),
        (r'(\d+(?:\.\d+)?)\s+billion\s+dollars', r'$\1B'),
        (r'(\d+(?:\.\d+)?)\s+thousand\s+dollars', r'$\1K'),
        # Simple percentages
        (r'(\w+)\s+percent', lambda m: f"{ones_en.get(m.group(1).lower(), m.group(1))}%" if m.group(1).lower() in ones_en else m.group(0)),
        # Ticker corrections (common Whisper errors)
        (r'\b[Ii][Bb][Bb]\b', 'IVV'),
        (r'\b[Bb][Oo][Oo]\b', 'VOO'),
        (r'\b[Ss][Pp][Yy][Ee]?\b', 'SPY'),
        (r'\b[Vv][Gg][Tt]\b', 'VGT'),
        (r'\b[Ss][Mm][Hh]\b', 'SMH'),
        (r'\b[Qq][Qq][Qq]\b', 'QQQ'),
    ]
    
    replacements_es = [
        # Percentages with decimals (Spanish uses comma)
        (r'(\d+)\s+coma\s+(\d+)\s+por\s+ciento', r'\1,\2%'),
        (r'cero\s+coma\s+cero\s+(\w+)\s+por\s+ciento', lambda m: f"0,0{ones_es.get(m.group(1), m.group(1))}%"),
        (r'cero\s+coma\s+(\w+)\s+por\s+ciento', lambda m: f"0,{ones_es.get(m.group(1), m.group(1))}%"),
        # Large numbers
        (r'(\d+(?:,\d+)?)\s+millones?\s+por\s+ciento', r'\1M%'),
        (r'(\d+(?:,\d+)?)\s+millones?\s+de\s+dólares', r'$\1M'),
        (r'(\d+(?:,\d+)?)\s+mil\s+millones?\s+de\s+dólares', r'$\1B'),
        (r'(\d+(?:,\d+)?)\s+mil\s+dólares', r'$\1K'),
        # Simple percentages
        (r'(\w+)\s+por\s+ciento', lambda m: f"{ones_es.get(m.group(1).lower(), m.group(1))}%" if m.group(1).lower() in ones_es else m.group(0)),
        # Ticker corrections
        (r'\b[Ii][Bb][Bb]\b', 'IVV'),
        (r'\b[Bb][Oo][Oo]\b', 'VOO'),
    ]
    
    replacements = replacements_es if language == "es" else replacements_en
    
    for pattern, replacement in replacements:
        if callable(replacement):
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        else:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result


# ---------------------------------------------------------------------------
# Script-Whisper Hybrid Alignment (for accurate captions)
# ---------------------------------------------------------------------------

def _clean_word_for_alignment(word: str) -> str:
    """Remove punctuation for word comparison."""
    import re
    return re.sub(r'[^\w]', '', word.lower())


def align_script_to_whisper_words(
    script_text: str,
    whisper_words: List[Dict]  # [{"word": "...", "start": 0.0, "end": 0.5}, ...]
) -> List[Dict]:
    """
    Align script words to Whisper word timings.
    
    Uses script text for accuracy, Whisper only for timing.
    This fixes mishearings like "John" → "Lumentum", "SIGAT" → "Seagate".
    
    Returns list of {"word": script_word, "start": float, "end": float}
    """
    import difflib
    
    if not script_text or not whisper_words:
        return whisper_words  # Fallback to original if no script
    
    # Tokenize script (preserve punctuation in output)
    script_tokens = script_text.split()
    script_clean = [_clean_word_for_alignment(w) for w in script_tokens]
    
    # Get whisper words cleaned
    whisper_clean = [_clean_word_for_alignment(w["word"]) for w in whisper_words]
    
    # Use SequenceMatcher to find alignment
    matcher = difflib.SequenceMatcher(None, whisper_clean, script_clean)
    
    aligned = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Words match - use script text with whisper timing
            for wi, si in zip(range(i1, i2), range(j1, j2)):
                aligned.append({
                    "word": script_tokens[si],
                    "start": whisper_words[wi]["start"],
                    "end": whisper_words[wi]["end"],
                })
        elif tag == 'replace':
            # Words differ - use script text with whisper timing
            whisper_chunk = whisper_words[i1:i2]
            script_chunk = script_tokens[j1:j2]
            
            if len(whisper_chunk) == len(script_chunk):
                # 1:1 replacement
                for wi, si in zip(range(len(whisper_chunk)), range(len(script_chunk))):
                    aligned.append({
                        "word": script_chunk[si],
                        "start": whisper_chunk[wi]["start"],
                        "end": whisper_chunk[wi]["end"],
                    })
            elif whisper_chunk:
                # Different word counts - interpolate timing
                total_start = whisper_chunk[0]["start"]
                total_end = whisper_chunk[-1]["end"]
                duration_per_word = (total_end - total_start) / len(script_chunk) if script_chunk else 0
                
                for idx, sw in enumerate(script_chunk):
                    aligned.append({
                        "word": sw,
                        "start": total_start + idx * duration_per_word,
                        "end": total_start + (idx + 1) * duration_per_word,
                    })
        elif tag == 'insert':
            # Script has words Whisper missed - interpolate timing
            if aligned:
                last_end = aligned[-1]["end"]
            else:
                last_end = 0.0
            
            if i2 < len(whisper_words):
                next_start = whisper_words[i2]["start"]
            else:
                next_start = last_end + 0.5
            
            duration = (next_start - last_end) / (j2 - j1) if (j2 - j1) > 0 else 0.3
            
            for idx, si in enumerate(range(j1, j2)):
                aligned.append({
                    "word": script_tokens[si],
                    "start": last_end + idx * duration,
                    "end": last_end + (idx + 1) * duration,
                })
        # 'delete' - Whisper heard words not in script - skip them
    
    return aligned


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TranscriptSegment:
    """A segment of transcribed audio with timing."""
    text: str
    start: float  # seconds
    end: float    # seconds
    words: List[dict] = field(default_factory=list)
    ticker: str = ""  # Stock ticker if company mentioned (e.g., "AAPL")

@dataclass
class VisualQuery:
    """A visual search query generated from transcript."""
    segment: TranscriptSegment
    query: str
    is_person: bool = False
    person_name: str = ""
    is_company: bool = False
    company_name: str = ""

@dataclass
class MediaClip:
    """A media clip (video or image) with metadata."""
    path: Path
    duration: float
    is_image: bool = False
    source: str = ""  # pexels, pixabay, brave, fallback

# ---------------------------------------------------------------------------
# AutoVideoBot Class
# ---------------------------------------------------------------------------

class AutoVideoBot:
    """Main class for automated video generation."""
    
    def __init__(self, output_dir: Path = None, keep_temp: bool = False):
        self.output_dir = output_dir or Path(tempfile.mkdtemp(prefix="autovideo_"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_clips_dir = self.output_dir / "clips"
        self.temp_clips_dir.mkdir(exist_ok=True)
        self.temp_images_dir = self.output_dir / "images"
        self.temp_images_dir.mkdir(exist_ok=True)
        
        # Clear MEDIA_CACHE to prevent cross-video pollution inside a single process
        MEDIA_CACHE.clear()
        
        # Store transcript for subtitles
        self._transcript_segments: List[TranscriptSegment] = []
        # Store visual queries for clip verification
        self._visual_queries: List[VisualQuery] = []
        # Store topic summary for vision verification
        self._video_topic: str = ""
        # Track tickers already shown (once per video)
        self._shown_tickers: set = set()
        # Store proper nouns for WhisperWithHints subtitle accuracy
        self._proper_nouns: List[str] = []
        # Store audio duration for ensuring video covers full audio
        self._audio_duration: float = 0.0
        
        # Track used video URLs (per-video, prevents intra-video repetition)
        self._used_video_urls: set = set()
        
        # Load global video cache (persistent across videos, prevents inter-video repetition)
        self._video_cache: dict = GLOBAL_VIDEO_CACHE.copy()
        self._audio_duration: float = 0.0
        # Store proper nouns for subtitle spelling (e.g., "Greg Abel" not "gregavel")
        self._proper_nouns: List[str] = []
        # Store extracted entities from original script (for name matching)
        self._script_entities: Optional[ScriptEntities] = None
        # Store spoken script text for hybrid alignment (script text + whisper timing)
        self._spoken_script: str = ""
        self.keep_temp = keep_temp

        # Brave API rate limit (1 req/sec) guard
        self._brave_lock = threading.Lock()
        self._brave_next_time = 0.0
        # Track used video URLs to avoid repetition (for non-person/company clips)
        self._used_video_urls: set = set()
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio file duration using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception as e:
            logger.warning("Failed to get audio duration: %s", e)
        return 0.0
        
        logger.info("AutoVideoBot initialized. Output dir: %s", self.output_dir)
        
    # -----------------------------------------------------------------------
    # EAR: Transcription with faster-whisper
    # -----------------------------------------------------------------------
    
    def transcribe_audio(self, audio_path: Path, model_size: str = "base", language: Optional[str] = None) -> List[TranscriptSegment]:
        """Transcribe audio using faster-whisper with word-level timestamps."""
        logger.info("Transcribing audio: %s", audio_path)
        
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise SystemExit("faster-whisper not installed. Run: pip install faster-whisper")
        
        # Load model (uses GPU if available)
        model = WhisperModel(model_size, device="auto", compute_type="auto")
        
        # Transcribe with word timestamps
        segments_generator, info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language=language or None,
        )
        
        logger.info("Detected language: %s (probability: %.2f)", info.language, info.language_probability)
        
        segments = []
        for segment in segments_generator:
            words = []
            if segment.words:
                words = [{"word": w.word, "start": w.start, "end": w.end} for w in segment.words]
            
            segments.append(TranscriptSegment(
                text=segment.text.strip(),
                start=segment.start,
                end=segment.end,
                words=words,
            ))
            logger.debug("Segment [%.2f-%.2f]: %s", segment.start, segment.end, segment.text[:50])
        
        logger.info("Transcribed %d segments", len(segments))
        return segments
    
    def _detect_and_assign_tickers(self, segments: List[TranscriptSegment]) -> None:
        """Detect company mentions in segments and assign stock tickers (once per company)."""
        import re
        
        for segment in segments:
            text_lower = segment.text.lower()
            
            # Check each company in the ticker map
            for company_name, ticker in COMPANY_TICKER_MAP.items():
                # Skip if ticker already shown
                if ticker in self._shown_tickers:
                    continue
                
                # Use word boundaries for accurate matching (avoid "ark" matching "market")
                # For multi-word names, check full phrase
                pattern = r'\b' + re.escape(company_name) + r'\b'
                if re.search(pattern, text_lower):
                    segment.ticker = ticker
                    self._shown_tickers.add(ticker)
                    logger.info("📊 Detected company '%s' → ticker $%s at %.2f-%.2fs", 
                               company_name, ticker, segment.start, segment.end)
                    break  # Only one ticker per segment
    
    def _find_company_word_timing(self, segment: TranscriptSegment, company_name: str) -> Optional[float]:
        """Find when a company name starts being spoken in a segment using word timestamps.
        
        Returns the start time of the first word of the company name, or None if not found.
        """
        if not segment.words:
            return None
        
        company_lower = company_name.lower()
        company_words = company_lower.split()
        
        # Build word list with cleaned text
        words_clean = [(w["word"].strip().lower().strip(".,!?"), w["start"]) for w in segment.words]
        
        # Search for company name (single word or multi-word)
        for i, (word_text, word_start) in enumerate(words_clean):
            # Single word company match
            if len(company_words) == 1:
                if company_lower in word_text or word_text in company_lower:
                    return word_start
            else:
                # Multi-word: check if this starts the company name
                if company_words[0] in word_text or word_text in company_words[0]:
                    # Verify remaining words follow
                    match = True
                    for j, cw in enumerate(company_words[1:], 1):
                        if i + j < len(words_clean):
                            if cw not in words_clean[i + j][0] and words_clean[i + j][0] not in cw:
                                match = False
                                break
                        else:
                            match = False
                            break
                    if match:
                        return word_start
        
        return None

    # -----------------------------------------------------------------------
    # BRAIN: Visual Query Generation with Gemini
    # -----------------------------------------------------------------------
    
    def generate_visual_queries(self, segments: List[TranscriptSegment]) -> List[VisualQuery]:
        """Use Gemini to generate visual search queries for each segment."""
        logger.info("Generating visual queries for %d segments", len(segments))
        
        # Prepare transcript for Gemini
        transcript_text = "\n".join([
            f"[{i+1}] ({s.start:.1f}s - {s.end:.1f}s): {s.text}"
            for i, s in enumerate(segments)
        ])
        
        # Build entity hints from extracted script entities
        entity_hints = ""
        if self._script_entities:
            if self._script_entities.people:
                people_list = ", ".join(self._script_entities.people)
                entity_hints += f"\n**IMPORTANT - PEOPLE IN THIS VIDEO**: {people_list}\n"
                entity_hints += f"→ You MUST generate 'IMAGE: [name]' queries for these people when they are mentioned.\n"
            if self._script_entities.companies:
                companies_list = ", ".join(self._script_entities.companies)
                entity_hints += f"\n**IMPORTANT - COMPANIES IN THIS VIDEO**: {companies_list}\n"
                entity_hints += f"→ You MUST generate 'COMPANY: [name]' queries for these companies when they are mentioned.\n"
        
        prompt = f"""You are a visual director for a High-End Finance YouTube channel.
Your goal is to select stock footage that is STRICTLY finance and business related.
{entity_hints}
CRITICAL RULES:
1. If the text mentions a FAMOUS PERSON (e.g., Warren Buffett, Elon Musk), output: IMAGE: [Full Name]
2. If the text mentions a FAMOUS COMPANY (e.g., Berkshire Hathaway, Apple, Tesla, Goldman Sachs), output: COMPANY: [Company Name]
3. ONLY use these FINANCE-SPECIFIC visual categories:
   - Stock market: trading floors, stock tickers, charts, graphs, red/green screens
   - Money: cash, dollars, gold bars, coins, vaults, stacks of money
   - Business: corporate offices, boardrooms, handshakes, suits, skyscrapers
   - Documents: contracts, stock certificates, financial reports
   - Technology: computer screens with data, Bloomberg terminals
3. NEVER use these BLOCKLISTED concepts: heaven, sky, clouds, bonfire, fire, flames, nature, mountains, oceans, beach, forest, trees, sunset, sunrise, landscape, waterfall, river, lake, abstract, artistic, spiritual, religious, meditation, weather, animals, plants
4. NEVER use abstract art or non-business imagery
5. ALWAYS include "finance" or "business" in every query
6. Add "cinematic 4k" to every query for quality

GOOD EXAMPLES:
- "Warren Buffett invests" → IMAGE: Warren Buffett
- "Berkshire Hathaway reported earnings" → COMPANY: Berkshire Hathaway
- "Apple stock rose" → COMPANY: Apple
- "Goldman Sachs analysts" → COMPANY: Goldman Sachs
- "Market dropped 20%" → stock market crash trading floor red screens finance cinematic 4k
- "Diversification" → financial portfolio pie chart stocks bonds investing cinematic 4k
- "Compound interest" → stacks of money growing wealth finance cinematic 4k
- "Long term investing" → stock market charts upward trend business finance cinematic 4k
- "Risk management" → businessman analyzing financial data office cinematic 4k

BAD (NEVER USE):
- Mountains, oceans, nature, weather, animals, plants, abstract shapes

TRANSCRIPT:
{transcript_text}

Respond ONLY with valid JSON:
[
    {{"segment": 1, "query": "..."}},
    {{"segment": 2, "query": "IMAGE: Person Name"}},
    ...
]"""

        # Try Gemini first, fallback to OpenRouter
        response_text = self._call_llm(prompt)
        
        if not response_text:
            logger.error("LLM returned empty response")
            return self._fallback_queries(segments)
        
        # Parse JSON response
        try:
            # Clean markdown code blocks if present
            response_text = re.sub(r'```json\s*', '', response_text)
            response_text = re.sub(r'```\s*', '', response_text)
            queries_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", e)
            logger.debug("Response was: %s", response_text[:500])
            return self._fallback_queries(segments)
        
        # Build VisualQuery objects
        queries = []
        for i, segment in enumerate(segments):
            query_text = ""
            for q in queries_data:
                if q.get("segment") == i + 1:
                    query_text = q.get("query", "")
                    break
            
            if not query_text:
                query_text = f"abstract {segment.text[:30]}"
            
            is_person = query_text.upper().startswith("IMAGE:")
            is_company = query_text.upper().startswith("COMPANY:")
            person_name = ""
            company_name = ""
            
            if is_person:
                person_name = query_text[6:].strip()
                # Correct name spelling using extracted entities
                if self._script_entities and self._script_entities.people:
                    person_name = match_name_fuzzy(person_name, self._script_entities.people)
                query_text = person_name
            elif is_company:
                company_name = query_text[8:].strip()
                # Correct company spelling using extracted entities
                if self._script_entities and self._script_entities.companies:
                    company_name = match_name_fuzzy(company_name, self._script_entities.companies)
                query_text = company_name
            
            queries.append(VisualQuery(
                segment=segment,
                query=query_text,
                is_person=is_person,
                person_name=person_name,
                is_company=is_company,
                company_name=company_name,
            ))
        
        # Post-process: Split company/person queries to sync logo with voiceover
        queries = self._split_entity_queries(queries)
        
        logger.info("Generated %d visual queries", len(queries))
        return queries
    
    def _split_entity_queries(self, queries: List[VisualQuery]) -> List[VisualQuery]:
        """Split company/person queries so logo appears when name is spoken, not before.
        
        If a company is mentioned 0.5+ seconds into the segment, split the query:
        - First part: Generic finance B-roll until the company name
        - Second part: Company logo from when the name is spoken
        """
        MIN_SPLIT_DELAY = 0.5  # Only split if company mentioned 0.5s+ after segment start
        MIN_BROLL_DURATION = 0.3  # Minimum duration for B-roll segment
        
        new_queries = []
        
        for query in queries:
            # Only process company queries with word timestamps
            if not (query.is_company or query.is_person) or not query.segment.words:
                new_queries.append(query)
                continue
            
            # Find when the entity name is spoken
            entity_name = query.company_name if query.is_company else query.person_name
            entity_start = self._find_company_word_timing(query.segment, entity_name)
            
            if entity_start is None:
                new_queries.append(query)
                continue
            
            # Calculate delay from segment start to entity mention
            delay = entity_start - query.segment.start
            
            if delay < MIN_SPLIT_DELAY:
                # Entity mentioned early enough, no split needed
                new_queries.append(query)
                continue
            
            # SPLIT: Create B-roll query for intro, then entity query
            logger.info("📍 Splitting segment for '%s': B-roll %.2fs-%.2fs, Logo %.2fs-%.2fs",
                       entity_name, query.segment.start, entity_start, 
                       entity_start, query.segment.end)
            
            # Create intro segment (B-roll)
            intro_segment = TranscriptSegment(
                text=query.segment.text,
                start=query.segment.start,
                end=entity_start,
                words=[w for w in query.segment.words if w["end"] <= entity_start],
                ticker="",
            )
            intro_query = VisualQuery(
                segment=intro_segment,
                query="stock market finance business charts",  # Generic finance B-roll
                is_person=False,
                person_name="",
                is_company=False,
                company_name="",
            )
            new_queries.append(intro_query)
            
            # Create entity segment (logo from when name is spoken)
            entity_segment = TranscriptSegment(
                text=query.segment.text,
                start=entity_start,
                end=query.segment.end,
                words=[w for w in query.segment.words if w["start"] >= entity_start],
                ticker=query.segment.ticker,
            )
            entity_query = VisualQuery(
                segment=entity_segment,
                query=query.query,
                is_person=query.is_person,
                person_name=query.person_name,
                is_company=query.is_company,
                company_name=query.company_name,
            )
            new_queries.append(entity_query)
        
        if len(new_queries) != len(queries):
            logger.info("Split %d company/person queries → %d total queries", 
                       len(queries), len(new_queries))
        
        return new_queries
    
    def extract_entities_from_script(self, script: str):
        """Extract people and companies from script text.
        
        Args:
            script: Original script text before audio generation
        """
        logger.info("Extracting entities from script...")
        self._script_entities = extract_entities_from_script(script, self._call_llm)
        
        # Add extracted names to proper nouns for Whisper transcription
        if self._script_entities:
            all_names = self._script_entities.all_names()
            if all_names:
                self._proper_nouns.extend(all_names)
                logger.info("Added %d entity names to proper nouns list", len(all_names))
    
    def _call_llm(self, prompt: str) -> str:
        """Call Gemini, fallback to OpenRouter if rate limited."""
        last_error = None
        # Try Gemini first with light retries
        if GEMINI_API_KEY:
            for attempt in range(1, MAX_LLM_RETRIES + 1):
                try:
                    response = self._call_gemini(prompt)
                    if response:
                        return response
                except Exception as e:
                    last_error = e
                    if attempt < MAX_LLM_RETRIES:
                        time.sleep(LLM_RETRY_DELAY)
                    else:
                        logger.warning("Gemini failed after %d attempt(s): %s", attempt, e)
        
        # Fallback 1: OpenRouter (Xiaomi)
        if OPENROUTER_API_KEY:
            for attempt in range(1, MAX_LLM_RETRIES + 1):
                try:
                    response = self._call_openrouter(prompt)
                    if response:
                        return response
                except Exception as e:
                    last_error = e
                    if attempt < MAX_LLM_RETRIES:
                        time.sleep(LLM_RETRY_DELAY)
                    else:
                        logger.warning("OpenRouter (Xiaomi) fallback failed after %d attempt(s): %s", attempt, e)
        
        # Fallback 2: Olmo 3.1 32B Think (reasoning model)
        if OPENROUTER_API_KEY:
            for attempt in range(1, MAX_LLM_RETRIES + 1):
                try:
                    response = self._call_olmo(prompt)
                    if response:
                        logger.info("Olmo 3.1 32B Think fallback succeeded")
                        return response
                except Exception as e:
                    last_error = e
                    if attempt < MAX_LLM_RETRIES:
                        time.sleep(LLM_RETRY_DELAY)
                    else:
                        logger.error("Olmo fallback failed after %d attempt(s): %s", attempt, e)
        
        return ""
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        url = GEMINI_API_URL.format(model=GEMINI_MODEL)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 8192},
        }
        
        response = requests.post(
            url,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        
        if response.status_code == 429:
            raise RuntimeError("Gemini rate limited (429)")
        
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no candidates")
        return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    
    def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API (Xiaomi fallback)."""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/autovideo",
            "X-Title": "AutoVideoBot",
        }
        payload = {
            "model": XIAOMI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
        }
        
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter returned no choices")
        return choices[0].get("message", {}).get("content", "")
    
    def _call_olmo(self, prompt: str) -> str:
        """Call OpenRouter API with Olmo 3.1 32B Think (reasoning model fallback)."""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/autovideo",
            "X-Title": "AutoVideoBot",
        }
        payload = {
            "model": OLMO_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
            "reasoning": {"enabled": True},  # Enable reasoning for better quality
        }
        
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("Olmo returned no choices")
        return choices[0].get("message", {}).get("content", "")
    
    def _fallback_queries(self, segments: List[TranscriptSegment]) -> List[VisualQuery]:
        """Generate simple fallback queries without LLM."""
        return [
            VisualQuery(
                segment=s,
                query=f"business finance {s.text[:20]}",
                is_person=False,
            )
            for s in segments
        ]
    
    # -----------------------------------------------------------------------
    # EYES: Stock Media Fetching
    # -----------------------------------------------------------------------
    
    def fetch_media_for_queries(self, queries: List[VisualQuery]) -> List[MediaClip]:
        """Fetch video/image for each visual query."""
        logger.info("Fetching media for %d queries", len(queries))
        
        clips = []
        
        # Use parallel downloads for better performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all fetch tasks
            future_to_query = {
                executor.submit(self._fetch_single_media, i, query): (i, query)
                for i, query in enumerate(queries)
            }
            
            completed = 0
            total = len(queries)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_query):
                i, query = future_to_query[future]
                try:
                    clip = future.result()
                    clips.append((i, clip))  # Keep index for sorting
                    completed += 1
                    logger.info("Media fetch progress: %d/%d completed", completed, total)
                except Exception as e:
                    logger.warning("Failed to fetch media for query %d: %s", i, e)
                    # Create fallback
                    duration = self._safe_duration(query.segment.end - query.segment.start)
                    fallback = self._create_fallback_clip(i, duration)
                    clips.append((i, fallback))
                    completed += 1
        
        # Sort by original order
        clips.sort(key=lambda x: x[0])
        final_clips = [clip for _, clip in clips]
        
        logger.info("Fetched %d media clips", len(final_clips))
        return final_clips
    
    def _fetch_single_media(self, index: int, query: VisualQuery) -> MediaClip:
        """Fetch media for a single query (used in parallel)."""
        logger.info("[%d] Query: %s (person=%s, company=%s)", index+1, query.query[:40], query.is_person, query.is_company)
        
        duration = self._safe_duration(query.segment.end - query.segment.start)
        
        if query.is_person:
            # Search Wikipedia/Brave for person image
            clip = self._fetch_person_image(query.person_name, index, duration)
            if not clip:
                # Fallback to finance-tilted stock video if image missing
                finance_query = self._financeify_query(f"{query.person_name} portrait")
                clip = self._fetch_stock_video(finance_query, index, duration)
        elif query.is_company:
            # Search Wikipedia/Brave for company logo/image
            clip = self._fetch_company_image(query.company_name, index, duration)
            if not clip:
                # Fallback to finance-tilted stock video if image missing
                finance_query = self._financeify_query(f"{query.company_name} headquarters building")
                clip = self._fetch_stock_video(finance_query, index, duration)
        else:
            # Search Pexels/Pixabay for video
            finance_query = self._financeify_query(query.query)
            clip = self._fetch_stock_video(finance_query, index, duration)
        
        if not clip:
            # Fallback to solid color
            clip = self._create_fallback_clip(index, duration)
        
        return clip
    
    def _fetch_stock_video(self, query: str, index: int, duration: float) -> Optional[MediaClip]:
        """Fetch stock video with aggressive finance-focused fallback logic."""
        # GUARANTEED FINANCE fallbacks - these ALWAYS have results on Pexels/Pixabay
        finance_fallbacks = [
            "stock market trading screens",
            "money cash dollars bills",
            "business meeting corporate office",
            "financial charts graphs data",
            "gold bars investment wealth",
            "stock exchange wall street",
            "coins money savings finance",
            "corporate skyscraper building",
        ]
        
        # Build search attempts: original query -> 3 random finance fallbacks
        attempts = [query]
        attempts.extend(random.sample(finance_fallbacks, min(3, len(finance_fallbacks))))
        
        sources = []
        if PEXELS_API_KEY:
            sources.append(("pexels", self._search_pexels_video))
        if PIXABAY_API_KEY:
            sources.append(("pixabay", self._search_pixabay_video))
        if COVERR_API_KEY:
            sources.append(("coverr", self._search_coverr_video))
        
        if not sources:
            logger.warning("No stock video API keys configured")
            return None
        
        for search_term in attempts:
            # Try Pexels/Pixabay/Coverr (randomized order)
            random.shuffle(sources)
            for source_name, search_func in sources:
                try:
                    video_url = search_func(search_term)
                    if not video_url:
                        continue

                    # Cooldown check (persistent across videos)
                    last_used = self._video_cache.get(video_url)
                    cooldown_seconds = VIDEO_COOLDOWN_DAYS * 24 * 60 * 60
                    if last_used and time.time() - last_used < cooldown_seconds:
                        logger.debug("Skipping '%s' (cooldown)", video_url[:80])
                        continue

                    # Intra-video dedupe
                    if video_url in self._used_video_urls:
                        continue

                    video_path = self._download_video(video_url, index, source_name)
                    if video_path:
                        self._used_video_urls.add(video_url)  # Mark as used within this video
                        self._video_cache[video_url] = time.time()  # Mark as used across videos
                        logger.info("Found video for '%s' via %s", search_term[:30], source_name)
                        return MediaClip(path=video_path, duration=duration, is_image=False, source=source_name)
                except Exception as e:
                    logger.warning("%s search failed: %s", source_name, e)
        
        # PRIORITY 2: If Pexels/Pixabay failed, try company logo as B-roll backup
        company_logo_clip = self._get_company_logo(query, index, duration)
        if company_logo_clip:
            logger.info("🏢 Using company logo as B-roll backup: %s", query[:50])
            return company_logo_clip
        
        # LAST RESORT: Use a guaranteed simple query with variety
        fallback_queries = ["money", "stock market", "finance charts", "business office", "trading floor", "investment", "coins gold", "dollar bills", "financial data", "economy"]
        random.shuffle(fallback_queries)  # Randomize order for variety
        
        for fallback_query in fallback_queries:
            for source_name, search_func in sources:
                try:
                    video_url = search_func(fallback_query)
                    if not video_url:
                        continue
                    
                    # Check cooldown (persistent across videos)
                    last_used = self._video_cache.get(video_url)
                    cooldown_seconds = VIDEO_COOLDOWN_DAYS * 24 * 60 * 60
                    if last_used and time.time() - last_used < cooldown_seconds:
                        logger.debug("Skipping fallback '%s' (cooldown)", video_url[:50])
                        continue
                    
                    # Intra-video dedupe
                    if video_url in self._used_video_urls:
                        continue
                    
                    video_path = self._download_video(video_url, index, source_name)
                    if video_path:
                        self._used_video_urls.add(video_url)  # Mark as used within this video
                        self._video_cache[video_url] = time.time()  # Mark as used across videos
                        logger.info("Using fallback '%s' via %s", fallback_query, source_name)
                        return MediaClip(path=video_path, duration=duration, is_image=False, source=source_name)
                except:
                    pass
        
        return None
    
    def _search_pexels_video(self, query: str) -> Optional[str]:
        """Search Pexels for vertical video with improved variety."""
        headers = {"Authorization": PEXELS_API_KEY}
        # Randomize page (1-3) for variety across calls
        page = random.randint(1, 3)
        params = {
            "query": query,
            "orientation": "portrait",
            "size": "medium",
            "per_page": 15,  # More results for better dedupe
            "page": page,
        }
        
        response = requests.get(
            "https://api.pexels.com/videos/search",
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        
        videos = data.get("videos", [])
        if not videos:
            return None
        
        # Shuffle and try each video, skipping already-used ones
        random.shuffle(videos)
        for video in videos:
            # Find HD quality file
            for file in video.get("video_files", []):
                url = file.get("link")
                if url and url in self._used_video_urls:
                    continue  # Skip already used
                if file.get("quality") == "hd" and file.get("width", 0) < file.get("height", 0):
                    return url
            
            # Fallback to any file not already used
            if video.get("video_files"):
                url = video["video_files"][0].get("link")
                if url and url not in self._used_video_urls:
                    return url
        
        return None
    
    def _search_pixabay_video(self, query: str) -> Optional[str]:
        """Search Pixabay for vertical video with improved variety."""
        # Randomize page (1-3) for variety across calls
        page = random.randint(1, 3)
        params = {
            "key": PIXABAY_API_KEY,
            "q": query,
            "video_type": "film",
            "per_page": 20,  # More results for better dedupe
            "page": page,
        }
        
        response = requests.get(
            "https://pixabay.com/api/videos/",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        
        hits = data.get("hits", [])
        if not hits:
            return None
        
        # Shuffle and try each video, skipping already-used ones
        random.shuffle(hits)
        for video in hits:
            videos = video.get("videos", {})
            # Try medium quality first
            if "medium" in videos:
                url = videos["medium"].get("url")
                if url and url not in self._used_video_urls:
                    return url
            # Then try small
            if "small" in videos:
                url = videos["small"].get("url")
                if url and url not in self._used_video_urls:
                    return url
        
        return None
    
    def _search_coverr_video(self, query: str) -> Optional[str]:
        """Search Coverr for vertical video."""
        headers = {"Authorization": f"Bearer {COVERR_API_KEY}"}
        params = {
            "query": query,
            "page": 1,
            "per_page": 10,
        }
        
        try:
            response = requests.get(
                "https://api.coverr.co/videos",
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            
            videos = data.get("videos", [])
            if not videos:
                return None
            
            # Filter for vertical videos (9:16 or portrait)
            vertical_videos = [
                v for v in videos 
                if v.get("width", 0) < v.get("height", 0)
            ]
            
            if not vertical_videos:
                vertical_videos = videos  # Fallback to any video
            
            # Get random video from results
            video = random.choice(vertical_videos)
            
            # Get download URL (try HD first, then SD)
            urls = video.get("urls", {})
            if "download" in urls:
                return urls["download"]
            elif "mp4" in urls:
                return urls["mp4"]
            
            return None
        except Exception as e:
            logger.warning("Coverr search failed: %s", e)
            return None
    
    def _download_video(self, url: str, index: int, source: str) -> Optional[Path]:
        """Download video to temp directory."""
        output_path = self.temp_clips_dir / f"clip_{index:03d}_{source}.mp4"
        last_error = None
        for attempt in range(2):
            try:
                response = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
                response.raise_for_status()
                
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.debug("Downloaded video: %s", output_path.name)
                return output_path
            except Exception as e:
                last_error = e
                time.sleep(1)
        logger.warning("Failed to download video after retries: %s", last_error)
        return None
    
    def _get_company_logo(self, query: str, index: int, duration: float) -> Optional[MediaClip]:
        """Check if query mentions a company we have a logo for, and return logo clip.
        
        Args:
            query: Search query (may contain company name)
            index: Clip index
            duration: Duration for clip
            
        Returns:
            MediaClip with company logo if found, None otherwise
        """
        import re
        
        # Check if any company name is mentioned in the query (case-insensitive)
        query_lower = query.lower()
        
        for company_name, logo_filename in COMPANY_LOGO_MAP.items():
            # Use word boundaries for accurate matching (avoid "ark" matching "market")
            pattern = r'\b' + re.escape(company_name) + r'\b'
            if re.search(pattern, query_lower):
                logo_path = COMPANY_LOGOS_DIR / logo_filename
                
                if logo_path.exists():
                    logger.info("Using company logo for '%s': %s", company_name, logo_filename)
                    
                    # Process logo - scale to fit (not crop)
                    try:
                        filtered_logo = self._process_company_logo(logo_path, index)
                        return MediaClip(path=filtered_logo, duration=duration, is_image=True, source="company_logo")
                    except Exception as e:
                        logger.warning("Failed to process company logo %s: %s", logo_filename, e)
                else:
                    logger.warning("Company logo file not found: %s", logo_path)
        
        return None
    
    def _get_ceo_image_from_folder(self, person_name: str, index: int, duration: float) -> Optional[MediaClip]:
        """Check Companies folder for CEO images matching the person name.
        
        Looks for files like 'cathie wood ark invest.jpg' or 'cathie wood.jpg'.
        """
        person_lower = person_name.lower()
        
        # Search for matching files in Companies folder
        if not COMPANY_LOGOS_DIR.exists():
            return None
        
        try:
            for file_path in COMPANY_LOGOS_DIR.iterdir():
                if not file_path.is_file():
                    continue
                
                filename_lower = file_path.stem.lower()  # filename without extension
                
                # Check if person name appears in filename
                # e.g., "cathie wood ark invest.jpg" matches "Cathie Wood"
                if person_lower in filename_lower:
                    logger.info("Found CEO image in folder: %s", file_path.name)
                    
                    # Apply fair use filter
                    try:
                        filtered_path = self._apply_fair_use_filter(file_path, index)
                        return MediaClip(path=filtered_path, duration=duration, is_image=True, source="ceo_folder")
                    except Exception as e:
                        logger.warning("Failed to process CEO image %s: %s", file_path.name, e)
        except Exception as e:
            logger.warning("Error searching Companies folder for CEO: %s", e)
        
        return None
    
    def _fetch_person_image(self, person_name: str, index: int, duration: float) -> Optional[MediaClip]:
        """Fetch celebrity image from Wikipedia/Brave and apply fair use filter.
        
        For CEOs: tries Pexels/Pixabay first, then folder images, then Wikipedia/Brave.
        For other people: Wikipedia/Brave first, then company logos as fallback.
        """
        logger.info("Searching for image: %s", person_name)
        
        # Check if this is a known CEO - if so, prioritize Pexels/Pixabay + folder images
        is_ceo = any(person_name.lower() in ceo_name.lower() or ceo_name.lower() in person_name.lower() 
                     for ceo_name in COMPANY_CEO_MAP.values())
        
        if is_ceo:
            logger.info("👔 Detected CEO name: %s - trying stock photos first", person_name)
            # Try folder first for CEOs (higher quality controlled images)
            ceo_folder_clip = self._get_ceo_image_from_folder(person_name, index, duration)
            if ceo_folder_clip:
                logger.info("✓ Using CEO image from folder")
                return ceo_folder_clip
        
        # 1. Try Wikipedia first (free, reliable for famous people)
        image_url = self._search_wikipedia_image(person_name)
        source = "wikipedia"
        
        # 2. Try Brave as fallback (API may not support image search on all tiers)
        if not image_url:
            logger.info("Wikipedia failed, trying Brave...")
            image_url = self._search_brave_image(person_name)
            source = "brave"
        
        # 3. If Wikipedia/Brave failed, check if it's a company name and we have a logo
        if not image_url:
            company_logo_clip = self._get_company_logo(person_name, index, duration)
            if company_logo_clip:
                logger.info("🏢 Using company logo as backup for person image: %s", person_name)
                return company_logo_clip
        
        if not image_url:
            return None
        
        # Download image
        image_path = self._download_image(image_url, index)
        if not image_path:
            return None
        
        # Apply transformative filter for fair use
        filtered_path = self._apply_fair_use_filter(image_path, index)
        
        return MediaClip(
            path=filtered_path,
            duration=duration,
            is_image=True,
            source=source,
        )
    
    def _fetch_company_image(self, company_name: str, index: int, duration: float) -> Optional[MediaClip]:
        """Fetch company logo/image from local folder first, then Wikipedia/Brave as fallback."""
        logger.info("Searching for company image: %s", company_name)
        
        # 0. Try local folder FIRST (highest priority - curated logos)
        local_logo = self._get_company_logo(company_name, index, duration)
        if local_logo:
            logger.info("✓ Using company logo from local folder")
            return local_logo
        
        # 1. Try Wikipedia (free, reliable for famous companies)
        image_url = self._search_wikipedia_company_image(company_name)
        source = "wikipedia"
        
        # 2. Try Brave as fallback
        if not image_url:
            logger.info("Wikipedia failed for company, trying Brave...")
            image_url = self._search_brave_image(f"{company_name} logo")
            source = "brave"
        
        if not image_url:
            return None
        
        # Download image
        image_path = self._download_company_image(image_url, index)
        if not image_path:
            return None
        
        # Apply transformative filter for fair use
        filtered_path = self._apply_fair_use_filter(image_path, index)
        
        return MediaClip(
            path=filtered_path,
            duration=duration,
            is_image=True,
            source=source,
        )
    
    def _search_wikipedia_company_image(self, company_name: str) -> Optional[str]:
        """
        Fetch company logo/image from Wikipedia API.
        NO API KEY REQUIRED but requires User-Agent header to avoid 403.
        """
        try:
            headers = {
                "User-Agent": "AutoVideoBot/1.0 (github.com/autovideo)"
            }
            params = {
                "action": "query",
                "format": "json",
                "prop": "pageimages",
                "piprop": "original",
                "titles": company_name,
                "pithumbsize": 1000,
            }
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params=params,
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if page_id == "-1":
                    continue
                if "original" in page:
                    return page["original"]["source"]
                elif "thumbnail" in page:
                    return page["thumbnail"]["source"]
        except Exception as e:
            logger.warning("Wikipedia company search failed for %s: %s", company_name, e)
        return None
    
    def _download_company_image(self, url: str, index: int) -> Optional[Path]:
        """Download company image to temp directory."""
        output_path = self.temp_images_dir / f"company_{index:03d}.jpg"
        headers = {
            "User-Agent": "AutoVideoBot/1.0 (github.com/autovideo)"
        }
        last_error = None
        for attempt in range(2):
            try:
                response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                return output_path
            except Exception as e:
                last_error = e
                time.sleep(1)
        logger.warning("Failed to download company image after retries: %s", last_error)
        return None
    
    def _search_wikipedia_image(self, person_name: str) -> Optional[str]:
        """
        Backup: Fetch person image from Wikipedia API.
        NO API KEY REQUIRED but requires User-Agent header to avoid 403.
        """
        try:
            # REQUIRED: Tell Wikipedia who you are (or they block you)
            headers = {
                "User-Agent": "AutoVideoBot/1.0 (github.com/autovideo)"
            }
            params = {
                "action": "query",
                "format": "json",
                "prop": "pageimages",
                "piprop": "original",
                "titles": person_name,
                "pithumbsize": 1000,
            }
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params=params,
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                # Page ID "-1" means not found
                if page_id == "-1":
                    continue
                if "original" in page:
                    return page["original"]["source"]
                elif "thumbnail" in page:
                    # Fallback to thumbnail if no original
                    return page["thumbnail"]["source"]
        except Exception as e:
            logger.warning("Wikipedia search failed for %s: %s", person_name, e)
        return None
    
    def _throttle_brave(self):
        """Enforce 1 req/sec Brave API limit (thread-safe)."""
        with self._brave_lock:
            now = time.monotonic()
            sleep_for = self._brave_next_time - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._brave_next_time = time.monotonic() + 1.05

    def _search_brave_image(self, query: str) -> Optional[str]:
        """Search Brave images via official API; fallback to scrape if API missing."""
        if BRAVE_API_KEY:
            self._throttle_brave()
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": BRAVE_API_KEY,
            }
            params = {
                "q": query,
                "count": 5,
                "search_lang": "en",
                "safesearch": "strict",  # Brave only accepts 'off' or 'strict'
            }
            try:
                resp = requests.get(
                    "https://api.search.brave.com/res/v1/images/search",
                    headers=headers,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code == 429:
                    logger.warning("Brave API rate limited (429)")
                    return None
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results") or data.get("images", {}).get("results", [])
                for item in results:
                    url = (
                        item.get("properties", {}).get("url")
                        or item.get("thumbnail")
                        or item.get("url")
                    )
                    if url and url.startswith("http"):
                        return url
            except Exception as e:
                logger.warning("Brave API search failed: %s", e)

        # Use Brave's hidden JSON API (no key required, just user-agent)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }
        params = {
            "q": query,
            "source": "web",
        }
        
        try:
            resp = requests.get(
                "https://search.brave.com/api/images",
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                for item in results:
                    # Get full resolution image from properties.url
                    url = item.get("properties", {}).get("url")
                    if url and url.startswith("http"):
                        return url
        except Exception as e:
            logger.warning("Brave JSON API failed: %s", e)

        return None
    
    def _download_image(self, url: str, index: int) -> Optional[Path]:
        """Download image to temp directory."""
        output_path = self.temp_images_dir / f"person_{index:03d}.jpg"
        # User-Agent required for Wikipedia/Wikimedia
        headers = {
            "User-Agent": "AutoVideoBot/1.0 (github.com/autovideo)"
        }
        last_error = None
        for attempt in range(2):
            try:
                response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                return output_path
            except Exception as e:
                last_error = e
                time.sleep(1)
        logger.warning("Failed to download image after retries: %s", last_error)
        return None
    
    def _apply_fair_use_filter(self, image_path: Path, index: int) -> Path:
        """Apply transformative filter for fair use compliance."""
        output_path = self.temp_images_dir / f"person_{index:03d}_filtered.jpg"
        
        try:
            img = Image.open(image_path).convert("RGB")
            
            # Resize to 9:16 aspect ratio
            img = self._crop_to_aspect(img, 9, 16)
            img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
            
            # --- MODERN CORPORATE FINANCE COLOR GRADING ---
            
            # 1. Sharpen (Makes details pop for a "4K" look)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.3)
            
            # 2. Contrast & Saturation (The "Punchy" look)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.15)
            # Decrease saturation slightly for a more serious, corporate tone
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(0.85)
            
            # 3. The "Corporate Cool Tint" (Deep blue overlay)
            tint_layer = Image.new("RGB", img.size, (10, 30, 60))
            img_rgba = img.convert("RGBA")
            tint_rgba = tint_layer.convert("RGBA")
            overlay = Image.blend(img_rgba, tint_rgba, 0.15)  # Subtle 15% tint
            img = overlay.convert("RGB")
            
            # 4. Modern Soft Vignette
            vignette_layer = self._create_soft_vignette_mask(img.size)
            img = ImageChops.multiply(img, vignette_layer)
            
            # --- END COLOR GRADING ---
            
            img.save(output_path, quality=95)
            logger.debug("Applied modern finance filter: %s", output_path.name)
            return output_path
            
        except Exception as e:
            logger.warning("Failed to apply filter: %s", e)
            # Fallback: return resized image without effects
            try:
                img = Image.open(image_path).convert("RGB")
                img = self._crop_to_aspect(img, 9, 16)
                img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
                img.save(output_path, quality=90)
                return output_path
            except:
                return image_path
    
    def _process_company_logo(self, logo_path: Path, index: int) -> Path:
        """Process company logo - scale to fit within 9:16 frame with dark background."""
        output_path = self.temp_images_dir / f"company_{index:03d}_logo.jpg"
        
        try:
            # Create dark blue-grey background (9:16) - not too dark to avoid black frame detection
            background = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), (25, 35, 50))
            
            # Open and process logo
            logo = Image.open(logo_path)
            
            # Convert to RGBA if not already
            if logo.mode != "RGBA":
                logo = logo.convert("RGBA")
            
            # Calculate scaling to fit within frame (with padding)
            max_width = int(VIDEO_WIDTH * 0.7)  # 70% of frame width
            max_height = int(VIDEO_HEIGHT * 0.5)  # 50% of frame height
            
            # Scale logo to fit
            logo.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # Center logo on background
            x = (VIDEO_WIDTH - logo.width) // 2
            y = (VIDEO_HEIGHT - logo.height) // 2
            
            # Composite logo onto background (respecting alpha channel)
            background.paste(logo, (x, y), logo if logo.mode == "RGBA" else None)
            
            # Apply subtle effects for quality
            enhancer = ImageEnhance.Sharpness(background)
            background = enhancer.enhance(1.2)
            
            background.save(output_path, quality=95)
            logger.debug("Processed company logo: %s", output_path.name)
            return output_path
            
        except Exception as e:
            logger.warning("Failed to process company logo: %s", e)
            # Fallback: simple resize
            try:
                img = Image.open(logo_path).convert("RGB")
                img.thumbnail((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
                bg = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), (15, 20, 30))
                x = (VIDEO_WIDTH - img.width) // 2
                y = (VIDEO_HEIGHT - img.height) // 2
                bg.paste(img, (x, y))
                bg.save(output_path, quality=90)
                return output_path
            except:
                return logo_path
    
    def _crop_to_aspect(self, img: Image.Image, w_ratio: int, h_ratio: int) -> Image.Image:
        """Crop image to target aspect ratio (center crop)."""
        w, h = img.size
        target_ratio = w_ratio / h_ratio
        current_ratio = w / h
        
        if current_ratio > target_ratio:
            # Too wide, crop width
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            img = img.crop((left, 0, left + new_w, h))
        else:
            # Too tall, crop height
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            img = img.crop((0, top, w, top + new_h))
        
        return img
    
    def _create_soft_vignette_mask(self, size) -> Image.Image:
        """
        Creates a smooth, modern radial gradient mask for vignetting.
        Returns an RGB image with dark corners and bright center.
        """
        w, h = size
        # Start with a black image
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw a white ellipse in the middle, slightly larger than the canvas
        padding_x = w * 0.15
        padding_y = h * 0.15
        draw.ellipse((-padding_x, -padding_y, w + padding_x, h + padding_y), fill=255)
        
        # Blur the mask significantly to create a soft gradient
        mask_blurred = mask.filter(ImageFilter.GaussianBlur(radius=w // 4))
        
        # Invert so corners are dark and center is bright
        inverted = ImageOps.invert(mask_blurred)
        
        # Make the vignette subtle (corners at ~80% brightness, not black)
        # Create a base white layer and blend
        white = Image.new('L', (w, h), 255)
        subtle_vignette = Image.blend(white, inverted, 0.25)
        
        return subtle_vignette.convert('RGB')
    
    def _extract_logo_colors(self, img: Image.Image) -> tuple:
        """Extract dominant colors from logo for gradient generation.
        
        Returns: (primary_color, secondary_color) as RGB tuples
        """
        # Resize to speed up processing
        img_small = img.copy()
        img_small.thumbnail((100, 100))
        
        # Get non-transparent pixels
        if img_small.mode == 'RGBA':
            pixels = []
            for x in range(img_small.width):
                for y in range(img_small.height):
                    r, g, b, a = img_small.getpixel((x, y))
                    if a > 128:  # Not transparent
                        pixels.append((r, g, b))
        else:
            pixels = list(img_small.getdata())
        
        if not pixels:
            return None
        
        # Calculate average color
        avg_r = sum(p[0] for p in pixels) // len(pixels)
        avg_g = sum(p[1] for p in pixels) // len(pixels)
        avg_b = sum(p[2] for p in pixels) // len(pixels)
        
        return (avg_r, avg_g, avg_b)
    
    def _generate_complementary_gradient_with_llm(self, logo_colors: tuple) -> tuple:
        """Use LLM to generate complementary gradient colors for logo background.
        
        Uses cascading fallbacks:
        1. Gemini 2.0 Flash (primary)
        2. Qwen 2.5 VL via OpenRouter
        3. Nemotron Nano VL via OpenRouter
        
        Args:
            logo_colors: RGB tuple of logo's dominant color
            
        Returns:
            (top_color, bottom_color) as RGB tuples, or None if failed
        """
        if not logo_colors:
            return None
        
        r, g, b = logo_colors
        
        prompt = f"""You are a professional brand designer. I have a company logo with dominant color RGB({r}, {g}, {b}).

Generate a DARK, professional gradient background (suitable for financial/corporate videos) that complements this logo.

Requirements:
- Dark colors (RGB values mostly under 60) for professional look
- Two colors for vertical gradient (top to bottom)
- Colors should complement the logo, not clash
- Maintain high contrast so logo is visible
- Professional finance aesthetic (think Bloomberg, CNBC)

Respond ONLY with two RGB colors in this exact format:
TOP: (R,G,B)
BOTTOM: (R,G,B)

Example:
TOP: (15,25,35)
BOTTOM: (25,15,30)"""

        # Try Gemini first
        if GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel("gemini-2.0-flash-exp")
                response = model.generate_content(prompt)
                text = response.text.strip()
                
                # Parse response
                import re
                matches = re.findall(r'TOP:\s*\((\d+),\s*(\d+),\s*(\d+)\)\s*BOTTOM:\s*\((\d+),\s*(\d+),\s*(\d+)\)', text)
                if matches:
                    top = (int(matches[0][0]), int(matches[0][1]), int(matches[0][2]))
                    bottom = (int(matches[0][3]), int(matches[0][4]), int(matches[0][5]))
                    logger.info("🎨 Gemini generated complementary gradient: TOP %s, BOTTOM %s", top, bottom)
                    return (top, bottom)
            except Exception as e:
                logger.debug("Gemini gradient generation failed: %s", e)
        
        # Fallback to Qwen VL via OpenRouter
        if OPENROUTER_API_KEY:
            try:
                result = self._generate_gradient_with_openrouter(prompt, QWEN_VL_MODEL)
                if result:
                    logger.info("🎨 Qwen VL generated complementary gradient: TOP %s, BOTTOM %s", result[0], result[1])
                    return result
            except Exception as e:
                logger.debug("Qwen VL gradient generation failed: %s", e)
        
        # Fallback to Nemotron VL via OpenRouter
        if OPENROUTER_API_KEY:
            try:
                result = self._generate_gradient_with_openrouter(prompt, NEMOTRON_VL_MODEL)
                if result:
                    logger.info("🎨 Nemotron VL generated complementary gradient: TOP %s, BOTTOM %s", result[0], result[1])
                    return result
            except Exception as e:
                logger.debug("Nemotron VL gradient generation failed: %s", e)
        
        return None
    
    def _generate_gradient_with_openrouter(self, prompt: str, model: str) -> Optional[tuple]:
        """Call OpenRouter LLM for gradient generation."""
        if not OPENROUTER_API_KEY:
            return None
        
        response = requests.post(
            url=OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            },
            timeout=30,
        )
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Parse response
        import re
        matches = re.findall(r'TOP:\s*\((\d+),\s*(\d+),\s*(\d+)\)\s*BOTTOM:\s*\((\d+),\s*(\d+),\s*(\d+)\)', text)
        if matches:
            top = (int(matches[0][0]), int(matches[0][1]), int(matches[0][2]))
            bottom = (int(matches[0][3]), int(matches[0][4]), int(matches[0][5]))
            return (top, bottom)
        
        return None
    
    def _create_financial_gradient_background(self, size: tuple, logo_img: Image.Image = None) -> Image.Image:
        """Create professional financial gradient background.
        
        Creates a modern gradient that matches the finance/corporate aesthetic:
        - Analyzes logo colors if provided
        - Uses LLM to generate complementary gradient
        - Falls back to default navy/charcoal gradient
        """
        w, h = size
        img = Image.new("RGB", size)
        pixels = img.load()
        
        # Default professional finance color palette
        top_color = (12, 20, 45)  # Dark navy
        bottom_color = (20, 28, 40)  # Charcoal blue
        
        # Try to generate complementary colors based on logo
        if logo_img:
            logo_colors = self._extract_logo_colors(logo_img)
            if logo_colors:
                llm_colors = self._generate_complementary_gradient_with_llm(logo_colors)
                if llm_colors:
                    top_color, bottom_color = llm_colors
        
        # Create vertical gradient
        for y in range(h):
            ratio = y / h
            r = int(top_color[0] + (bottom_color[0] - top_color[0]) * ratio)
            g = int(top_color[1] + (bottom_color[1] - top_color[1]) * ratio)
            b = int(top_color[2] + (bottom_color[2] - top_color[2]) * ratio)
            for x in range(w):
                pixels[x, y] = (r, g, b)
        
        # Add subtle radial lighting effect (spotlight on center)
        overlay = Image.new("RGBA", size, (0, 0, 0, 0))
        overlay_pixels = overlay.load()
        center_x, center_y = w // 2, h // 2
        max_dist = ((w // 2) ** 2 + (h // 2) ** 2) ** 0.5
        
        for y in range(h):
            for x in range(w):
                dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                # Subtle white glow in center (20% max opacity)
                glow = int((1 - (dist / max_dist)) * 51)  # 51 = 20% of 255
                overlay_pixels[x, y] = (255, 255, 255, glow)
        
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        return img
    
    def _process_transparent_png_with_background(self, image_path: Path, index: int) -> Path:
        """Process PNG with transparency by adding professional gradient background.
        
        For company logos and images with transparent backgrounds:
        1. Creates financial gradient background
        2. Scales logo to fit screen (80% of frame width for visibility)
        3. Centers logo on gradient
        4. Saves as JPEG for FFmpeg processing
        """
        output_path = self.temp_images_dir / f"logo_with_bg_{index:03d}.jpg"
        
        try:
            # Open image and check for transparency
            img = Image.open(image_path)
            
            # Check if image has alpha channel (transparency)
            if img.mode not in ('RGBA', 'LA') and 'transparency' not in img.info:
                # No transparency, process normally
                return image_path
            
            # Convert to RGBA if needed
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Create gradient background at target size (with logo color analysis)
            background = self._create_financial_gradient_background((VIDEO_WIDTH, VIDEO_HEIGHT), logo_img=img)
            
            # Scale logo to fit nicely (80% of frame width for good visibility)
            logo_max_width = int(VIDEO_WIDTH * 0.8)
            logo_max_height = int(VIDEO_HEIGHT * 0.8)
            
            # Calculate scaling to fit within bounds while maintaining aspect ratio
            img_w, img_h = img.size
            scale_w = logo_max_width / img_w
            scale_h = logo_max_height / img_h
            scale = min(scale_w, scale_h)
            
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Center logo on background
            x_offset = (VIDEO_WIDTH - new_w) // 2
            y_offset = (VIDEO_HEIGHT - new_h) // 2
            
            # Composite logo onto gradient background
            background_rgba = background.convert('RGBA')
            background_rgba.paste(img_resized, (x_offset, y_offset), img_resized)
            
            # Convert to RGB and save
            final = background_rgba.convert('RGB')
            final.save(output_path, quality=95)
            
            logger.info("✨ Added gradient background to transparent logo: %s", image_path.name)
            return output_path
            
        except Exception as e:
            logger.warning("Failed to process transparent PNG: %s", e)
            return image_path

    def _create_fallback_clip(self, index: int, duration: float) -> MediaClip:
        """
        Create fallback clip by fetching a GUARANTEED finance stock video.
        Uses hardcoded finance queries that always have results on Pexels/Pixabay.
        Falls back to bright gradient only if all fetches fail.
        """
        # Guaranteed finance queries that ALWAYS have stock footage
        guaranteed_queries = [
            "stock market trading floor screens finance",
            "money cash dollars bills stacks finance",
            "business meeting corporate office suits",
            "financial charts graphs computer screen",
            "gold bars coins wealth investment",
            "wall street new york stock exchange",
            "businessman handshake deal corporate",
            "stock ticker numbers scrolling finance",
            "bank vault money safe deposit",
            "cryptocurrency bitcoin trading digital",
            "real estate investment property building",
            "economy growth statistics reports",
        ]
        
        # Use clip index to rotate through queries (ensures variety across clips)
        # Plus shuffle for additional randomness
        shuffled = guaranteed_queries.copy()
        random.shuffle(shuffled)
        
        # Try multiple queries until one works
        for i, query in enumerate(shuffled):
            try:
                clip = self._fetch_stock_video(query, index, duration)
                if clip and clip.path.exists() and clip.path.stat().st_size > 1000:  # At least 1KB
                    # Validate video is actually readable with ffprobe
                    probe_cmd = [
                        "ffprobe", "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(clip.path)
                    ]
                    result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 and result.stdout.strip():
                        probed_duration = float(result.stdout.strip())
                        if probed_duration > 0.5:  # Must have at least 0.5s of video
                            logger.info("Fallback fetched real finance video: %s", query[:30])
                            clip.source = "fallback-stock"
                            return clip
            except Exception as e:
                logger.debug("Fallback query '%s' failed: %s", query[:20], e)
                continue
        
        # Last resort: Create a bright, visible gradient (not dark/black)
        output_path = self.temp_images_dir / f"fallback_{index:03d}.jpg"
        
        # BRIGHT blue gradient (finance-themed) - NOT dark
        img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), (30, 60, 120))
        draw = ImageDraw.Draw(img)
        for y in range(VIDEO_HEIGHT):
            # Gradient from dark blue to lighter blue
            r = int(30 + (y / VIDEO_HEIGHT) * 40)
            g = int(60 + (y / VIDEO_HEIGHT) * 60)
            b = int(120 + (y / VIDEO_HEIGHT) * 80)
            draw.line([(0, y), (VIDEO_WIDTH, y)], fill=(r, g, b))
        
        img.save(output_path, quality=90)
        logger.warning("Using gradient fallback for clip %d (all stock fetches failed)", index)
        
        return MediaClip(
            path=output_path,
            duration=self._safe_duration(duration),
            is_image=True,
            source="fallback-gradient",
        )

    def _financeify_query(self, query: str) -> str:
        """Ensure a finance keyword is present to bias search results."""
        if not query:
            return "finance markets stocks investing"
        finance_terms = ["finance", "markets", "stocks", "equities", "investing", "dividends", "portfolio", "bonds", "cashflow", "valuation"]
        lower = query.lower()
        if any(term in lower for term in finance_terms):
            return query
        return f"{query} finance markets stocks"

    def _safe_duration(self, raw_duration: float) -> float:
        """Clamp duration to a sensible minimum and default."""
        if raw_duration is None or raw_duration != raw_duration:
            return CLIP_DURATION_DEFAULT
        if raw_duration <= 0:
            return CLIP_DURATION_DEFAULT
        return max(raw_duration, MIN_CLIP_DURATION)
    
    # -----------------------------------------------------------------------
    # QUALITY CONTROL: Keyframe Extraction, Blank Detection, Vision Verification
    # -----------------------------------------------------------------------
    
    def _extract_keyframe(self, clip: MediaClip) -> Optional[Image.Image]:
        """Extract a representative frame from a video or load image directly."""
        try:
            if clip.is_image:
                return Image.open(clip.path).convert("RGB")
            else:
                # Extract frame at 1 second (or middle) using FFmpeg
                output_path = self.temp_images_dir / f"keyframe_{clip.path.stem}.jpg"
                seek_time = min(1.0, clip.duration / 2)
                cmd = [
                    "ffmpeg", "-y", "-ss", str(seek_time),
                    "-i", str(clip.path),
                    "-frames:v", "1", "-q:v", "2",
                    str(output_path)
                ]
                subprocess.run(cmd, capture_output=True, timeout=10)
                if output_path.exists():
                    return Image.open(output_path).convert("RGB")
        except Exception as e:
            logger.debug("Failed to extract keyframe: %s", e)
        return None
    
    def _is_frame_blank(self, img: Image.Image, threshold: float = 0.90) -> bool:
        """
        Check if frame is mostly blank/black/white/sky (empty content).
        Returns True if frame is blank and should be replaced.
        More aggressive detection to catch empty/boring frames.
        """
        try:
            # Resize for faster analysis
            small = img.resize((100, 100))
            arr = np.array(small)
            
            # Check 1: Mostly black (avg brightness < 30) - raised from 20
            avg_brightness = np.mean(arr)
            if avg_brightness < 30:
                logger.debug("Blank: too dark (brightness=%.1f)", avg_brightness)
                return True
            
            # Check 2: Mostly white/bright (avg brightness > 230) - lowered from 240
            if avg_brightness > 230:
                logger.debug("Blank: too bright (brightness=%.1f)", avg_brightness)
                return True
            
            # Check 3: Very low variance (solid color) - raised from 100
            variance = np.var(arr)
            if variance < 200:  # More aggressive
                logger.debug("Blank: low variance (%.1f)", variance)
                return True
            
            # Check 4: Sky detection - mostly blue/white upper half
            upper_half = arr[:50, :, :]
            avg_r = np.mean(upper_half[:, :, 0])
            avg_g = np.mean(upper_half[:, :, 1])
            avg_b = np.mean(upper_half[:, :, 2])
            # Sky is typically high blue, high brightness, low red-blue difference
            if avg_b > 180 and avg_brightness > 180 and (avg_b - avg_r) > 20:
                logger.debug("Blank: sky detected (R=%.0f G=%.0f B=%.0f)", avg_r, avg_g, avg_b)
                return True
            
            # Check 5: Single dominant color (>90% of pixels similar) - lowered from 95%
            gray = np.mean(arr, axis=2)
            hist, _ = np.histogram(gray, bins=16, range=(0, 256))
            max_bin_ratio = np.max(hist) / np.sum(hist)
            if max_bin_ratio > threshold:
                logger.debug("Blank: dominant color (ratio=%.2f)", max_bin_ratio)
                return True
            
            return False
        except Exception as e:
            logger.debug("Blank detection failed: %s", e)
            return False
    
    def _contains_blocklisted_content(self, query: str) -> bool:
        """Check if query contains blocklisted terms."""
        query_lower = query.lower()
        return any(term in query_lower for term in IMAGE_BLOCKLIST)
    
    def _verify_clip_with_vision(self, img: Image.Image, topic: str, query: str) -> tuple[bool, str]:
        """
        Use LLM vision to verify clip matches the topic.
        Returns (is_valid, reason).
        Falls back through: Gemini -> Qwen VL -> Nemotron VL
        """
        import base64
        from io import BytesIO
        
        # Convert image to base64
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=70)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        blocklist_str = ", ".join(IMAGE_BLOCKLIST[:10])
        
        prompt = f"""Look at this image and answer with ONLY 'YES' or 'NO' followed by a brief reason.

Is this image appropriate for a FINANCE/BUSINESS video about: {topic}

The image should show: {query}

REJECT if the image shows any of these: {blocklist_str}
REJECT if it's nature, abstract art, religious/spiritual, or off-topic.
ACCEPT if it shows finance, business, money, charts, corporate, or the requested person.

Answer format: YES - [reason] or NO - [reason]"""

        # Try Gemini Vision first
        try:
            result = self._verify_with_gemini_vision(img_base64, prompt)
            if result is not None:
                return result
        except Exception as e:
            logger.debug("Gemini Vision failed: %s", e)
        
        # Fallback to Qwen VL
        try:
            result = self._verify_with_openrouter_vision(img_base64, prompt, QWEN_VL_MODEL)
            if result is not None:
                return result
        except Exception as e:
            logger.debug("Qwen VL failed: %s", e)
        
        # Fallback to Nemotron VL
        try:
            result = self._verify_with_openrouter_vision(img_base64, prompt, NEMOTRON_VL_MODEL)
            if result is not None:
                return result
        except Exception as e:
            logger.debug("Nemotron VL failed: %s", e)
        
        # All vision APIs failed - assume valid to not block pipeline
        logger.warning("All vision APIs failed, assuming clip is valid")
        return (True, "Vision verification unavailable")
    
    def _verify_with_gemini_vision(self, img_base64: str, prompt: str) -> Optional[tuple[bool, str]]:
        """Call Gemini Vision API for image verification."""
        if not GEMINI_API_KEY:
            return None
        
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Gemini accepts PIL Image directly or base64
        import base64
        from io import BytesIO
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_bytes))
        
        response = model.generate_content([prompt, img])
        answer = response.text.strip().upper()
        
        is_valid = answer.startswith("YES")
        reason = response.text.strip()
        
        return (is_valid, reason)
    
    def _verify_with_openrouter_vision(self, img_base64: str, prompt: str, model: str) -> Optional[tuple[bool, str]]:
        """Call OpenRouter Vision API (Qwen/Nemotron) for image verification."""
        if not OPENROUTER_API_KEY:
            return None
        
        response = requests.post(
            url=OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ]
            },
            timeout=30
        )
        
        if response.status_code != 200:
            logger.debug("OpenRouter vision failed: %s", response.text[:100])
            return None
        
        data = response.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        answer_upper = answer.strip().upper()
        
        is_valid = answer_upper.startswith("YES")
        return (is_valid, answer.strip())
    
    def _generate_subtitles(self, segments: List[TranscriptSegment]) -> Optional[Path]:
        """Generate SRT subtitle file from transcript segments."""
        if not segments:
            return None
        
        subtitle_path = self.output_dir / "subtitles.srt"
        
        try:
            with open(subtitle_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, 1):
                    # Format timestamps as HH:MM:SS,mmm
                    start_time = self._format_timestamp(segment.start)
                    end_time = self._format_timestamp(segment.end)
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment.text.strip()}\n\n")
            
            logger.info("Generated subtitles: %s", subtitle_path)
            return subtitle_path
        except Exception as e:
            logger.warning("Failed to generate subtitles: %s", e)
            return None
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _cleanup_temp_files(self):
        """Clean up temporary files after successful completion."""
        try:
            import shutil
            shutil.rmtree(self.output_dir, ignore_errors=True)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning("Failed to cleanup temp files: %s", e)
    
    def _create_pycaps_transcriber_with_hints(self):
        """Create a custom Whisper transcriber with hybrid alignment support.
        
        When spoken_script is available, uses script text for accuracy and Whisper only for timing.
        This fixes mishearings like 'John' → 'Lumentum', 'SIGAT' → 'Seagate'.
        """
        try:
            from pycaps.transcriber import WhisperAudioTranscriber
            from pycaps.common.models import Document, Segment, Line, Word
            from pycaps.common.models import TimeFragment
            
            proper_nouns = self._proper_nouns
            spoken_script = self._spoken_script  # The actual script text for alignment
            
            class WhisperWithHybridAlignment(WhisperAudioTranscriber):
                """Custom Whisper transcriber with hybrid alignment for 100% accurate captions."""
                
                def __init__(self, model_size: str = "base", language=None, hints: List[str] = None, script_text: str = ""):
                    super().__init__(model_size=model_size, language=language)
                    self._hints = hints or []
                    self._script_text = script_text  # Original spoken text for alignment
                
                def transcribe(self, audio_path: str) -> Document:
                    # Build initial_prompt from hints
                    initial_prompt = ""
                    if self._hints:
                        initial_prompt = "Names mentioned: " + ", ".join(self._hints) + ". "
                    
                    # Call Whisper with initial_prompt
                    result = self._get_model().transcribe(
                        audio_path,
                        word_timestamps=True,
                        language=self._language,
                        initial_prompt=initial_prompt if initial_prompt else None,
                        verbose=False
                    )
                    
                    if "segments" not in result or not result["segments"]:
                        return Document()
                    
                    # Collect all whisper words with timing
                    whisper_words = []
                    for segment_info in result["segments"]:
                        if "words" in segment_info and isinstance(segment_info["words"], list):
                            for word_entry in segment_info["words"]:
                                word_text = str(word_entry["word"]).strip()
                                if word_text:
                                    whisper_words.append({
                                        "word": word_text,
                                        "start": float(word_entry["start"]),
                                        "end": float(word_entry["end"])
                                    })
                    
                    # If we have script text, use hybrid alignment
                    if self._script_text and whisper_words:
                        aligned_words = align_script_to_whisper_words(self._script_text, whisper_words)
                        logger.info("🎯 Hybrid alignment: %d whisper words → %d script words", 
                                   len(whisper_words), len(aligned_words))
                    else:
                        # Fallback: use Whisper's words with corrections
                        aligned_words = whisper_words
                        for w in aligned_words:
                            w["word"] = self._correct_word(w["word"])
                    
                    # Build PyCaps Document from aligned words
                    document = Document()
                    
                    # Group words into segments (roughly every 10-15 words or by pauses)
                    words_per_segment = 12
                    for i in range(0, len(aligned_words), words_per_segment):
                        chunk = aligned_words[i:i + words_per_segment]
                        if not chunk:
                            continue
                        
                        segment_start = chunk[0]["start"]
                        segment_end = chunk[-1]["end"]
                        if segment_start == segment_end:
                            segment_end = segment_start + 0.01
                        
                        segment_time = TimeFragment(start=segment_start, end=segment_end)
                        segment = Segment(time=segment_time)
                        line = Line(time=segment_time)
                        segment.lines.add(line)
                        
                        for word_data in chunk:
                            word_start = word_data["start"]
                            word_end = word_data["end"]
                            if word_start == word_end:
                                word_end = word_start + 0.01
                            word_time = TimeFragment(start=word_start, end=word_end)
                            word = Word(text=word_data["word"], time=word_time)
                            line.words.add(word)
                        
                        document.segments.add(segment)
                    
                    return document
                
                def _correct_word(self, word: str) -> str:
                    """Correct common Whisper transcription errors for tickers and decimals."""
                    import re
                    
                    # Ticker corrections (Whisper often mishears these)
                    ticker_corrections = {
                        'ibb': 'IVV', 'IBB': 'IVV',
                        'boo': 'VOO', 'BOO': 'VOO', 'voo': 'VOO',
                        'spy': 'SPY', 'spie': 'SPY', 'SPIE': 'SPY',
                        'vgt': 'VGT', 'VGT': 'VGT',
                        'smh': 'SMH', 'SMH': 'SMH',
                        'qqq': 'QQQ', 'QQQ': 'QQQ',
                        'ivv': 'IVV', 'IVV': 'IVV',
                        'schd': 'SCHD', 'SCHD': 'SCHD',
                        'jepi': 'JEPI', 'JEPI': 'JEPI',
                    }
                    
                    # Check if word (without punctuation) is a ticker
                    clean_word = re.sub(r'[^\w]', '', word.lower())
                    if clean_word in ticker_corrections:
                        prefix = ''
                        suffix = ''
                        if word and not word[0].isalnum():
                            prefix = word[0]
                        if word and not word[-1].isalnum():
                            suffix = word[-1]
                        return prefix + ticker_corrections[clean_word] + suffix
                    
                    # For Spanish: convert decimal points to commas (0.03 → 0,03)
                    if self._language == "es":
                        decimal_match = re.match(r'^(\d+)\.(\d+)(.*)$', word)
                        if decimal_match:
                            return f"{decimal_match.group(1)},{decimal_match.group(2)}{decimal_match.group(3)}"
                    
                    return word
            
            return WhisperWithHybridAlignment(model_size="medium", hints=proper_nouns, script_text=spoken_script)
        except Exception as e:
            logger.warning("Failed to create custom transcriber: %s", e)
            return None
    
    def add_subtitles(self, input_video: Path, output_video: Path):
        """Burn subtitles using PyCaps library or fallback to FFmpeg."""
        logger.info("Step 5/5: Burning subtitles...")
        
        # Remove existing output file if it exists (PyCaps doesn't overwrite)
        if output_video.exists():
            output_video.unlink()
        
        # Try PyCaps library first with "hype" template (TikTok-style animated subtitles)
        try:
            from pycaps import TemplateLoader
            
            logger.info("Using PyCaps 'hype' template for TikTok-style subtitles...")
            # Load template, get builder, set output, build and run
            builder = TemplateLoader("hype").with_input_video(str(input_video)).load(False)
            builder.with_output_video(str(output_video))
            
            # Use custom transcriber with proper noun hints if available
            if self._proper_nouns:
                logger.info("Using proper noun hints for subtitles: %s", self._proper_nouns[:5])
                custom_transcriber = self._create_pycaps_transcriber_with_hints()
                if custom_transcriber:
                    builder.with_custom_audio_transcriber(custom_transcriber)
            
            pipeline = builder.build()
            pipeline.run()
            logger.info("Subtitles added with PyCaps hype template!")
            
            # Add ticker overlays if any tickers were detected
            if any(seg.ticker for seg in self._transcript_segments):
                logger.info("Adding stock ticker overlays above main subtitles...")
                self._add_ticker_overlays(output_video)
            
            return
        except ImportError:
            logger.info("PyCaps library not available, using FFmpeg subtitles...")
        except Exception as e:
            logger.warning("PyCaps failed: %s. Falling back to FFmpeg.", e)
        
        # Fallback: Generate SRT and burn with FFmpeg
        srt_path = self._generate_srt_subtitles()
        if not srt_path:
            logger.warning("No subtitles to burn, copying video as-is")
            shutil.copy(input_video, output_video)
            return
        
        # Escape path for FFmpeg on Windows (replace \ with / and escape :)
        srt_path_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
        
        # Burn subtitles with FFmpeg (high-energy style - BIG for mobile)
        subtitle_style = (
            "FontName=Arial Black,FontSize=42,PrimaryColour=&H00FFFFFF&,"
            "OutlineColour=&H000000&,BorderStyle=1,Outline=4,Shadow=2,"
            "MarginV=180,Alignment=2,Bold=1"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_video),
            "-vf", f"subtitles='{srt_path_escaped}':force_style='{subtitle_style}'",
            "-c:a", "copy",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            str(output_video),
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("Subtitles burned with FFmpeg!")
        except subprocess.CalledProcessError as e:
            logger.error("FFmpeg subtitle burn failed: %s", e.stderr.decode()[:300])
            # Last resort: just copy the video
            shutil.copy(input_video, output_video)
    
    def _add_ticker_overlays(self, video_path: Path):
        """Add stock ticker overlays above main subtitles using FFmpeg drawtext.
        Tickers appear for ~4 seconds when companies are first mentioned.
        """
        # Generate ticker SRT file
        ticker_srt = self._generate_ticker_srt()
        if not ticker_srt:
            return
        
        # Create temporary output file
        temp_output = video_path.with_name(video_path.stem + "_with_tickers.mp4")
        
        # Escape path for FFmpeg
        ticker_srt_escaped = str(ticker_srt).replace("\\", "/").replace(":", "\\:")
        
        # Ticker subtitle style: positioned ABOVE main subtitles
        # MarginV=800 puts it high on screen (main subs at ~180)
        ticker_style = (
            "FontName=Arial Black,FontSize=32,PrimaryColour=&H00FFFF00&,"  # Yellow/gold color
            "OutlineColour=&H000000&,BorderStyle=1,Outline=3,Shadow=2,"
            "MarginV=800,Alignment=2,Bold=1"  # High position, centered
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"subtitles='{ticker_srt_escaped}':force_style='{ticker_style}'",
            "-c:a", "copy",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            str(temp_output),
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            # Replace original with ticker version
            video_path.unlink()
            temp_output.rename(video_path)
            logger.info("✓ Stock tickers added above main subtitles")
        except subprocess.CalledProcessError as e:
            logger.error("Failed to add ticker overlays: %s", e.stderr.decode()[:300])
            if temp_output.exists():
                temp_output.unlink()
    
    def _generate_ticker_srt(self) -> Optional[Path]:
        """Generate SRT file for stock ticker overlays (only segments with tickers)."""
        ticker_segments = [seg for seg in self._transcript_segments if seg.ticker]
        if not ticker_segments:
            return None
        
        ticker_srt_path = self.output_dir / "tickers.srt"
        
        with open(ticker_srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(ticker_segments, 1):
                start_ts = self._format_timestamp(seg.start)
                # Limit ticker display to 4 seconds max
                ticker_end = min(seg.end, seg.start + 4.0)
                end_ts = self._format_timestamp(ticker_end)
                
                f.write(f"{i}\n")
                f.write(f"{start_ts} --> {end_ts}\n")
                f.write(f"${seg.ticker}\n\n")
        
        logger.info("Generated ticker SRT: %s (%d tickers)", ticker_srt_path, len(ticker_segments))
        return ticker_srt_path
    
    def _generate_srt_subtitles(self) -> Optional[Path]:
        """Generate SRT file from transcript segments."""
        if not self._transcript_segments:
            return None
        
        srt_path = self.output_dir / "subtitles.srt"
        
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(self._transcript_segments, 1):
                start_ts = self._format_timestamp(seg.start)
                end_ts = self._format_timestamp(seg.end)
                # Split long text into chunks for better readability
                text = seg.text.strip()
                if len(text) > 50:
                    # Insert line break at middle word
                    words = text.split()
                    mid = len(words) // 2
                    text = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
                
                f.write(f"{i}\n")
                f.write(f"{start_ts} --> {end_ts}\n")
                f.write(f"{text}\n\n")
        
        logger.info("Generated SRT: %s", srt_path)
        return srt_path
    
    # -----------------------------------------------------------------------
    # HANDS: Video Assembly with FFmpeg
    # -----------------------------------------------------------------------
    
    def _validate_timeline(self, clips: List[MediaClip]) -> List[MediaClip]:
        """
        Timeline Validation - like CapCut/Premiere timeline view.
        Checks each clip:
        1. File exists and readable
        2. Keyframe not blank/black/white
        3. Content matches topic (LLM vision verification)
        Replaces invalid clips with finance-focused fallbacks.
        Includes: replacement cap, keyframe cache, parallel vision, JSON log.
        """
        validated_clips = []
        total_duration = 0.0
        replaced_count = 0
        replaced_blank = 0
        replaced_file = 0
        replaced_corrupt = 0
        replaced_offtopic = 0
        keyframe_cache: dict[str, Image.Image] = {}
        timeline_log = []
        vision_candidates = []  # tuples of (index, clip, keyframe, query)
        vision_results = {}
        replacements_capped = False
        
        logger.info("=" * 60)
        logger.info("TIMELINE VALIDATION - Checking %d clips", len(clips))
        logger.info("=" * 60)
        
        # First pass: structural checks + blank detection, collect vision tasks
        for i, clip in enumerate(clips):
            clip_status = "✓"
            issue = None
            needs_replacement = False
            
            clip_query = ""
            if i < len(self._visual_queries):
                clip_query = self._visual_queries[i].query
            
            # Check 1: Does the file exist?
            if not clip.path.exists():
                clip_status = "✗"
                issue = "FILE MISSING"
                needs_replacement = True
                replaced_file += 1
            
            # Check 2: Is file size > 0?
            elif clip.path.stat().st_size == 0:
                clip_status = "✗"
                issue = "EMPTY FILE (0 bytes)"
                needs_replacement = True
                replaced_file += 1
            
            # Check 3: For images, verify it's a valid image
            elif clip.is_image:
                try:
                    with Image.open(clip.path) as img:
                        img.verify()
                except Exception as e:
                    clip_status = "✗"
                    issue = f"CORRUPT IMAGE: {str(e)[:30]}"
                    needs_replacement = True
                    replaced_corrupt += 1
            
            # Check 4: For videos, probe with FFmpeg
            elif not clip.is_image:
                probe_cmd = [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(clip.path)
                ]
                try:
                    result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode != 0 or not result.stdout.strip():
                        clip_status = "✗"
                        issue = "UNREADABLE VIDEO"
                        needs_replacement = True
                        replaced_corrupt += 1
                except Exception as e:
                    clip_status = "✗"
                    issue = f"PROBE FAILED: {str(e)[:20]}"
                    needs_replacement = True
                    replaced_corrupt += 1
            
            # Check 5: Keyframe blank detection (skip if already needs replacement or if company logo/ceo)
            keyframe = None
            if not needs_replacement and clip.source not in ["fallback", "company_logo", "ceo_folder"]:
                try:
                    if str(clip.path) in keyframe_cache:
                        keyframe = keyframe_cache[str(clip.path)]
                    else:
                        keyframe = self._extract_keyframe(clip)
                        if keyframe:
                            keyframe_cache[str(clip.path)] = keyframe
                    if keyframe and self._is_frame_blank(keyframe):
                        clip_status = "✗"
                        issue = "BLANK/EMPTY FRAME"
                        needs_replacement = True
                        replaced_blank += 1
                except Exception as e:
                    logger.debug("Keyframe check failed: %s", e)
            
            # Collect vision candidates (only if still good and not fallback and topic exists)
            # SKIP vision validation for CEO/company images from folder (already validated by entity extraction)
            if (not needs_replacement
                and clip.source not in ["fallback", "ceo_folder", "company_logo"]
                and self._video_topic):
                if keyframe is None:
                    try:
                        if str(clip.path) in keyframe_cache:
                            keyframe = keyframe_cache[str(clip.path)]
                        else:
                            keyframe = self._extract_keyframe(clip)
                            if keyframe:
                                keyframe_cache[str(clip.path)] = keyframe
                    except Exception as e:
                        logger.debug("Keyframe fetch for vision failed: %s", e)
                if keyframe is not None:
                    vision_candidates.append((i, clip, keyframe, clip_query))
            
            # Log entry now; we may update later after vision
            start_time = total_duration
            end_time = total_duration + clip.duration
            source_tag = f"[{clip.source}]" if clip.source else "[unknown]"
            timeline_log.append({
                "index": i,
                "start": start_time,
                "end": end_time,
                "source": clip.source,
                "path": str(clip.path),
                "status": clip_status,
                "issue": issue,
                "action": "replace" if needs_replacement else "keep",
            })
            
            total_duration = end_time
            
        # Second pass: vision verification in parallel (respect replacement cap)
        remaining_replacements = MAX_REPLACEMENTS - replaced_count
        if remaining_replacements > 0 and vision_candidates:
            max_workers = min(VISION_MAX_WORKERS, len(vision_candidates))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {}
                for idx, (i, clip, keyframe, clip_query) in enumerate(vision_candidates):
                    future = executor.submit(self._verify_clip_with_vision, keyframe, self._video_topic, clip_query)
                    future_to_idx[future] = (i, clip, clip_query)
                for future in concurrent.futures.as_completed(future_to_idx):
                    i, clip, clip_query = future_to_idx[future]
                    try:
                        is_valid, reason = future.result()
                        vision_results[i] = (is_valid, reason)
                    except Exception as e:
                        logger.debug("Vision future failed: %s", e)
                        vision_results[i] = (True, "vision error; keep")
        else:
            replacements_capped = replaced_count >= MAX_REPLACEMENTS
            if replacements_capped:
                logger.warning("Replacement cap reached (%d); skipping vision checks", MAX_REPLACEMENTS)
        
        # Final assembly: apply replacements, including vision outcomes, with cap
        total_duration = 0.0
        validated_clips.clear()
        final_log = []
        replaced_count = 0
        for entry, clip in zip(timeline_log, clips):
            i = entry["index"]
            needs_replacement = entry["action"] == "replace"
            issue = entry["issue"]
            clip_status = entry["status"]
            clip_query = ""
            if i < len(self._visual_queries):
                clip_query = self._visual_queries[i].query
            
            # Apply vision result if available and cap not exceeded
            if not needs_replacement and i in vision_results and replaced_count < MAX_REPLACEMENTS:
                is_valid, reason = vision_results[i]
                if not is_valid:
                    needs_replacement = True
                    issue = f"OFF-TOPIC: {reason[:60]}"
                    clip_status = "⚠"
                    replaced_offtopic += 1
            
            # Enforce replacement cap
            if needs_replacement and replaced_count >= MAX_REPLACEMENTS:
                needs_replacement = False
                issue = None
                clip_status = "✓" if clip_status != "⚠" else clip_status
            
            start_time = total_duration
            end_time = total_duration + clip.duration
            source_tag = f"[{clip.source}]" if clip.source else "[unknown]"
            
            if needs_replacement:
                fallback = self._create_fallback_clip(i, clip.duration)
                validated_clips.append(fallback)
                replaced_count += 1
                logger.warning(
                    "[%s] Clip %02d | %6.2fs - %6.2fs | %s | %s → REPLACED",
                    clip_status, i, start_time, end_time, source_tag, issue
                )
                final_log.append({**entry, "action": "replace", "issue": issue, "status": clip_status})
            else:
                validated_clips.append(clip)
                logger.info(
                    "[%s] Clip %02d | %6.2fs - %6.2fs | %s | %s",
                    clip_status, i, start_time, end_time, source_tag, clip.path.name[:30]
                )
                final_log.append({**entry, "action": "keep", "issue": issue, "status": clip_status})
            
            total_duration = end_time
        
        logger.info("=" * 60)
        logger.info(
            "TIMELINE: %.2fs total | %d clips | replaced=%d (file:%d blank:%d corrupt:%d offtopic:%d)",
            total_duration, len(validated_clips), replaced_count,
            replaced_file, replaced_blank, replaced_corrupt, replaced_offtopic
        )
        if replacements_capped:
            logger.info("Replacement cap hit (%d); some clips kept without vision check", MAX_REPLACEMENTS)
        logger.info("=" * 60)
        
        # Write JSON log for debugging/review
        try:
            log_path = self.output_dir / "timeline_validation.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(final_log, f, ensure_ascii=False, indent=2)
            logger.info("Timeline validation log saved: %s", log_path)
        except Exception as e:
            logger.debug("Failed to write validation log: %s", e)
        
        return validated_clips
    
    def assemble_video(self, clips: List[MediaClip], audio_path: Path, output_path: Path) -> Path:
        """Assemble final video using FFmpeg."""
        logger.info("Assembling video with %d clips", len(clips))
        
        # TIMELINE VALIDATION - Check for blanks/gaps before assembly
        clips = self._validate_timeline(clips)
        
        # Create clip list file for FFmpeg concat
        clips_list_path = self.output_dir / "clips_list.txt"
        processed_clips = []
        
        for i, clip in enumerate(clips):
            # Process each clip to ensure correct format
            processed_path = self._process_clip(clip, i)
            processed_clips.append(processed_path)
            
        # Write concat file
        with open(clips_list_path, "w") as f:
            for clip_path in processed_clips:
                f.write(f"file '{clip_path}'\n")
        
        # Concat all clips (hard cuts, faster rendering)
        concat_video = self.output_dir / "concat_video.mp4"
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(clips_list_path),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(concat_video),
        ]
        
        logger.info("Concatenating clips (hard cuts)...")
        try:
            result = subprocess.run(concat_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error("FFmpeg concat failed: %s", e.stderr)
            raise RuntimeError(f"Video concatenation failed: {e.stderr}")
        
        # Add audio + watermark logo in one pass (efficient)
        # Industry standard: Small transparent logo throughout video (subconscious branding)
        
        # Calculate watermark position based on config
        # For vertical videos (1080x1920), position logo to avoid UI elements
        if WATERMARK_POSITION == "top-left":
            overlay_pos = f"{WATERMARK_PADDING}:{WATERMARK_PADDING}"
        elif WATERMARK_POSITION == "top-right":
            overlay_pos = f"W-w-{WATERMARK_PADDING}:{WATERMARK_PADDING}"
        elif WATERMARK_POSITION == "top-right-safe":
            # 15% from top to avoid search bar/UI (1920 * 0.15 = 288px)
            safe_y = int(VIDEO_HEIGHT * 0.15)
            overlay_pos = f"W-w-{WATERMARK_PADDING}:{safe_y}"
        elif WATERMARK_POSITION == "bottom-left":
            overlay_pos = f"{WATERMARK_PADDING}:H-h-{WATERMARK_PADDING}"
        else:  # bottom-right (default)
            overlay_pos = f"W-w-{WATERMARK_PADDING}:H-h-{WATERMARK_PADDING}"
        
        # Check if logo file exists
        logo_path = Path(WATERMARK_LOGO_PATH)
        if not logo_path.exists():
            logger.warning("Watermark logo not found: %s - skipping watermark", logo_path)
            # Fallback: no watermark
            final_cmd = [
                "ffmpeg", "-y",
                "-i", str(concat_video),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                str(output_path),
            ]
        else:
            # Build FFmpeg command with watermark overlay
            # Format: scale logo, set opacity, overlay at position
            watermark_filter = (
                f"[1:v]scale={WATERMARK_SIZE}:{WATERMARK_SIZE}:force_original_aspect_ratio=decrease,"
                f"format=rgba,colorchannelmixer=aa={WATERMARK_OPACITY}[logo];"
                f"[0:v][logo]overlay={overlay_pos}"
            )
            
            final_cmd = [
                "ffmpeg", "-y",
                "-i", str(concat_video),
                "-i", str(logo_path),
                "-i", str(audio_path),
                "-filter_complex", watermark_filter,
                "-map", "0:a?",  # Keep original audio if any
                "-map", "2:a",   # Add voiceover audio
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                str(output_path),
            ]
            logger.info("Adding audio + watermark logo (60%% opacity, %s)", WATERMARK_POSITION)
        
        try:
            result = subprocess.run(final_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error("FFmpeg final assembly failed: %s", e.stderr)
            raise RuntimeError(f"Video assembly failed: {e.stderr}")
        
        # POST-ASSEMBLY: Detect black frames and warn
        black_frames = self._detect_black_frames(output_path)
        if black_frames:
            logger.warning("=" * 60)
            logger.warning("BLACK FRAMES DETECTED in final video!")
            for bf in black_frames[:5]:  # Show first 5
                logger.warning("  Black frame at %.2fs - %.2fs", bf['start'], bf['end'])
            if len(black_frames) > 5:
                logger.warning("  ... and %d more", len(black_frames) - 5)
            logger.warning("=" * 60)
        else:
            logger.info("✓ No black frames detected in final video")
        
        logger.info("Video assembled: %s", output_path)
        return output_path
    
    def _detect_black_frames(self, video_path: Path) -> list:
        """
        Use FFmpeg blackdetect filter to find black frames in the video.
        Returns list of {'start': float, 'end': float, 'duration': float}
        """
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"blackdetect=d={BLACK_FRAME_THRESHOLD}:pix_th=0.1",
            "-f", "null", "-"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            stderr = result.stderr
            
            # Parse blackdetect output
            # Format: [blackdetect @ ...] black_start:0.0 black_end:1.5 black_duration:1.5
            black_frames = []
            import re
            pattern = r"black_start:(\d+\.?\d*)\s+black_end:(\d+\.?\d*)\s+black_duration:(\d+\.?\d*)"
            for match in re.finditer(pattern, stderr):
                black_frames.append({
                    'start': float(match.group(1)),
                    'end': float(match.group(2)),
                    'duration': float(match.group(3))
                })
            
            return black_frames
        except Exception as e:
            logger.debug("Black frame detection failed: %s", e)
            return []
    
    def _find_clips_with_black_frames(self, clips: List[MediaClip], black_frames: list) -> List[int]:
        """
        Given black frame timestamps and clip list, find which clip indices have black frames.
        Returns list of clip indices that need to be re-fetched.
        """
        bad_clip_indices = set()
        
        # Build timeline: clip index -> (start_time, end_time)
        timeline = []
        current_time = 0.0
        for i, clip in enumerate(clips):
            start = current_time
            end = current_time + clip.duration
            timeline.append((i, start, end))
            current_time = end
        
        # Find which clips contain black frames
        for bf in black_frames:
            bf_mid = (bf['start'] + bf['end']) / 2  # Middle of black region
            for i, start, end in timeline:
                if start <= bf_mid <= end:
                    bad_clip_indices.add(i)
                    break
        
        return sorted(list(bad_clip_indices))
    
    def _refetch_clips(self, clips: List[MediaClip], bad_indices: List[int]) -> List[MediaClip]:
        """
        Re-fetch stock videos for the specified clip indices.
        Uses guaranteed finance queries to ensure good results.
        """
        guaranteed_queries = [
            "stock market trading floor screens finance",
            "money cash dollars bills stacks finance",
            "business meeting corporate office suits",
            "financial charts graphs computer screen",
            "gold bars coins wealth investment",
            "wall street new york stock exchange",
            "businessman handshake deal corporate",
            "stock ticker numbers scrolling finance",
        ]
        
        new_clips = list(clips)  # Copy
        
        for idx in bad_indices:
            if idx >= len(new_clips):
                continue
            
            old_clip = new_clips[idx]
            duration = old_clip.duration
            
            # Try multiple guaranteed queries until we get a good one
            for attempt, query in enumerate(random.sample(guaranteed_queries, min(3, len(guaranteed_queries)))):
                logger.info("Re-fetching clip %d with query: %s", idx, query[:30])
                try:
                    new_clip = self._fetch_stock_video(query, idx + 1000, duration)  # Use different index
                    if new_clip and new_clip.path.exists() and new_clip.path.stat().st_size > 1000:
                        # Verify it's not blank
                        keyframe = self._extract_keyframe(new_clip)
                        if keyframe and not self._is_frame_blank(keyframe):
                            new_clip.source = f"refetch-{attempt+1}"
                            new_clips[idx] = new_clip
                            logger.info("Successfully re-fetched clip %d", idx)
                            break
                except Exception as e:
                    logger.debug("Re-fetch attempt %d failed: %s", attempt, e)
            else:
                # All attempts failed - create bright gradient fallback
                logger.warning("All re-fetch attempts failed for clip %d, using gradient", idx)
                new_clips[idx] = self._create_fallback_clip(idx, duration)
        
        return new_clips
    
    def _get_random_ken_burns_filter(self, duration: float) -> str:
        """
        Generates HIGH ENERGY camera moves for TikTok/Reels retention.
        Includes Crash Zooms, Whip Pans, and Diagonals.
        """
        # Duration in frames (approximate)
        d = int(duration * VIDEO_FPS)
        
        # We need a variety of speeds. 
        # 0.002 = Slow/Cinematic
        # 0.005 = Fast/Snappy
        
        moves = [
            # 1. THE CRASH ZOOM (Fast Zoom In to Center)
            # Starts at 1.0, zooms in fast (0.004/frame) to focus on the subject.
            (f"zoompan=z='min(zoom+0.004,1.5)':d={d}:"
             f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS}"),

            # 2. THE REVEAL (Zoom Out from Center)
            # Starts zoomed in at 1.5, pulls back to reveal the full image.
            (f"zoompan=z='1.5-on/{d}*0.5':d={d}:"
             f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS}"),

            # 3. THE WHIP PAN RIGHT (Slide Right)
            # Zooms in slightly (1.2) to allow movement, then slides X axis.
            (f"zoompan=z=1.3:d={d}:"
             f"x='(1-on/{d})*(iw-iw/zoom)':y='(ih-ih/zoom)/2':s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS}"),

            # 4. THE WHIP PAN LEFT (Slide Left)
            (f"zoompan=z=1.3:d={d}:"
             f"x='(on/{d})*(iw-iw/zoom)':y='(ih-ih/zoom)/2':s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS}"),

            # 5. THE DIAGONAL RISE (Bottom-Left to Top-Right)
            # Great for graphs or skyscrapers.
            (f"zoompan=z=1.4:d={d}:"
             f"x='(on/{d})*(iw-iw/zoom)':y='(1-on/{d})*(ih-ih/zoom)':s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS}"),

            # 6. THE FOCUS DRIFT (Slow zoom + subtle slide)
            # The "Documentary" look.
            (f"zoompan=z='min(zoom+0.0015,1.2)':d={d}:"
             f"x='iw/2-(iw/zoom/2)+((on/{d})-0.5)*200':y='ih/2-(ih/zoom/2)':s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS}")
        ]
        
        return random.choice(moves)
    
    def _process_clip(self, clip: MediaClip, index: int) -> Path:
        """Process clip: applies Zoom to images and enforces 9:16."""
        output_path = self.temp_clips_dir / f"processed_{index:03d}.mp4"
        duration = self._safe_duration(clip.duration)
        
        # Standard filter for vertical crop (9:16)
        # This centers the crop so we don't distort the video
        crop_filter = f"scale=-1:1920,crop=1080:1920:(iw-1080)/2:0,setsar=1"

        if clip.is_image:
            # Check if image is transparent PNG and add gradient background
            processed_image_path = self._process_transparent_png_with_background(clip.path, index)
            
            # HIGH ENERGY camera moves for TikTok/Reels retention
            # Includes Crash Zooms, Whip Pans, and Diagonals
            zoom_filter = (
                f"{self._get_random_ken_burns_filter(duration)},"
                f"scale=1080:1920,setsar=1"
            )
            
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", str(processed_image_path),  # Use processed image with background if needed
                "-c:v", "libx264",
                "-t", str(duration),
                "-pix_fmt", "yuv420p",
                "-vf", zoom_filter,  # <--- Apply Zoom here
                "-r", str(VIDEO_FPS),
                str(output_path),
            ]
        else:
            # Video: Just crop and cut
            cmd = [
                "ffmpeg", "-y",
                "-i", str(clip.path),
                "-c:v", "libx264",
                "-t", str(duration),
                "-pix_fmt", "yuv420p",
                "-vf", crop_filter, # <--- Apply Center Crop here
                "-r", str(VIDEO_FPS),
                "-an",
                str(output_path),
            ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            # Show LAST 500 chars of stderr (actual error, not version banner)
            stderr_text = e.stderr.decode(errors='ignore')
            logger.error("FFmpeg failed for clip %d: ...%s", index, stderr_text[-500:])
            
            # Track retry attempts per clip to prevent infinite loops
            retry_key = f"clip_{index}_retries"
            retries = getattr(self, '_clip_retries', {})
            current_retries = retries.get(retry_key, 0)
            
            # If not already a fallback source and under retry limit, try fallback
            if clip.source not in ("fallback-stock", "fallback-gradient") and current_retries < 3:
                retries[retry_key] = current_retries + 1
                self._clip_retries = retries
                fallback = self._create_fallback_clip(index, duration)
                return self._process_clip(fallback, index)
            
            # If retries exhausted or already fallback, use GUARANTEED gradient (simple image)
            if current_retries >= 3 or clip.source in ("fallback-stock", "fallback-gradient"):
                logger.warning("Using gradient fallback for clip %d after FFmpeg failures", index)
                # Create ultra-simple gradient fallback that can't fail
                gradient_path = self.temp_images_dir / f"emergency_fallback_{index:03d}.jpg"
                from PIL import Image, ImageDraw
                img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), (30, 60, 120))
                draw = ImageDraw.Draw(img)
                for y in range(VIDEO_HEIGHT):
                    r = int(30 + (y / VIDEO_HEIGHT) * 40)
                    g = int(60 + (y / VIDEO_HEIGHT) * 60) 
                    b = int(120 + (y / VIDEO_HEIGHT) * 80)
                    draw.line([(0, y), (VIDEO_WIDTH, y)], fill=(r, g, b))
                img.save(gradient_path, quality=90)
                
                # Create video from image with simple command
                gradient_cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1", "-i", str(gradient_path),
                    "-c:v", "libx264", "-t", str(duration),
                    "-pix_fmt", "yuv420p", "-vf", "scale=1080:1920",
                    "-r", str(VIDEO_FPS),
                    str(output_path),
                ]
                subprocess.run(gradient_cmd, check=True, capture_output=True)
                return output_path
            
            raise e
    
    # -----------------------------------------------------------------------
    # Main Pipeline
    # -----------------------------------------------------------------------
    
    def generate_video(self, audio_path: Path, output_path: Path, whisper_model: str = "base", skip_subtitles: bool = False, language: Optional[str] = None, proper_nouns: Optional[List[str]] = None, original_script: Optional[str] = None, spoken_script: Optional[str] = None) -> Path:
        """Run the full video generation pipeline.
        
        Args:
            audio_path: Path to audio file
            output_path: Path to output video
            whisper_model: Whisper model size
            skip_subtitles: Skip subtitle generation
            language: Language code (e.g., 'en', 'es')
            proper_nouns: List of proper nouns/names for correct subtitle spelling
                         (e.g., ['Greg Abel', 'Warren Buffett', 'Berkshire Hathaway'])
            original_script: Original script text (before audio generation) for entity extraction.
                            This helps correct name spelling in visual queries.
            spoken_script: The actual spoken text (TTS input) for hybrid caption alignment.
                          When provided, captions use this text with Whisper timing for 100% accuracy.
        """
        logger.info("=" * 60)
        logger.info("AUTOVIDEO BOT - Starting pipeline")
        logger.info("=" * 60)
        logger.info("Audio: %s", audio_path)
        logger.info("Output: %s", output_path)
        
        # Step 0: Extract entities from original script (if provided)
        if original_script:
            self.extract_entities_from_script(original_script)
        
        # Store spoken script for hybrid alignment (script text + whisper timing)
        if spoken_script:
            self._spoken_script = spoken_script
            logger.info("Using hybrid alignment: script text (%d words) + Whisper timing", len(spoken_script.split()))
        
        # Store proper nouns for subtitle generation
        if proper_nouns:
            self._proper_nouns = proper_nouns
            logger.info("Proper nouns for subtitles: %s", proper_nouns[:10])
        
        # Step 1: Transcribe
        logger.info("Step 1/5: Transcribing audio...")
        segments = self.transcribe_audio(audio_path, model_size=whisper_model, language=language)
        self._transcript_segments = segments  # Store for subtitles
        
        # Detect company mentions and assign tickers (once per company)
        self._detect_and_assign_tickers(segments)
        
        # Extract topic summary from first few segments
        topic_text = " ".join([s.text for s in segments[:5]])
        self._video_topic = topic_text[:200]  # First 200 chars as topic context
        
        # Step 2: Generate visual queries
        logger.info("Step 2/5: Generating visual queries...")
        queries = self.generate_visual_queries(segments)
        self._visual_queries = queries  # Store for clip verification
        
        # Step 3: Fetch media
        logger.info("Step 3/5: Fetching stock media...")
        clips = self.fetch_media_for_queries(queries)
        
        # Step 3.5: Ensure clips cover full audio duration
        # Whisper may not transcribe silence at the end, so clips might be shorter than audio
        self._audio_duration = self._get_audio_duration(audio_path)
        if self._audio_duration > 0 and clips:
            total_clips_duration = sum(c.duration for c in clips)
            if self._audio_duration > total_clips_duration + 0.1:  # 0.1s tolerance
                gap = self._audio_duration - total_clips_duration
                logger.info("⚠️ Audio (%.2fs) > Clips (%.2fs) - extending last clip by %.2fs",
                           self._audio_duration, total_clips_duration, gap)
                # Extend the last clip to cover the gap
                clips[-1] = MediaClip(
                    path=clips[-1].path,
                    duration=clips[-1].duration + gap + 0.5,  # +0.5s safety margin
                    is_image=clips[-1].is_image,
                    source=clips[-1].source,
                )
                logger.info("✓ Last clip extended to %.2fs", clips[-1].duration)
        
        # Step 4: Assemble video with BLACK FRAME RETRY LOOP
        logger.info("Step 4/5: Assembling video...")
        temp_video = self.output_dir / "temp_video_no_subs.mp4"

        # Retry until NO black frames (unlimited retries - we GUARANTEE clean video)
        # Track per-clip attempts so we can escalate to bright gradient if a clip fails repeatedly.
        retry_attempt = 0
        bad_attempts: Dict[int, int] = defaultdict(int)
        while True:
            # Assemble video
            temp_video = self.assemble_video(clips, audio_path, temp_video)
            
            # Check for black frames
            black_frames = self._detect_black_frames(temp_video)
            
            if not black_frames:
                if retry_attempt == 0:
                    logger.info("✅ No black frames detected - video is clean!")
                else:
                    logger.info("✅ No black frames after %d retries - video is now clean!", retry_attempt)
                break
            
            # Found black frames
            total_black = sum(bf['duration'] for bf in black_frames)
            logger.warning(
                "⚠️ Found %d black frame regions (%.2fs total) at: %s",
                len(black_frames),
                total_black,
                ", ".join([f"{bf['start']:.1f}-{bf['end']:.1f}s" for bf in black_frames])
            )
            
            # Find which clips caused black frames
            bad_clip_indices = self._find_clips_with_black_frames(clips, black_frames)
            
            if not bad_clip_indices:
                logger.warning("Could not identify bad clips by timestamp - proceeding with current video")
                break
            
            retry_attempt += 1
            logger.info(
                "🔄 Retry %d: Re-fetching %d bad clips (indices: %s) - good clips stay intact",
                retry_attempt,
                len(bad_clip_indices), bad_clip_indices
            )

            # Escalation: if a clip failed 3 times, force gradient fallback to break any loop.
            clips_escalated = []
            for idx in bad_clip_indices:
                bad_attempts[idx] += 1
                if bad_attempts[idx] >= 3:
                    # Persistent offender: use bright gradient fallback to guarantee non-black frames.
                    duration = clips[idx].duration
                    clips[idx] = self._create_fallback_clip(idx, duration)
                    clips_escalated.append(idx)
            if clips_escalated:
                logger.info("Escalated %d clips to gradient fallback (indices: %s)", len(clips_escalated), clips_escalated)

            # Re-fetch ONLY remaining bad clips with new stock videos (good clips untouched)
            clips = self._refetch_clips(clips, bad_clip_indices)
            
            # Re-validate new clips (quick check)
            clips = self._validate_timeline(clips)
            
            logger.info("Rebuilding video with %d good clips + %d new clips...", 
                       len(clips) - len(bad_clip_indices), len(bad_clip_indices))
        
        # Step 5: Add subtitles (optional)
        if skip_subtitles:
            logger.info("Step 5/5: Skipping subtitles (--no-subtitles flag)")
            shutil.copy(temp_video, output_path)
        else:
            logger.info("Step 5/5: Adding subtitles...")
            self.add_subtitles(temp_video, output_path)
        
        # Cleanup temp files (unless keep_temp is True)
        if not self.keep_temp:
            self._cleanup_temp_files()
        
        # Save video cache to disk (persist for future videos)
        _save_video_cache(self._video_cache)
        
        logger.info("=" * 60)
        logger.info("VIDEO GENERATION COMPLETE")
        logger.info("Output: %s", output_path)
        if self.keep_temp:
            logger.info("Temp files kept in: %s", self.output_dir)
        logger.info("=" * 60)
        
        return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AutoVideoBot - Generate faceless vertical videos from audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to audio file (WAV, MP3, etc.)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output video path. Default: audio_name_video.mp4",
    )
    parser.add_argument(
        "--whisper-model", "-w",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Language code for transcription (e.g., en, es). Default: auto-detect",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for temp files",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files after completion (for debugging)",
    )
    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Skip subtitle burning step",
    )
    
    args = parser.parse_args()
    
    if not args.audio.exists():
        raise SystemExit(f"Audio file not found: {args.audio}")
    
    output_path = args.output or args.audio.with_suffix(".mp4").with_name(args.audio.stem + "_video.mp4")
    
    bot = AutoVideoBot(output_dir=args.work_dir, keep_temp=args.keep_temp)
    bot.generate_video(
        args.audio,
        output_path,
        whisper_model=args.whisper_model,
        skip_subtitles=args.no_subtitles,
        language=args.lang,
    )


if __name__ == "__main__":
    main()
