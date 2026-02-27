#!/usr/bin/env python3
"""
ShortsVideoBot - TikTok/Reels/Shorts Generator with AI Script + Voice Clone

End-to-end pipeline:
1. RESEARCH: Web scrape via DuckDuckGo/Brave (no APIs needed)
2. SCRIPT: Gemini/Xiaomi generates 90-second scripts (200-225 words) in EN + ES
3. VOICE: MiniMax TTS with your voice clone generates audio
4. VIDEO: auto_video_bot pipeline creates faceless vertical video

Outputs: 2 scripts, 2 voiceovers, 2 videos (English + LATAM Spanish)

Usage:
    python shorts_video_bot.py "Why Bitcoin is crashing in 2025" --output "bitcoin_short.mp4"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Import our existing video bot
from auto_video_bot import AutoVideoBot

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

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_VOICE_ID = os.getenv("MINIMAX_CLONE_VOICE_ID")
WEBSHARE_PROXY_URL = os.getenv("WEBSHARE_PROXY_URL")

# Models
GEMINI_MODEL = "gemini-2.5-flash"
XIAOMI_MODEL = "xiaomi/mimo-v2-flash:free"
GLM_MODEL = "zhipuai/glm-4-plus:free"
OLMO_MODEL = "allenai/olmo-3.1-32b-think:free"  # Reasoning fallback when Gemini rate limited

# Endpoints
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MINIMAX_TTS_URL = "https://api.minimax.io/v1/t2a_v2"

# Proxy configuration
# Webshare proxy IPs for DNS fallback (if p.webshare.io doesn't resolve)
WEBSHARE_PROXY_IPS = [
    "51.77.20.223", "141.95.202.227", "141.94.162.15", "54.38.13.221",
    "135.125.3.89", "141.95.173.161", "141.95.157.159", "54.38.13.176",
]

def _fix_proxy_dns(proxy_url: str) -> str:
    """Just return the proxy URL as-is - DNS fallback disabled since IP auth configured."""
    return proxy_url if proxy_url else ""

def _build_us_proxy_url(proxy_url: str) -> str:
    """Convert proxy URL to US-specific."""
    if not proxy_url:
        return ""
    parsed = urlparse(proxy_url)
    if parsed.username and "-rotate" in parsed.username:
        us_username = parsed.username.replace("-rotate", "-US-rotate")
        netloc = f"{us_username}:{parsed.password}@{parsed.hostname}:{parsed.port}"
        return f"{parsed.scheme}://{netloc}"
    return proxy_url

# Fix proxy URL if DNS blocked
WEBSHARE_PROXY_URL_FIXED = _fix_proxy_dns(WEBSHARE_PROXY_URL) if WEBSHARE_PROXY_URL else ""
WEBSHARE_PROXY_URL_US = _build_us_proxy_url(WEBSHARE_PROXY_URL_FIXED) if WEBSHARE_PROXY_URL_FIXED else ""
REQUESTS_PROXY = {"http": WEBSHARE_PROXY_URL_FIXED, "https": WEBSHARE_PROXY_URL_FIXED} if WEBSHARE_PROXY_URL_FIXED else {}
REQUESTS_PROXY_US = {"http": WEBSHARE_PROXY_URL_US, "https": WEBSHARE_PROXY_URL_US} if WEBSHARE_PROXY_URL_US else {}

# Request settings
REQUEST_TIMEOUT = 120
MAX_LLM_RETRIES = 2
MIN_SCRIPT_WORDS = 180
MAX_SCRIPT_WORDS = 250

# Domains to filter from search results
SEARCH_FILTER_DOMAINS = [
    "youtube.com", "youtu.be", "google.com", "bing.com", "duckduckgo.com",
    "brave.com", "reddit.com", "facebook.com", "twitter.com", "x.com",
]

# ---------------------------------------------------------------------------
# Master Prompts for Short-Form Fact-Check Videos
# ---------------------------------------------------------------------------

# PHASE 1: Extract key insights from large research
EXTRACT_KEY_POINTS_PROMPT_EN = """You are a data journalist analyzing research for a 90-second TikTok/Shorts video.

TOPIC: {topic}

RESEARCH DATA:
{research}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK: Extract exactly 5 KEY INSIGHTS from this research.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each insight must:
- Include SPECIFIC NUMBERS (percentages, amounts, dates)
- Be directly relevant to the topic
- Be verifiable from the research
- Focus on patterns, trends, comparisons, or surprising facts

OUTPUT FORMAT:
1. [First key insight with specific data]
2. [Second key insight with specific data]
3. [Third key insight with specific data]
4. [Fourth key insight with specific data]
5. [Fifth key insight with specific data]

Output ONLY the numbered list. No commentary."""

EXTRACT_KEY_POINTS_PROMPT_ES = """Eres un periodista de datos analizando investigación para un video de TikTok/Shorts de 90 segundos.

TEMA: {topic}

DATOS DE INVESTIGACIÓN:
{research}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TAREA: Extrae exactamente 5 INSIGHTS CLAVE de esta investigación.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cada insight debe:
- Incluir NÚMEROS ESPECÍFICOS (porcentajes, cantidades, fechas)
- Ser directamente relevante al tema
- Ser verificable desde la investigación
- Enfocarse en patrones, tendencias, comparaciones o hechos sorprendentes

FORMATO DE SALIDA:
1. [Primer insight clave con datos específicos]
2. [Segundo insight clave con datos específicos]
3. [Tercer insight clave con datos específicos]
4. [Cuarto insight clave con datos específicos]
5. [Quinto insight clave con datos específicos]

Devuelve SOLO la lista numerada. Sin comentarios."""

# PHASE 2: Generate script from extracted insights
SHORTS_SCRIPT_PROMPT_EN = """⚠️ MANDATORY: Your script MUST be 200-225 words (MINIMUM 180). Count before submitting. ⚠️

ROLE: You are a Viral Storyteller creating 90-second TikTok/Shorts videos that hook viewers with surprising narratives.

MISSION: Transform these key insights into ONE compelling script (200-225 words) that tells a STORY, not a data dump.

INPUT DATA:
TOPIC: {topic}
KEY INSIGHTS:
{key_points}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOOK RULES (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The hook MUST be:
- 5-8 words MAX (not 10, not 20)
- A provocative statement, NOT a number
- Creates curiosity gap ("Wait, what?")

GOOD HOOKS:
- "He's not a CEO. He's a time machine."
- "This man beat the market 140 times over."
- "Everyone got this completely wrong."
- "The math here breaks your brain."

BAD HOOKS (NEVER DO):
- "5,502,284% return wasn't luck" (too many digits)
- "From 1965 to 2024, returns were..." (boring setup)
- Statistics as hooks (save them for the body)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUMBER FORMATTING (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USE DIGITS, NOT WORDS. The TTS will pronounce them correctly.

FOR STOCK INDEX NAMES - use SPACE not hyphen:
BAD: "Nasdaq-100", "Russell-2000", "S&P-500"
GOOD: "Nasdaq 100", "Russell 2000", "S&P 500", "Russell 1000", "Russell 3000"
(Hyphens cause TTS to read each digit separately: "one zero zero")

FOR LARGE NUMBERS - round and simplify:
BAD: "$2,665,555" or "5,502,284%"
GOOD: "2.6 million dollars" or "over 5 million percent"

FOR SMALL DECIMALS - keep the digits:
BAD: "three-hundredths of a percent" or "point zero three percent"
GOOD: "0.03%" or "0.06%" (TTS reads these perfectly)

FOR PERCENTAGES - use digits:
BAD: "fifteen point one percent" or "nearly ten percent"
GOOD: "15.1%" or "10%"

FOR YEARS - use digits:
GOOD: "from 1965 to 2024" or "in 2025"

FOR MONEY - digits with word magnitude:
BAD: "$174,494"
GOOD: "175 thousand dollars" or "$175K"

RULE: Write numbers as you want them to appear in captions.
MAX 3 numbers per paragraph. Pick only the most dramatic.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRUCTURE (STORYTELLING FLOW)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOOK (5-8 words): Provocative statement that creates curiosity.

SETUP (40 words): Who, what, when. Set the scene with ONE key number.

TENSION (60 words): The surprising contrast. "But here's what nobody talks about..."

RESOLUTION (60 words): Why it matters. Connect the dots. Human details that stick.

VERDICT (25 words): One-sentence takeaway. What should the viewer remember?

TOTAL: ~200 words

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TONE & STYLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Write like you're telling a friend, not reading a report
- Use human details: "He still lives in the same house since 1952"
- ACTIVE VOICE only: "Buffett bought" not "was bought by Buffett"
- NO academic words: "methodology", "paradigm", "allocation"
- NO hedging: "might", "perhaps", "approximately"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOOK: [5-8 words, provocative, NO numbers]

[Script body as continuous prose - just the words to read on camera]

SOURCES:
1. [source]
2. [source]

⚠️ LENGTH: 200-225 words (MINIMUM 180). Scripts under 180 words = REJECTED. ⚠️

Generate the script now."""

SHORTS_SCRIPT_PROMPT_ES = """⚠️ OBLIGATORIO: Tu guion DEBE tener 200-225 palabras (MÍNIMO 180). Cuenta antes de enviar. ⚠️

ROL: Eres un Periodista de Datos Viral creando videos de "Fact-Check" de 90 segundos para TikTok/Shorts.

MISIÓN: Transformar estos insights clave en UN guion (200-225 palabras) que desmiente mitos, revela tendencias o visualiza datos complejos usando números duros.

DATOS DE ENTRADA:
TEMA: {topic}
INSIGHTS CLAVE:
{key_points}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATEO DE NÚMEROS (OBLIGATORIO)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USA DÍGITOS, NO PALABRAS. El TTS los pronunciará correctamente.
⚠️ USA COMA (,) COMO SEPARADOR DECIMAL - Formato español/europeo. ⚠️

PARA NOMBRES DE ÍNDICES - usa ESPACIO no guión:
MAL: "Nasdaq-100", "Russell-2000", "S&P-500"
BIEN: "Nasdaq 100", "Russell 2000", "S&P 500", "Russell 1000", "Russell 3000"
(El guión hace que el TTS lea cada dígito por separado: "uno cero cero")

PARA NÚMEROS GRANDES - redondea y simplifica:
MAL: "$2,665,555" o "5,502,284%"
BIEN: "2,6 millones de dólares" o "más de 5 millones por ciento"

PARA DECIMALES PEQUEÑOS - usa COMA como decimal:
MAL: "0.03%" o "punto cero tres por ciento"
BIEN: "0,03%" o "0,06%" (el TTS los lee perfectamente con coma)

PARA PORCENTAJES - usa dígitos con COMA:
MAL: "15.1%" o "quince punto uno por ciento"
BIEN: "15,1%" o "10%"

PARA AÑOS - usa dígitos (sin coma ni punto):
BIEN: "de 1965 a 2024" o "en 2025"

PARA DINERO - dígitos con magnitud en palabras:
MAL: "$174,494"
BIEN: "175 mil dólares" o "$175K"

REGLA: Usa COMA para decimales (0,03% no 0.03%). Escribe los números como quieres en subtítulos.
MÁXIMO 3 números por párrafo. Elige solo los más impactantes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGLAS PRINCIPALES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIEMPO: 90 segundos máximo (~200-225 palabras). Ritmo rápido, dicción clara.

PRECISIÓN DE DATOS:
- Usa números REDONDEADOS que suenen naturales al hablar
- Incluye desgloses demográficos cuando existan
- Expón diferencias objetivamente: "Grupo A: 10% retorno. Grupo B: 4%."

TONO:
- Investigador objetivo, no influencer
- Muestra los datos, no prediques
- Cero relleno
- Escribe en VOZ ACTIVA
- Español LATAM neutro (no España, no modismos regionales)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ESTRUCTURA (CON PRESUPUESTO DE PALABRAS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0:00-0:05 | GANCHO (20 PALABRAS)
Afirmación o estadística que interrumpe el patrón. Debe generar curiosidad.

0:05-0:35 | EVIDENCIA (80 PALABRAS)
Estadísticas rápidas, comparaciones, datos duros de los 5 insights.

0:35-1:15 | CONTEXTO (80 PALABRAS)
Por qué los datos se ven así (breve explicación técnica).

1:15-1:30 | VEREDICTO (25 PALABRAS)
Conclusión en una oración + llamado a la acción.

TOTAL: 205 palabras (ESTRICTAMENTE entre 180-225)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATO DE SALIDA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Primero, proporciona el GANCHO como línea separada:
GANCHO: [máximo 10 palabras]

Luego entrega el guion hablado como prosa continua (SIN encabezados de sección, SIN marcas de tiempo).

Solo las palabras exactas para leer en cámara.

Termina con "FUENTES:" seguido de lista numerada.

⚠️ REQUISITO CRÍTICO DE LONGITUD ⚠️
Tu guion DEBE tener EXACTAMENTE 200-225 palabras (MÍNIMO 180 palabras).
Cuenta tus palabras antes de enviar. Guiones con menos de 180 palabras serán RECHAZADOS.
Usa TODA la investigación proporcionada para alcanzar la longitud requerida.

Genera el guion completo de 200-225 palabras ahora."""


# ---------------------------------------------------------------------------
# Web Scraping Functions (from youtube_transcript_merger.py)
# ---------------------------------------------------------------------------

def _is_filtered_domain(url: str) -> bool:
    """Check if URL should be filtered out."""
    return any(domain in url.lower() for domain in SEARCH_FILTER_DOMAINS)


def _search_youtube_videos(query: str, max_results: int = 2) -> List[str]:
    """Search YouTube via yt-dlp (no API key needed) and return video IDs."""
    try:
        import yt_dlp
    except ImportError:
        logger.warning("yt-dlp not installed - skipping YouTube search")
        return []
    
    logger.info("Searching YouTube for: %s", query)
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'skip_download': True,
    }
    
    # Add proxy if configured
    if WEBSHARE_PROXY_URL:
        ydl_opts['proxy'] = WEBSHARE_PROXY_URL
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Search YouTube directly
            results = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            video_ids = []
            for entry in results.get('entries', []):
                if entry and entry.get('id'):
                    video_ids.append(entry['id'])
            logger.info("YouTube found %d videos", len(video_ids))
            return video_ids
    except Exception as e:
        logger.warning("YouTube search failed: %s", e)
        return []


def _fetch_youtube_captions(video_id: str) -> str:
    """Fetch captions for YouTube video using yt-dlp via proxy."""
    try:
        import yt_dlp
        
        # Configure yt-dlp with proxy
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'quiet': True,
            'no_warnings': True,
        }
        
        # Add proxy if configured
        if WEBSHARE_PROXY_URL:
            ydl_opts['proxy'] = WEBSHARE_PROXY_URL
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Get subtitles
            if 'subtitles' in info and 'en' in info['subtitles']:
                sub_url = info['subtitles']['en'][0]['url']
            elif 'automatic_captions' in info and 'en' in info['automatic_captions']:
                sub_url = info['automatic_captions']['en'][0]['url']
            else:
                logger.warning("No captions found for video %s", video_id)
                return ""
            
            # Fetch caption content with proxy
            proxies = REQUESTS_PROXY if REQUESTS_PROXY else {}
            response = requests.get(sub_url, proxies=proxies, timeout=30)
            response.raise_for_status()
            
            # Parse VTT/SRT and extract text
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'lxml')
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up timestamps and duplicates
            text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3}', '', text)
            text = re.sub(r'\d+\s+', '', text)
            text = re.sub(r'\s+', ' ', text)
            
            logger.info("Fetched captions for %s: %d chars", video_id, len(text))
            return text
            
    except ImportError:
        logger.warning("yt-dlp not installed - skipping YouTube captions")
        return ""
    except Exception as e:
        logger.warning("Failed to fetch captions for %s: %s", video_id, e)
        return ""


def _search_duckduckgo(query: str, num_results: int = 4) -> List[str]:
    """Search DuckDuckGo HTML version and return result URLs."""
    logger.info("Searching DuckDuckGo for: %s", query)
    
    if not REQUESTS_PROXY_US:
        logger.error("WEBSHARE_PROXY_URL not set - proxy is required for web scraping")
        return []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query, "kl": "us-en"},
            headers=headers,
            proxies=REQUESTS_PROXY_US,
            timeout=60,
        )
        response.raise_for_status()
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return []
    
    soup = BeautifulSoup(response.text, "lxml")
    urls = []
    
    for a_tag in soup.select("a.result__a"):
        href = a_tag.get("href", "")
        if "uddg=" in href:
            from urllib.parse import unquote
            parsed = parse_qs(urlparse(href).query)
            if "uddg" in parsed:
                href = unquote(parsed["uddg"][0])
        
        if href.startswith("http") and not _is_filtered_domain(href) and href not in urls:
            urls.append(href)
            if len(urls) >= num_results:
                break
    
    logger.info("DuckDuckGo found %d URLs", len(urls))
    return urls


def _search_brave(query: str, num_results: int = 4) -> List[str]:
    """Search Brave and return result URLs."""
    logger.info("Searching Brave for: %s", query)
    
    if not REQUESTS_PROXY_US:
        logger.error("WEBSHARE_PROXY_URL not set - proxy is required for web scraping")
        return []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        response = requests.get(
            "https://search.brave.com/search",
            params={"q": query, "source": "web"},
            headers=headers,
            proxies=REQUESTS_PROXY_US,
            timeout=60,
        )
        response.raise_for_status()
    except Exception as exc:
        logger.warning("Brave search failed: %s", exc)
        return []
    
    soup = BeautifulSoup(response.text, "lxml")
    urls = []
    
    for a_tag in soup.select("a.result-header, div.snippet a[href], a[data-url]"):
        href = a_tag.get("href") or a_tag.get("data-url") or ""
        if href.startswith("http") and not _is_filtered_domain(href) and href not in urls:
            urls.append(href)
            if len(urls) >= num_results:
                break
    
    logger.info("Brave found %d URLs", len(urls))
    return urls


def extract_page_text(url: str, max_chars: int = 8000) -> str:
    """Extract main text content from a webpage."""
    logger.debug("Extracting text from: %s", url)
    
    if not REQUESTS_PROXY:
        logger.error("WEBSHARE_PROXY_URL not set - proxy is required for web scraping")
        return ""
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    try:
        response = requests.get(url, headers=headers, proxies=REQUESTS_PROXY, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return ""
    
    soup = BeautifulSoup(response.content, "lxml")
    
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        element.decompose()
    
    main_content = None
    for selector in ["article", "main", '[role="main"]', ".content", "#content", ".post-content"]:
        main_content = soup.select_one(selector)
        if main_content:
            break
    
    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        body = soup.find("body")
        text = body.get_text(separator="\n", strip=True) if body else ""
    
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[... truncated ...]"
    
    return text


def scrape_research(topic: str, num_links: int = 6) -> str:
    """Search DuckDuckGo + Brave + YouTube, scrape top results, return merged research text."""
    logger.info("Scraping research for: %s", topic)
    
    all_texts = []
    
    # 1. Get URLs from web search engines
    all_urls = []
    seen = set()
    
    for search_func in [_search_duckduckgo, _search_brave]:
        try:
            urls = search_func(topic, num_results=4)
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    all_urls.append(url)
        except Exception as e:
            logger.warning("Search engine error: %s", e)
    
    # Extract text from web URLs
    for i, url in enumerate(all_urls[:num_links], 1):
        text = extract_page_text(url)
        if text and len(text) > 200:
            all_texts.append(f"=== WEB SOURCE {i}: {url} ===\n{text}\n")
    
    # 2. Get YouTube video captions
    try:
        video_ids = _search_youtube_videos(topic, max_results=2)
        for i, video_id in enumerate(video_ids, 1):
            captions = _fetch_youtube_captions(video_id)
            if captions and len(captions) > 500:
                all_texts.append(f"=== YOUTUBE SOURCE {i}: https://youtube.com/watch?v={video_id} ===\n{captions}\n")
    except Exception as e:
        logger.warning("YouTube scraping error: %s", e)
    
    if not all_texts:
        logger.error("No research content scraped!")
        return f"Topic: {topic}\n(Failed to extract content from any sources)"
    
    merged = "\n".join(all_texts)
    logger.info("Scraped research: %d sources, %d chars", len(all_texts), len(merged))
    
    # Validate research quality
    if len(merged) < 5000:
        logger.warning("Research content is short (%d chars) - may produce poor scripts", len(merged))
    
    # Truncate to avoid overwhelming LLM (shorts only need key facts)
    MAX_RESEARCH_CHARS = 20000
    if len(merged) > MAX_RESEARCH_CHARS:
        logger.info("Truncating research from %d to %d chars", len(merged), MAX_RESEARCH_CHARS)
        merged = merged[:MAX_RESEARCH_CHARS] + "\n[... additional sources truncated ...]\n"
    
    return merged


# ---------------------------------------------------------------------------
# PHASE 1: Key Points Extraction Functions
# ---------------------------------------------------------------------------

def extract_key_points_gemini(prompt: str) -> Optional[str]:
    """Extract key points using Gemini."""
    if not GEMINI_API_KEY:
        return None
    
    try:
        url = GEMINI_API_URL.format(model=GEMINI_MODEL) + f"?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.5, "maxOutputTokens": 800},
        }
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            points = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            if points and len(points) > 100:
                return points
        else:
            logger.warning("Gemini extraction returned %d", response.status_code)
    except Exception as e:
        logger.warning("Gemini extraction failed: %s", e)
    
    return None


def extract_key_points_xiaomi(prompt: str) -> Optional[str]:
    """Extract key points using Xiaomi via OpenRouter."""
    if not OPENROUTER_API_KEY:
        return None
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": XIAOMI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 800,
        }
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.warning("Xiaomi extraction failed: %s", e)
    
    return None


def extract_key_points_glm(prompt: str) -> Optional[str]:
    """Extract key points using GLM-4-Plus via OpenRouter."""
    if not OPENROUTER_API_KEY:
        return None
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 800,
        }
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.warning("GLM extraction failed: %s", e)
    
    return None


def extract_key_points_olmo(prompt: str) -> Optional[str]:
    """Extract key points using Olmo via OpenRouter."""
    if not OPENROUTER_API_KEY:
        return None
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OLMO_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 800,
        }
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.warning("Olmo extraction failed: %s", e)
    
    return None


def extract_key_points(topic: str, research: str, language: str = "en") -> str:
    """Extract 5 key insights from research (Phase 1 of two-phase generation)."""
    logger.info("Phase 1: Extracting key points from research (%d chars)...", len(research))
    
    if language == "es":
        extract_prompt = EXTRACT_KEY_POINTS_PROMPT_ES.format(topic=topic, research=research)
    else:
        extract_prompt = EXTRACT_KEY_POINTS_PROMPT_EN.format(topic=topic, research=research)
    
    # Try models in order until one succeeds
    models = [
        (extract_key_points_gemini, "Gemini"),
        (extract_key_points_xiaomi, "Xiaomi"),
        (extract_key_points_glm, "GLM-4-Plus"),
        (extract_key_points_olmo, "Olmo"),
    ]
    
    for model_func, model_name in models:
        key_points = model_func(extract_prompt)
        if key_points and len(key_points) > 100:
            logger.info("✓ %s extracted key points (%d chars)", model_name, len(key_points))
            return key_points
        logger.warning("%s failed to extract key points", model_name)
        time.sleep(2)
    
    # Fallback: if all models fail, return a truncated version of research
    logger.warning("All models failed extraction. Using truncated research as fallback.")
    return research[:3000] + "\n[... truncated ...]\n"


# ---------------------------------------------------------------------------
# PHASE 2: Script Generation Functions
# ---------------------------------------------------------------------------

def generate_script_gemini(prompt: str) -> Optional[str]:
    """Generate script using Gemini."""
    if not GEMINI_API_KEY:
        return None
    
    for attempt in range(MAX_LLM_RETRIES):
        try:
            url = GEMINI_API_URL.format(model=GEMINI_MODEL) + f"?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1500},
            }
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                script = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if script and len(script) > 50:
                    return script
            else:
                logger.warning("Gemini returned %d: %s", response.status_code, response.text[:200])
        except Exception as e:
            logger.warning("Gemini attempt %d failed: %s", attempt + 1, e)
            time.sleep(2)
    
    return None


def generate_script_xiaomi(prompt: str) -> Optional[str]:
    """Generate script using Xiaomi via OpenRouter (uses same scraped research)."""
    if not OPENROUTER_API_KEY:
        return None
    
    logger.info("Falling back to Xiaomi for script generation...")
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": XIAOMI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.warning("Xiaomi via OpenRouter returned %d: %s", response.status_code, response.text[:200])
    except Exception as e:
        logger.error("Xiaomi script generation failed: %s", e)
    
    return None


def generate_script_glm(prompt: str) -> Optional[str]:
    """Generate script using GLM-4-Plus via OpenRouter."""
    if not OPENROUTER_API_KEY:
        return None
    
    logger.info("Falling back to GLM-4-Plus for script generation...")
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.warning("GLM via OpenRouter returned %d: %s", response.status_code, response.text[:200])
    except Exception as e:
        logger.error("GLM script generation failed: %s", e)
    
    return None


def generate_script_olmo(prompt: str) -> Optional[str]:
    """Generate script using AllenAI Olmo 3.1 32B Think via OpenRouter (reasoning model fallback)."""
    if not OPENROUTER_API_KEY:
        return None
    
    logger.info("Falling back to Olmo 3.1 32B Think for script generation...")
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OLMO_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000,
            "reasoning": {"enabled": True},  # Enable reasoning for better quality
        }
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.warning("Olmo via OpenRouter returned %d: %s", response.status_code, response.text[:200])
    except Exception as e:
        logger.error("Olmo script generation failed: %s", e)
    
    return None


# Expansion prompt for scripts that are too short
EXPAND_PROMPT_EN = """The following script is only {word_count} words, which is BELOW the 180-word minimum.

You MUST expand it to 200-225 words. This is MANDATORY.

EXPANSION PROTOCOL:
1. Add more statistics and data points from the research
2. Include additional context explaining WHY the data matters
3. Add comparisons (before/after, group A vs group B)
4. Expand the verdict section with implications

RULES:
- Keep the hook intact
- Maintain the same structure (hook → evidence → context → verdict)
- Add substance, not fluff
- Every added sentence must include a fact or number

Output ONLY the expanded script. No commentary.

SCRIPT TO EXPAND:
{script}"""

EXPAND_PROMPT_ES = """El siguiente guion tiene solo {word_count} palabras, que está POR DEBAJO del mínimo de 180 palabras.

DEBES expandirlo a 200-225 palabras. Esto es OBLIGATORIO.

PROTOCOLO DE EXPANSIÓN:
1. Agrega más estadísticas y datos de la investigación
2. Incluye contexto adicional explicando POR QUÉ importan los datos
3. Agrega comparaciones (antes/después, grupo A vs grupo B)
4. Expande la sección del veredicto con implicaciones

REGLAS:
- Mantén el gancho intacto
- Mantén la misma estructura (gancho → evidencia → contexto → veredicto)
- Agrega sustancia, no relleno
- Cada oración agregada debe incluir un dato o número

Devuelve SOLO el guion expandido. Sin comentarios.

GUIÓN A EXPANDIR:
{script}"""


def generate_script(topic: str, research: str, language: str = "en") -> str:
    """Generate script in specified language using two-phase approach:
    Phase 1: Extract key points from large research
    Phase 2: Generate script from focused key points
    """
    logger.info("Generating %s script for: %s", language.upper(), topic[:50])

    if not GEMINI_API_KEY and not OPENROUTER_API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY or OPENROUTER_API_KEY to generate scripts")
    
    # PHASE 1: Extract key insights from research (reduces 20K chars → ~500 chars)
    key_points = extract_key_points(topic, research, language)
    logger.info("Phase 1 complete. Key points: %d chars", len(key_points))
    
    # PHASE 2: Generate script from key points
    logger.info("Phase 2: Generating script from key points...")
    
    if language == "es":
        base_prompt = SHORTS_SCRIPT_PROMPT_ES.format(topic=topic, key_points=key_points)
        expand_prompt_template = EXPAND_PROMPT_ES
    else:
        base_prompt = SHORTS_SCRIPT_PROMPT_EN.format(topic=topic, key_points=key_points)
        expand_prompt_template = EXPAND_PROMPT_EN
    
    # Model chain: try each until we get a script with sufficient length
    models = [
        (generate_script_gemini, "Gemini"),
        (generate_script_xiaomi, "Xiaomi"),
        (generate_script_glm, "GLM-4-Plus"),
        (generate_script_olmo, "Olmo"),
    ]
    
    MAX_TOTAL_ATTEMPTS = 8  # Try up to 8 times across all models
    
    for attempt in range(1, MAX_TOTAL_ATTEMPTS + 1):
        # Select model (cycle through them)
        model_func, model_name = models[(attempt - 1) % len(models)]
        
        # Generate script
        script = model_func(base_prompt)
        if not script:
            logger.warning("%s failed to generate script (attempt %d)", model_name, attempt)
            time.sleep(2)
            continue
        
        word_count = len(script.split())
        logger.info("%s generated %d words (attempt %d/%d)", model_name, word_count, attempt, MAX_TOTAL_ATTEMPTS)
        
        # Check if length is sufficient
        if word_count >= MIN_SCRIPT_WORDS:
            logger.info("✓ Script length OK (%d words)", word_count)
            return script
        
        # Try expansion with current script before moving to next model
        if word_count > 20:  # Only try expansion if we have something to work with
            logger.info("Trying expansion on %d-word script...", word_count)
            expand_prompt = expand_prompt_template.format(word_count=word_count, script=script)
            expanded = model_func(expand_prompt)
            
            if expanded:
                expanded_count = len(expanded.split())
                logger.info("Expansion resulted in %d words", expanded_count)
                if expanded_count >= MIN_SCRIPT_WORDS:
                    logger.info("✓ Expanded script meets requirement")
                    return expanded
                # Keep the expanded version if it's longer
                if expanded_count > word_count:
                    script = expanded
                    word_count = expanded_count
        
        logger.warning("Script too short (%d words, need %d+). Trying next model...", word_count, MIN_SCRIPT_WORDS)
        time.sleep(3)
    
    # If we exhausted all attempts, return the last script with a warning
    logger.warning("⚠️ Failed to generate script with %d+ words after %d attempts. Returning last attempt (%d words).", 
                  MIN_SCRIPT_WORDS, MAX_TOTAL_ATTEMPTS, word_count)
    return script


def extract_spoken_text(script: str) -> str:
    """Extract only the spoken text from script (remove HOOK/GANCHO label, SOURCES, etc.)."""
    lines = script.strip().split("\n")
    spoken_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip SOURCES/FUENTES section
        if any(line.upper().startswith(prefix) for prefix in ["SOURCES:", "SOURCE:", "FUENTES:", "FUENTE:"]):
            break
        
        # Handle HOOK/GANCHO prefix
        for prefix in ["HOOK:", "GANCHO:"]:
            if line.upper().startswith(prefix):
                hook_text = line[len(prefix):].strip()
                if hook_text:
                    spoken_lines.append(hook_text)
                line = ""
                break
        
        if not line:
            continue
        
        # Skip [Image of...] tags
        if line.startswith("[") and "]" in line:
            after_tag = line[line.index("]") + 1:].strip()
            if after_tag:
                spoken_lines.append(after_tag)
            continue
        
        spoken_lines.append(line)
    
    return " ".join(spoken_lines)


def extract_proper_nouns(script: str) -> List[str]:
    """Extract proper nouns (names, companies, places) from script for subtitle hints.
    
    This helps Whisper spell names correctly in subtitles (e.g., 'Greg Abel' not 'gregavel').
    Works for both English and Spanish scripts.
    """
    proper_nouns = []
    
    # Remove SOURCES section and HOOK/GANCHO labels first
    script_text = script.split("SOURCES:")[0].split("FUENTES:")[0]
    script_text = re.sub(r'^(HOOK|GANCHO):\s*', '', script_text, flags=re.MULTILINE)
    
    # Common words to filter out (English and Spanish) - expanded list
    common_words = {
        # English
        'the', 'and', 'but', 'for', 'with', 'from', 'that', 'this', 'what', 'when', 
        'where', 'which', 'who', 'why', 'how', 'not', 'are', 'was', 'were', 'been',
        'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might',
        'his', 'her', 'its', 'their', 'our', 'your', 'my', 'one', 'two', 'three',
        'first', 'second', 'third', 'last', 'next', 'new', 'old', 'good', 'bad',
        'big', 'small', 'high', 'low', 'long', 'short', 'real', 'true', 'false',
        'still', 'just', 'only', 'even', 'also', 'most', 'more', 'less', 'much',
        'many', 'some', 'any', 'all', 'every', 'each', 'both', 'few', 'other',
        'same', 'different', 'such', 'than', 'then', 'now', 'here', 'there',
        'think', 'know', 'see', 'look', 'want', 'give', 'take', 'make', 'get',
        'come', 'go', 'say', 'tell', 'ask', 'use', 'find', 'put', 'try', 'leave',
        # Spanish common words that start sentences (capitalized)
        'pero', 'desde', 'cuando', 'donde', 'como', 'porque', 'aunque', 'mientras',
        'hasta', 'sobre', 'entre', 'hacia', 'durante', 'mediante', 'tras', 'según',
        'antes', 'después', 'eso', 'esa', 'este', 'esta', 'esto', 'ese', 'aquel',
        'aquella', 'aquello', 'paciencia', 'tiempo', 'años', 'dato', 'datos',
        'solo', 'cada', 'toda', 'todo', 'ningún', 'algún', 'otro', 'otra',
        'más', 'menos', 'muy', 'mucho', 'poco', 'tanto', 'tan', 'bien', 'mal',
        'ahora', 'hoy', 'ayer', 'mañana', 'siempre', 'nunca', 'también', 'además',
        'primero', 'segundo', 'tercero', 'último', 'nuevo', 'viejo', 'grande',
        'pequeño', 'alto', 'bajo', 'largo', 'corto', 'real', 'verdadero', 'falso',
        'mismo', 'diferente', 'igual', 'mejor', 'peor', 'mayor', 'menor',
        'estos', 'estas', 'esos', 'esas', 'aquellos', 'aquellas', 'unos', 'unas',
        'sus', 'nuestro', 'nuestra', 'vuestro', 'vuestra', 'suyo', 'suya',
        'quien', 'cual', 'cuyo', 'cuya', 'cuanto', 'cuanta', 'algo', 'nada',
        'nadie', 'alguien', 'cualquier', 'cualquiera', 'quienquiera',
        'embargo', 'así', 'entonces', 'luego', 'pues', 'sino', 'siquiera',
        # Labels to filter
        'hook', 'gancho', 'sources', 'fuentes', 'source', 'fuente',
    }
    
    # Regex patterns for proper nouns (works for EN and ES)
    patterns = [
        # Full names: "Warren Buffett", "Greg Abel", "Carlos Slim" (2-3 capitalized words)
        r'\b([A-Z][a-záéíóúñ]+\s+[A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+)?)\b',
        # Acronyms/tickers: "S&P 500", "NYSE", "NASDAQ", "IBEX 35"
        r'\b(S&P\s*\d+|NYSE|NASDAQ|IBEX\s*\d+|DAX|CAC\s*\d+|FTSE\s*\d*)\b',
        # ETF/Stock tickers: 2-5 uppercase letters (VOO, IVV, SPY, QQQ, etc.)
        r'\b([A-Z]{2,5})\b',
    ]
    
    # Common uppercase words that are NOT tickers (to filter out)
    common_uppercase = {
        'THE', 'AND', 'FOR', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS',
        'ONE', 'OUR', 'OUT', 'ARE', 'HIS', 'HAS', 'HAD', 'ITS', 'WHO', 'HOW',
        'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'WAY', 'ANY', 'DAY', 'DID', 'GET',
        'HIM', 'MAN', 'TOO', 'TWO', 'USE', 'HER', 'SAY', 'SHE', 'OWN', 'LET',
        'PUT', 'END', 'WHY', 'TRY', 'ASK', 'BIG', 'PAY', 'YES', 'TOP', 'SET',
        # Spanish uppercase
        'POR', 'SUS', 'MAS', 'SIN', 'CON', 'UNA', 'LOS', 'LAS', 'DEL', 'UNO',
        'QUE', 'ESO', 'ESA', 'HAY', 'HOY', 'AUN', 'TAN', 'MUY', 'YA',
    }
    
    seen = set()
    for pattern in patterns:
        matches = re.findall(pattern, script_text)
        for match in matches:
            name = match.strip()
            # Filter out if ALL words are common
            words = name.lower().split()
            if all(w in common_words for w in words):
                continue
            # Filter out common uppercase words that look like tickers
            if name.upper() in common_uppercase:
                continue
            # Skip single words (likely false positives) unless it's an all-caps ticker
            if ' ' not in name and len(name) < 6 and not name.isupper():
                continue
            if len(name) >= 2 and name not in seen:
                seen.add(name)
                proper_nouns.append(name)
    
    # Known names/companies that often appear in finance videos (EN + ES + International)
    known_entities = [
        # People - Investors/CEOs
        "Warren Buffett", "Charlie Munger", "Greg Abel", "Ajit Jain",
        "Elon Musk", "Tim Cook", "Jeff Bezos", "Bill Gates", "Mark Zuckerberg",
        "Satya Nadella", "Sundar Pichai", "Jensen Huang", "Sam Altman",
        "Jamie Dimon", "Larry Fink", "Ray Dalio", "Carl Icahn", "George Soros",
        "Peter Lynch", "Howard Marks", "Michael Burry", "Cathie Wood",
        "Jerome Powell", "Janet Yellen", "Christine Lagarde",
        # People - LATAM/Spain
        "Carlos Slim", "Ricardo Salinas", "Germán Larrea", "María Asunción Aramburuzabala",
        "Jorge Paulo Lemann", "Marcel Herrmann Telles", "Eduardo Saverin",
        "Amancio Ortega", "Rafael del Pino", "Juan Roig", "Florentino Pérez",
        # Companies - US Tech
        "Apple", "Microsoft", "Google", "Alphabet", "Amazon", "Meta", "Facebook",
        "Tesla", "Nvidia", "Netflix", "Adobe", "Salesforce", "Oracle", "Intel", "AMD",
        "OpenAI", "Anthropic", "DeepMind",
        # Companies - US Finance/Traditional
        "Berkshire Hathaway", "JPMorgan", "Goldman Sachs", "Morgan Stanley",
        "Bank of America", "Wells Fargo", "Citigroup", "BlackRock", "Vanguard",
        "Coca-Cola", "PepsiCo", "McDonald's", "Walmart", "Costco", "Target",
        "Johnson & Johnson", "Pfizer", "Moderna", "UnitedHealth",
        "ExxonMobil", "Chevron", "Boeing", "Lockheed Martin",
        "American Express", "Visa", "Mastercard", "PayPal",
        # Companies - International
        "Samsung", "Toyota", "Sony", "Nintendo", "Alibaba", "Tencent", "TSMC",
        "Inditex", "Zara", "Santander", "BBVA", "Telefónica", "Iberdrola",
        "Pemex", "América Móvil", "Grupo Bimbo", "FEMSA", "Cemex",
        "Petrobras", "Vale", "Itaú", "Bradesco", "Mercado Libre",
        # Indices/Markets
        "S&P 500", "Dow Jones", "NASDAQ", "Nasdaq 100", "NYSE", "Wall Street",
        "Russell 1000", "Russell 2000", "Russell 3000",
        "IBEX 35", "DAX", "CAC 40", "FTSE 100", "Nikkei", "Hang Seng",
        "Bolsa Mexicana", "Bovespa", "BMV",
        # Popular ETF tickers (frequently mentioned)
        "SPY", "VOO", "IVV", "QQQ", "VTI", "VGT", "SMH", "ARKK", "XLF", "XLK",
        "VIG", "SCHD", "JEPI", "VYM", "DVY", "HDV", "VNQ", "DIA", "IWM", "EFA",
        # Institutions
        "Federal Reserve", "Fed", "BCE", "Banco Central Europeo",
        "Reserva Federal", "FMI", "Banco Mundial", "SEC", "CNMV", "CNBV",
        # Places
        "Omaha", "Nebraska", "Wall Street", "New York", "Silicon Valley",
        "Ciudad de México", "São Paulo", "Buenos Aires", "Madrid", "Barcelona",
    ]
    
    # Check for known entities in script (case-insensitive matching)
    script_lower = script_text.lower()
    for name in known_entities:
        if name.lower() in script_lower and name not in seen:
            proper_nouns.append(name)
            seen.add(name)
    
    # Also extract single capitalized words that are likely company names (5+ chars, not common)
    single_caps = re.findall(r'\b([A-Z][a-záéíóúñ]{4,})\b', script_text)
    for word in single_caps:
        if word.lower() not in common_words and word not in seen:
            # Check if it's likely a proper noun (appears multiple times or in known list)
            if script_text.count(word) >= 2 or any(word in entity for entity in known_entities):
                proper_nouns.append(word)
                seen.add(word)
    
    logger.info("Extracted %d proper nouns for subtitles: %s", len(proper_nouns), proper_nouns[:15])
    return proper_nouns


# ---------------------------------------------------------------------------
# MiniMax TTS Voice Generation
# ---------------------------------------------------------------------------

def generate_voiceover(text: str, output_path: Path, language: str = "en") -> Path:
    """Generate voiceover using InWorld TTS (max model) with language-specific voices.

    Applies small normalizations to improve TTS pronunciation without changing captions.
    - English: "S&P 500" → "S and P 500" (natural English pronunciation)
    - Spanish: "S&P 500" → "ese i pe 500" (spelled out in Spanish)
    """
    logger.info("Generating voiceover (%s) - %d chars", language, len(text))
    import re

    text_for_tts = text
    if language == "en":
        # English: pronounce as "S and P 500"
        text_for_tts = re.sub(r"\bS\s*&?\s*P\s*[-\s]?\s*500\b", "S and P 500", text_for_tts, flags=re.IGNORECASE)
    elif language == "es":
        # Spanish: pronounce as "ese i pe 500" (spelled out)
        text_for_tts = re.sub(r"\bS\s*&?\s*P\s*[-\s]?\s*500\b", "ese i pe 500", text_for_tts, flags=re.IGNORECASE)

    return _try_inworld_tts(text_for_tts, output_path, language)


def _try_minimax_tts(text: str, output_path: Path) -> Path:
    """Try to generate voiceover using MiniMax TTS."""
    if not MINIMAX_API_KEY:
        raise RuntimeError("MINIMAX_API_KEY not set in .env")
    if not MINIMAX_VOICE_ID:
        raise RuntimeError("MINIMAX_CLONE_VOICE_ID not set in .env")
    
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": "speech-2.6-hd",
        "text": text,
        "stream": False,
        "voice_setting": {
            "voice_id": MINIMAX_VOICE_ID,
            "speed": 1.0,
            "vol": 1,
            "pitch": 0,
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 1,
        },
    }
    
    try:
        response = requests.post(MINIMAX_TTS_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        
        if "base_resp" in data and data["base_resp"].get("status_code") != 0:
            error_msg = data["base_resp"].get("status_msg", "Unknown error")
            raise RuntimeError(f"MiniMax TTS error: {error_msg}")
        
        audio_hex = data.get("data", {}).get("audio")
        if not audio_hex:
            raise RuntimeError("No audio data in MiniMax response")
        
        audio_bytes = bytes.fromhex(audio_hex)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        
        logger.info("Voiceover saved via MiniMax: %s (%d bytes)", output_path.name, len(audio_bytes))
        return output_path
        
    except requests.exceptions.RequestException as e:
        logger.error("MiniMax TTS request failed: %s", e)
        raise RuntimeError(f"MiniMax TTS failed: {e}")


def _try_inworld_tts(text: str, output_path: Path, language: str = "en") -> Path:
    """Generate voiceover using InWorld TTS (max model) with language-specific voices."""
    import base64
    
    inworld_api_key = os.getenv("INWORLD_API_KEY")
    if not inworld_api_key:
        raise RuntimeError("INWORLD_API_KEY not set in .env")
    
    # Language-specific voice IDs
    voice_map = {
        "en": "default-neh9nlbe9oahtxpstieafq__david_tarazona",  # English voice
        "es": "default-neh9nlbe9oahtxpstieafq__dtspa1",           # Spanish voice
    }
    voice_id = voice_map.get(language, voice_map["en"])
    
    # Prepare InWorld request using max model
    headers = {
        "Authorization": f"Basic {inworld_api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "text": text,
        "voice_id": voice_id,
        "audio_config": {
            "audio_encoding": "MP3",
            "speaking_rate": 1
        },
        "temperature": 1.1,
        "model_id": "inworld-tts-1-max"
    }
    
    try:
        response = requests.post(
            "https://api.inworld.ai/tts/v1/voice",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        
        result = response.json()
        audio_content = base64.b64decode(result['audioContent'])
        if not audio_content:
            raise RuntimeError("No audio data in InWorld response")
        
        with open(output_path, "wb") as f:
            f.write(audio_content)
        
        logger.info("Voiceover saved via InWorld TTS (%s): %s (%d bytes)", language, output_path.name, len(audio_content))
        return output_path
        
    except requests.exceptions.RequestException as e:
        logger.error("InWorld TTS request failed: %s", e)
        raise RuntimeError(f"InWorld TTS failed: {e}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

class ShortsVideoBot:
    """End-to-end shorts video generator - produces EN + ES versions."""
    
    def __init__(self, work_dir: Path = None, keep_temp: bool = False):
        self.work_dir = work_dir or Path(tempfile.mkdtemp(prefix="shorts_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.keep_temp = keep_temp
        logger.info("ShortsVideoBot initialized. Work dir: %s", self.work_dir)
    
    def generate(
        self,
        topic: str,
        output_base: Path,
        whisper_model: str = "medium",
        skip_subtitles: bool = False,
    ) -> Tuple[Path, Path]:
        """Generate complete short-form videos in EN + ES from topic.
        
        Returns:
            Tuple of (english_video_path, spanish_video_path)
        """
        
        logger.info("=" * 60)
        logger.info("SHORTS VIDEO BOT - DUAL LANGUAGE PIPELINE")
        logger.info("Topic: %s", topic)
        logger.info("=" * 60)
        
        # Step 1: Scrape Research (one time, used for both languages)
        logger.info("Step 1/6: Scraping web research...")
        research = scrape_research(topic, num_links=6)
        
        research_path = self.work_dir / "research.txt"
        research_path.write_text(research, encoding="utf-8")
        logger.info("Research saved: %s (%d chars)", research_path, len(research))
        
        # Step 2: Generate English Script
        logger.info("Step 2/6: Generating ENGLISH script...")
        script_en = generate_script(topic, research, language="en")
        script_en_path = self.work_dir / "script_en.txt"
        script_en_path.write_text(script_en, encoding="utf-8")
        
        spoken_en = extract_spoken_text(script_en)
        spoken_en_path = self.work_dir / "spoken_en.txt"
        spoken_en_path.write_text(spoken_en, encoding="utf-8")
        logger.info("English script: %d words", len(spoken_en.split()))
        
        # Step 3: Generate Spanish Script
        logger.info("Step 3/6: Generating SPANISH (LATAM) script...")
        script_es = generate_script(topic, research, language="es")
        script_es_path = self.work_dir / "script_es.txt"
        script_es_path.write_text(script_es, encoding="utf-8")
        
        spoken_es = extract_spoken_text(script_es)
        spoken_es_path = self.work_dir / "spoken_es.txt"
        spoken_es_path.write_text(spoken_es, encoding="utf-8")
        logger.info("Spanish script: %d words", len(spoken_es.split()))
        
        # Extract proper nouns from scripts for accurate subtitle spelling
        proper_nouns_en = extract_proper_nouns(script_en)
        proper_nouns_es = extract_proper_nouns(script_es)
        
        # Step 4: Generate Voiceovers (skip if already generated to save credits)
        logger.info("Step 4/6: Generating voiceovers with MiniMax...")
        audio_en_path = self.work_dir / "voiceover_en.mp3"
        audio_es_path = self.work_dir / "voiceover_es.mp3"
        
        # Check if audio already exists and has reasonable size (>10KB)
        if audio_en_path.exists() and audio_en_path.stat().st_size > 10000:
            logger.info("✓ Reusing existing EN voiceover: %s (%d bytes)", audio_en_path.name, audio_en_path.stat().st_size)
        else:
            generate_voiceover(spoken_en, audio_en_path, language="en")
        
        if audio_es_path.exists() and audio_es_path.stat().st_size > 10000:
            logger.info("✓ Reusing existing ES voiceover: %s (%d bytes)", audio_es_path.name, audio_es_path.stat().st_size)
        else:
            generate_voiceover(spoken_es, audio_es_path, language="es")
        
        # Step 5: Generate English Video
        logger.info("Step 5/6: Generating ENGLISH video...")
        output_en = output_base.with_name(output_base.stem + "_EN.mp4")
        # Create shared bot instance to maintain used video URLs across both languages
        shared_video_bot = AutoVideoBot(output_dir=self.work_dir / "video_en", keep_temp=self.keep_temp)
        shared_video_bot.generate_video(
            audio_path=audio_en_path,
            output_path=output_en,
            whisper_model=whisper_model,
            skip_subtitles=skip_subtitles,
            language="en",
            proper_nouns=proper_nouns_en,
            original_script=script_en,  # Pass original script for entity extraction
            spoken_script=spoken_en,    # Pass spoken text for hybrid caption alignment
        )
        
        # Step 6: Generate Spanish Video
        logger.info("Step 6/6: Generating SPANISH video...")
        output_es = output_base.with_name(output_base.stem + "_ES.mp4")
        # Reuse bot instance to prevent B-roll repetition (CEO/company images can repeat)
        shared_video_bot.output_dir = self.work_dir / "video_es"
        shared_video_bot.temp_clips_dir = shared_video_bot.output_dir / "clips"
        shared_video_bot.temp_clips_dir.mkdir(parents=True, exist_ok=True)
        shared_video_bot.temp_images_dir = shared_video_bot.output_dir / "images"
        shared_video_bot.temp_images_dir.mkdir(parents=True, exist_ok=True)
        shared_video_bot.generate_video(
            audio_path=audio_es_path,
            output_path=output_es,
            whisper_model=whisper_model,
            skip_subtitles=skip_subtitles,
            language="es",
            proper_nouns=proper_nouns_es,
            original_script=script_es,  # Pass original script for entity extraction
            spoken_script=spoken_es,    # Pass spoken text for hybrid caption alignment
        )
        
        logger.info("=" * 60)
        logger.info("SHORTS VIDEO COMPLETE - DUAL LANGUAGE!")
        logger.info("English: %s", output_en)
        logger.info("Spanish: %s", output_es)
        logger.info("Scripts: %s, %s", script_en_path, script_es_path)
        if self.keep_temp:
            logger.info("Work dir: %s", self.work_dir)
        logger.info("=" * 60)
        
        return output_en, output_es


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ShortsVideoBot - Generate TikTok/Reels/Shorts in EN + ES from a topic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "topic",
        type=str,
        help="Topic or idea for the short video (e.g., 'Why Bitcoin is crashing in 2025')",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Base output video path. Will generate _EN.mp4 and _ES.mp4",
    )
    parser.add_argument(
        "--whisper-model", "-w",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for transcription (default: medium)",
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
    
    # Generate output filename from topic
    if args.output is None:
        # Default to C:\Users\david\Videos folder
        videos_folder = Path(r"C:\Users\david\Videos")
        videos_folder.mkdir(parents=True, exist_ok=True)
        
        safe_name = re.sub(r'[^\w\s-]', '', args.topic)[:40].strip().replace(' ', '_').lower()
        args.output = videos_folder / f"{safe_name}_short.mp4"
    
    bot = ShortsVideoBot(work_dir=args.work_dir, keep_temp=args.keep_temp)
    bot.generate(
        topic=args.topic,
        output_base=args.output,
        whisper_model=args.whisper_model,
        skip_subtitles=args.no_subtitles,
    )


if __name__ == "__main__":
    main()

