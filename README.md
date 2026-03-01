# ai-video-pipeline

End-to-end AI video pipeline: TTS voice synthesis, FFmpeg video assembly, automated YouTube upload via OAuth 2.0 & Data API.

## Scripts

| Script | Description |
|--------|-------------|
| `auto_video_bot.py` | Main automation bot: generates script with AI → TTS voiceover → assembles video with FFmpeg → uploads to YouTube |
| `ai_video_editor.py` | AI-powered video editing: selects clips, adds captions, applies transitions using FFmpeg |
| `batch_video_generator.py` | Batch mode: processes a queue of topics and generates/uploads multiple videos unattended |
| `shorts_video_bot.py` | Generates YouTube Shorts (vertical 9:16 format) — optimised for short-form content |
| `process_clips_with_captions.py` | Adds burned-in captions to video clips — synchronised with TTS audio |

## Prerequisites

- Python 3.9+
- **FFmpeg** installed and on PATH: `ffmpeg -version`
- Google Cloud project with **YouTube Data API v3** enabled
- OAuth 2.0 client credentials (`client_secrets.json`) — stored outside repo
- OpenAI API key (for script generation) or Gemini API key

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY, YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET
```

Place `client_secrets.json` in the project root (it is in `.gitignore`).

## Usage

```bash
# Generate and upload a single video
python auto_video_bot.py --topic "Top 5 Python tips 2025"

# Batch generate from a topics file
python batch_video_generator.py --topics topics.txt

# Generate a YouTube Short
python shorts_video_bot.py --topic "Quick Python tip"

# Add captions to existing clips
python process_clips_with_captions.py --input clips/ --output output/
```

First run will open a browser for YouTube OAuth authentication. Token is cached for future runs.

## Notes

- YouTube Data API quota: 10,000 units/day on free tier. Each video upload costs ~1,600 units.
- FFmpeg must support the codec used by your source clips (H.264 recommended).

## Built with

Python · FFmpeg · OpenAI API · YouTube Data API v3 · OAuth 2.0  
AI-assisted development (Claude, GitHub Copilot) — architecture, requirements, QA validation and debugging by me.
