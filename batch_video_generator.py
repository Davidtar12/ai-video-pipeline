"""
Batch Video Generator - Run multiple video topics automatically
Usage: python batch_video_generator.py
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'batch_videos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# List of video topics to generate
VIDEO_TOPICS = [
    "Low expense ETFs that outperformed the SP500 in 2025",
    "Top 5 Nasdaq-100 stocks in 2025. Micron Technology, Warner Bros. Discovery, Lam Research, Palantir Technologies, Applovin Corp.",
    "Top 5 Russell-1000 stocks in 2025. MP Materials, Lumentum, Western Digital, Robinhood, Insmed.",
]

def sanitize_filename(topic: str) -> str:
    """Convert topic to safe filename."""
    # Remove special characters, replace spaces with underscores
    safe = "".join(c if c.isalnum() or c in (' ', '-') else '' for c in topic)
    safe = safe.replace(' ', '_').lower()
    # Limit length
    return safe[:50]

def generate_video(topic: str, output_dir: Path) -> bool:
    """Generate video for a single topic."""
    try:
        # Create output filename
        filename = sanitize_filename(topic)
        output_path = output_dir / f"{filename}.mp4"
        
        logger.info("=" * 80)
        logger.info(f"Starting video generation for: {topic}")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 80)
        
        # Run shorts_video_bot.py
        cmd = [
            sys.executable,
            "shorts_video_bot.py",
            topic,
            "--output", str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout per video
        )
        
        if result.returncode == 0:
            logger.info(f"✅ SUCCESS: {topic}")
            logger.info(f"   English: {output_path.with_name(output_path.stem + '_EN.mp4')}")
            logger.info(f"   Spanish: {output_path.with_name(output_path.stem + '_ES.mp4')}")
            return True
        else:
            logger.error(f"❌ FAILED: {topic}")
            logger.error(f"   Error: {result.stderr[-500:]}")  # Last 500 chars of error
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏱️ TIMEOUT: {topic} (exceeded 30 minutes)")
        return False
    except Exception as e:
        logger.error(f"❌ ERROR: {topic} - {str(e)}")
        return False

def main():
    """Generate all videos in batch."""
    output_dir = Path(r"C:\Users\USERNAME\Videos")
    
    logger.info("=" * 80)
    logger.info("BATCH VIDEO GENERATOR - Starting")
    logger.info(f"Total topics: {len(VIDEO_TOPICS)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    
    results = {
        'success': [],
        'failed': []
    }
    
    for i, topic in enumerate(VIDEO_TOPICS, 1):
        logger.info(f"\n[{i}/{len(VIDEO_TOPICS)}] Processing: {topic}")
        
        success = generate_video(topic, output_dir)
        
        if success:
            results['success'].append(topic)
        else:
            results['failed'].append(topic)
        
        # Brief pause between videos
        if i < len(VIDEO_TOPICS):
            logger.info("Waiting 5 seconds before next video...\n")
            import time
            time.sleep(5)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("BATCH GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"✅ Successful: {len(results['success'])}/{len(VIDEO_TOPICS)}")
    for topic in results['success']:
        logger.info(f"   • {topic[:60]}")
    
    if results['failed']:
        logger.info(f"\n❌ Failed: {len(results['failed'])}/{len(VIDEO_TOPICS)}")
        for topic in results['failed']:
            logger.info(f"   • {topic[:60]}")
    
    logger.info("=" * 80)
    
    return len(results['failed']) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
