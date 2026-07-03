# ==========================================
# CELL 1: ENVIRONMENT SETUP & DEPENDENCIES
# ==========================================

# 1. Install required packages (yt-dlp, faster-whisper, modern GenAI SDK, opencv, ultralytics, pydantic)
!pip install -U "yt-dlp[default]" faster-whisper google-genai opencv-python ultralytics pydantic

# 2. Install Deno to handle modern YouTube JS cipher algorithms (n-challenge)
!curl -fsSL https://deno.land/install.sh | sh

# 3. Ensure FFmpeg is installed and up-to-date on the system for video rendering
!apt-get update && apt-get install -y ffmpeg
