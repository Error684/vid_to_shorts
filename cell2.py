# ==========================================
# CELL 2: Batched Transcription
# ==========================================
import yt_dlp, torch, gc, os, time
from faster_whisper import WhisperModel, BatchedInferencePipeline

source_video = os.path.join(WORK_DIR, "source_video.mp4")
ydl_opts = {
    'format': 'bestvideo[vcodec^=avc1][height<=1080]+bestaudio[ext=m4a]/best[vcodec^=avc1]/best',
    'outtmpl': source_video,
    'merge_output_format': 'mp4',
    'cookiefile': '/content/cookies.txt',
    'quiet': True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    print(f"[*] Ingesting High-Definition Source...")
    ydl.download(["https://www.youtube.com/watch?v=YOUR_VIDEO_ID"]) # Replace with TARGET_URL

print("[*] Initializing Whisper Large-V3-Turbo (FP16 Batched)")
model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model)

# batch_size=24 (can be adjusted depending on VRAM usage)
segments, _ = batched_model.transcribe(source_video, batch_size=24, word_timestamps=True)

word_timestamps = []
for segment in segments:
    for word in segment.words:
        word_timestamps.append({"word": word.word.strip(), "start": word.start, "end": word.end})

print(f"[*] Transcription Complete.")
del model, batched_model
torch.cuda.empty_cache(); gc.collect()
