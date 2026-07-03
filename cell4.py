# ==========================================
# CELL 4: Processing
# ==========================================
import cv2, subprocess, os, torch, gc
import numpy as np
from scipy.signal import savgol_filter
from ultralytics import YOLO

if 'curation_data' not in locals():
    raise RuntimeError("ERROR: Run CELL 3 first.")

def format_ass_time(seconds):
    if seconds < 0: seconds = 0
    h, m = divmod(int(seconds // 60), 60)
    s, cs = divmod(int(seconds * 100 % 6000), 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

print("[*] Initializing YOLOv8 Medium in FP16")
yolo_model = YOLO('yolov8m.pt').to('cuda').half()

for clip in curation_data.shorts_batch:
    print(f"[*] Rendering : {clip.title}")

    # 1. Metadata export (.txt)
    metadata_path = os.path.join(WORK_DIR, f"Metadata_Short_{clip.short_id}.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {clip.title}\n\nDESCRIPTION:\n{clip.description}\n\nHASHTAGS: {' '.join(clip.hashtags)}")

    # 2. FFmpeg lossless pre-cut 
    temp_segment = os.path.join(WORK_DIR, f"t_seg_{clip.short_id}.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(clip.start_time), "-to", str(clip.end_time),
        "-i", source_video, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-c:a", "aac", temp_segment
    ], capture_output=True)

    # 3. Camera Tracking
    cap = cv2.VideoCapture(temp_segment)
    fps = cap.get(cv2.CAP_PROP_FPS)
    v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_x = []
    last_known_x = v_w // 2

    print(f"[*] Running AI Saliency Engine (Predictive Momentum)...")
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Inference (half=True removed to stop warning spam, model is already .half())
        results = yolo_model.predict(frame, classes=[0], verbose=False, conf=0.45)
        boxes = results[0].boxes

        if len(boxes) > 0:
            # Transfer coordinates to CPU once for math analysis
            boxes_cpu = boxes.cpu().xyxy.numpy()
            centers = (boxes_cpu[:, 0] + boxes_cpu[:, 2]) / 2

            # Anchor to the person closest to the previous center
            dist_idx = np.argmin(np.abs(centers - last_known_x))
            target_x = centers[dist_idx]

            # AI Weighting (95% Subject Lock / 5% Neutral Center Pull)
            last_known_x = int((0.95 * target_x) + (0.05 * (v_w // 2)))

        raw_x.append(last_known_x)
    cap.release()

    # 4. Savitzky-Golay polynomial smoothing 
    # This removes micro-stutters and simulates a tripod / real camera operator 
    if len(raw_x) > 31:
        smoothed_x = savgol_filter(np.array(raw_x), window_length=31, polyorder=3)
    else:
        smoothed_x = np.array(raw_x)

    # 5. Crop / Render
    temp_crop = os.path.join(WORK_DIR, f"t_crop_{clip.short_id}.mp4")
    crop_w = int(v_h * (9/16))
    out = cv2.VideoWriter(temp_crop, cv2.VideoWriter_fourcc(*'mp4v'), fps, (crop_w, v_h))
    cap = cv2.VideoCapture(temp_segment)

    for f_idx in range(len(smoothed_x)):
        ret, frame = cap.read()
        if not ret: break
        cx = int(smoothed_x[f_idx])
        # Ensure crop box never overshoots the source video resolution
        x1 = np.clip(cx - crop_w // 2, 0, v_w - crop_w)
        out.write(frame[:, x1:x1+crop_w])
    out.release(); cap.release()

    # 6. Subtitles (Arial Bold, Size 65)
    ass_path = os.path.join(WORK_DIR, f"s_{clip.short_id}.ass")
    clip_words = [w for w in word_timestamps if w['start'] >= clip.start_time and w['end'] <= clip.end_time]

    ass_header = f"[Script Info]\nScriptType: v4.00+\nPlayResX: {crop_w}\nPlayResY: {v_h}\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,65,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,4,1,2,10,10,180,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header)
        for i in range(0, len(clip_words), 2):
            chunk = clip_words[i:i+2]
            for j, hw in enumerate(chunk):
                start = format_ass_time(hw['start']-clip.start_time)
                end = format_ass_time((chunk[j+1]['start'] if j<len(chunk)-1 else hw['end'])-clip.start_time)
                # Word-by-word Yellow highlight
                txt = " ".join([f"{{\\c&H00FFFF&}}{w['word'].upper()}{{\\c&HFFFFFF&}}" if w==hw else w['word'].upper() for w in chunk])
                f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{txt}\n")

    # 7. Multiplex
    final_out = os.path.join(WORK_DIR, f"FINAL_PREDICTIVE_SHORT_{clip.short_id}.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", temp_crop, "-i", temp_segment,
        "-map", "0:v:0", "-map", "1:a:0", "-vf", f"ass={ass_path}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac", final_out
    ], capture_output=True)

    # Cleanup temp video slices
    os.remove(temp_segment); os.remove(temp_crop); os.remove(ass_path)
    print(f" Rendered: {final_out}")

print("\n" + "\nProcessing complete ! Do not forget to download !")
