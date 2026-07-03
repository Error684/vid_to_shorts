# vid_to_shorts
A Google Colab stack i wrote that converts youtube URLs to production ready shorts with subtitles etc...

This project provides a high-performance Python pipeline designed for Google Colab environments equipped with NVIDIA T4 GPUs. It automates the transformation of long-form video content into vertical 9:16 shorts. The system integrates LLMs(large language models) for curation, distilled neural networks for transcription, and computer vision models for kinetic subject tracking.

## Technical Specifications

### Core Engine
    Hardware:
*   **Inference Hardware:** NVIDIA T4 (16GB VRAM).
*   
*   **Transcription:** Faster-Whisper-Large-V3-Turbo (CTranslate2).
*   **Curation Engine:** Google Gemini 2.5 Flash API.
*   **Vision Model:** YOLOv8 Medium (FP16).
*   **Signal Processing:** Savitzky-Golay Polynomial Filtering.
*   **Multimedia Backend:** FFmpeg and OpenCV (H.264/AVC).

### Key Architectural Solutions
*   **Predictive Tracking:** Utilizes Savitzky-Golay algorithms to fit third-order polynomials to subject coordinates, simulating fluid-head tripod movement and eliminating micro-stuttering.
*   **Frame-Seeking Fix:** Implements an FFmpeg Pre-Cut architecture. By isolating segments into independent files before processing, the system bypasses the "black screen" seeking errors common in OpenCV-Python.
*   **Batched Inference:** Configures a batch size of 24 for transcription to fully saturate the T4 VRAM (this still has to be improved), significantly reducing total processing time.
*   **Typography:** Employs Advanced SubStation Alpha (.ass) for word-by-word animated highlights with precise sub-second timing.

## Workflow Execution

### 1. Environment Preparation
The system requires a Google Gemini API Key, which must be stored in the Google Colab "Secrets" pane under the name `GEMINI_API_KEY`. 

### 2. Dependency Installation
Execution of the setup module installs the necessary binary runtimes, including the Deno JavaScript engine required for sandboxed YouTube signature resolution.

### 3. Transcription and Analysis
The pipeline ingests a YouTube URL, forces an H.264 stream download for compatibility, and generates a micro-timestamped transcript. Gemini AI then analyzes the transcript through a structured Pydantic schema to identify hooks and generate SEO metadata.

### 4. Visual Processing and Export
The tracking engine performs frame-by-frame inference using YOLOv8m. Subject coordinates are filtered for momentum and "stickiness" to ensure the camera remains locked on foreground subjects. The final output is multiplexed with stylized captions and the original audio.

## Directory Structure

All assets are stored in `/content/Automated_Shorts/`:

*   `source_video.mp4`: The raw high-definition source.
*   `FINAL_PREDICTIVE_SHORT_X.mp4`: The finalized vertical video.
*   `Metadata_Short_X.txt`: AI-generated titles, descriptions, and hashtags.
*   `s_X.ass`: Stylized subtitle stream files.

## Configuration Parameters

Users can modify values within the rendering module to adjust the output aesthetic:

*   **Tracking Responsiveness:** Adjust the `window_length` in the `savgol_filter`. Higher values result in slower, more cinematic panning.
*   **Subtitle Sizing:** Modify `Fontsize` within the ASS header (default 65 for 1080p).
*   **Vertical Alignment:** Adjust `MarginV` to move subtitles up or down (default 180).
*   **Encoding Quality:** Adjust the CRF (Constant Rate Factor) in the FFmpeg call. A value of 18 provides near-lossless quality.

## Disclaimer

This software is provided for automated content production. Users are responsible for ensuring compliance with the Terms of Service of content platforms and copyright regulations for the source material processed. Colab storage is ephemeral; all data is purged upon session termination.
