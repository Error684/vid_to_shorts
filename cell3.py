# ==========================================
# CELL 3: Batch Curation (using Gemini)
# ==========================================
import json
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types
from google.colab import userdata

# 1. Pydantic Target Schema for Structured Output + Metadata
class ShortClip(BaseModel):
    short_id: int
    start_time: float = Field(description="Clip window start timestamp in seconds")
    end_time: float = Field(description="Clip window end timestamp in seconds")
    reason: str = Field(description="Deep virality retention hook explanation")
    title: str = Field(description="A highly engaging, click-worthy YouTube Shorts title (under 60 characters)")
    description: str = Field(description="An engaging, algorithm-optimized short description summarizing the clip")
    hashtags: List[str] = Field(description="A list of 3-10 highly relevant viral hashtags (including #shorts)")

class ShortsBatch(BaseModel):
    shorts_batch: List[ShortClip]

# 2. Authenticate / prepare GenAI client
try:
    api_key = userdata.get('GEMINI_API_KEY')
except Exception:
    raise RuntimeError("[!] GEMINI_API_KEY missing! Add it to the Colab Secrets pane (key icon on the left sidebar).")

client = genai.Client(api_key=api_key)

# 3. Format transcript for LLM context window
transcript_text = ""
current_line = []
if word_timestamps:
    line_start = word_timestamps[0]['start']
    for w in word_timestamps:
        current_line.append(w['word'])
        if w['end'] - line_start > 15.0 or w == word_timestamps[-1]:
            transcript_text += f"[{line_start:.1f}s - {w['end']:.1f}s] {' '.join(current_line)}\n"
            if w != word_timestamps[-1]:
                line_start = w['end']
                current_line = []

prompt = """
You are an elite TikTok/YouTube Shorts retention algorithm and editing strategist.
Analyze the following video transcript. Extract up to 5 distinct, highly viral standalone segments.
Each segment MUST be strictly between 30 and 60 seconds long.
Prioritize hooks, emotional spikes, or high-value action/knowledge gaps. Generate optimized YouTube metadata for each.

Transcript Timeline:
"""

print("[*] Connecting to Gemini API for Batch Curation...")
response = client.models.generate_content(
    model='gemini-2.5-flash', #model can be edited depending on token usage / quality and speed
    contents=prompt + transcript_text,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ShortsBatch,
        temperature=0.4
    )
)

# 4. Parse the structured response
curation_data = ShortsBatch.model_validate_json(response.text)

print(f"\n[*] CURATION SUCCESS! Gemini isolated {len(curation_data.shorts_batch)} viral short(s):\n")
for clip in curation_data.shorts_batch:
    print(f"🎬 Short #{clip.short_id} | {clip.start_time}s to {clip.end_time}s")
    print(f"   ↳ Title: {clip.title}")
    print(f"   ↳ Hashtags: {' '.join(clip.hashtags)}")
    print(f"   ↳ Hook Reason: {clip.reason}\n")
