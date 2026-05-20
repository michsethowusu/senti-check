"""
precompute.py
Run ONCE to transcribe all audio samples using both Qwen models (via Modal)
and predict emotions with Gemini. Results are pushed to a HF dataset.

Usage:   python precompute.py
Make sure your .env file contains:
    HF_TOKEN, MODAL_ASR_URL, MODAL_SHARED_SECRET, GEMINI_API_KEY
"""

import os
import json
import time
import base64
import requests
import tempfile
import numpy as np
import soundfile as sf
from datasets import load_dataset, Dataset
from google import genai
from dotenv import load_dotenv

load_dotenv()  # reads the .env file into os.environ

# ── Config from environment ───────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set in .env or environment")

MODAL_URL     = os.environ.get("MODAL_ASR_URL")
MODAL_SECRET  = os.environ.get("MODAL_SHARED_SECRET")
GEMINI_KEY    = os.environ.get("GEMINI_API_KEY")
if not all([MODAL_URL, MODAL_SECRET, GEMINI_KEY]):
    raise RuntimeError("Missing one of MODAL_ASR_URL, MODAL_SHARED_SECRET, GEMINI_API_KEY")

PRE_COMP_DATASET = "ghananlpcommunity/unicef-asr-precomputed"  # change to your repo

EMOTION_LABELS = [
    "Happy", "Excited", "Proud", "Loved", "Relieved", "Grateful",
    "Sad", "Lonely", "Hopeless", "Guilty", "Ashamed",
    "Angry", "Frustrated", "Irritated",
    "Afraid", "Anxious", "Overwhelmed",
    "Confused", "Curious", "Doubtful",
    "Embarrassed", "Insecure", "Jealous", "Rejected"
]

AUDIO_DATASETS = {
    "TWI": "ghananlpcommunity/ghana-nlp-health-UNICEF-asr-twi",
    "EWE": "ghananlpcommunity/ghana-nlp-health-UNICEF-asr-ewe",
    "DAGBANI": "ghananlpcommunity/ghana-nlp-health-UNICEF-asr-dagbani",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def transcribe_modal(audio_path: str, model: str) -> str:
    """
    Calls your Modal ASR endpoint.
    model = 'base'   --> Qwen/Qwen3-ASR-0.6B
    model = 'finetuned' --> ghananlpcommunity/qwen3-asr-0.6b-ghana-multilang
    """
    ext = audio_path.rsplit(".", 1)[-1]
    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = requests.post(
        MODAL_URL,
        json={
            "audio_b64": b64,
            "audio_fmt": ext,
            "model": model,
            "secret": MODAL_SECRET,
        },
        timeout=120,
    )
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Modal error: {data['error']}")
    return data["transcription"].strip()

def get_sentiment(text: str, lang: str) -> dict:
    client = genai.Client(api_key=GEMINI_KEY)
    labels_str = ", ".join(EMOTION_LABELS)
    prompt = (
        f"Language: {lang}\nTranscription: {text}\n"
        f"Choose the closest emotion from: {labels_str}\n"
        "Respond ONLY with JSON: {\"sentiment\":\"...\", \"confidence\":0.xx, \"reasoning\":\"...\"}"
    )
    resp = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(thinking_level="HIGH"),
        ),
    )
    raw = resp.text.strip()
    if "```" in raw:
        raw = raw.split("```")[1].replace("json","")
    s, e = raw.find("{"), raw.rfind("}") + 1
    return json.loads(raw[s:e])

# ── Process all languages ─────────────────────────────────────────────────────
def process_language(lang: str):
    print(f"Processing {lang}...")
    ds = load_dataset(AUDIO_DATASETS[lang], split="train", token=HF_TOKEN)
    rows = []
    for i, row in enumerate(ds):
        text = str(row["text"]).strip()
        arr = np.array(row["audio"]["array"], dtype=np.float32)
        sr = row["audio"]["sampling_rate"]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, arr, sr)
            path = tmp.name
        try:
            # 1. Transcription with BOTH Qwen models
            txt_base = transcribe_modal(path, "base")
            txt_ft   = transcribe_modal(path, "finetuned")

            # 2. Emotion prediction on each transcription
            s_base = get_sentiment(txt_base, lang)
            s_ft   = get_sentiment(txt_ft, lang)

            rows.append({
                "language": lang,
                "original_index": i,
                "text": text,
                "transcription_base": txt_base,
                "sentiment_base": s_base["sentiment"],
                "confidence_base": s_base["confidence"],
                "reasoning_base": s_base["reasoning"],
                "transcription_ft": txt_ft,
                "sentiment_ft": s_ft["sentiment"],
                "confidence_ft": s_ft["confidence"],
                "reasoning_ft": s_ft["reasoning"],
            })
            print(f"  {i+1}/{len(ds)} done")
        finally:
            os.unlink(path)
        time.sleep(0.2)   # be gentle to the API
    return rows

if __name__ == "__main__":
    all_rows = []
    for lang in AUDIO_DATASETS:
        all_rows.extend(process_language(lang))
    Dataset.from_list(all_rows).push_to_hub(PRE_COMP_DATASET, token=HF_TOKEN, private=True)
    print(f"✅ Precomputed dataset pushed to {PRE_COMP_DATASET}")
