"""
precompute_modal.py
Run on Modal:  modal run precompute_modal.py::main

1) Transcribes all audio using BOTH Qwen models (GPU) – parallel via .map
2) Predicts emotions for all transcriptions (CPU) – parallel via .map
3) Pushes results to a private HF dataset with resume capability.
"""

import os
import json
import tempfile
from datasets import load_dataset, Dataset, Audio
import modal

# ── Configuration (non‑secret) ────────────────────────────────────────────────
PRE_COMP_DATASET = "ghananlpcommunity/unicef-asr-precomputed"

AUDIO_DATASETS = {
    "TWI": "ghananlpcommunity/ghana-nlp-health-UNICEF-asr-twi",
    "EWE": "ghananlpcommunity/ghana-nlp-health-UNICEF-asr-ewe",
    "DAGBANI": "ghananlpcommunity/ghana-nlp-health-UNICEF-asr-dagbani",
}

EMOTION_LABELS = [
    "Happy","Excited","Proud","Loved","Relieved","Grateful",
    "Sad","Lonely","Hopeless","Guilty","Ashamed",
    "Angry","Frustrated","Irritated",
    "Afraid","Anxious","Overwhelmed",
    "Confused","Curious","Doubtful",
    "Embarrassed","Insecure","Jealous","Rejected"
]

# ── Modal app & resources ─────────────────────────────────────────────────────
app = modal.App("unicef-asr-precompute")

# Shared image – no torchcodec (we load audio bytes directly)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0", "torchaudio>=2.2.0",
        "transformers>=4.41.0", "accelerate>=0.30.0",
        "soundfile", "librosa", "qwen-asr",
        "google-genai", "datasets", "huggingface-hub",
    )
)

model_volume = modal.Volume.from_name("unicef-asr-models", create_if_missing=True)

hf_secret = modal.Secret.from_name("unicef-hf-token")         # HF_TOKEN
gemini_secret = modal.Secret.from_name("unicef-gemini-key")   # GEMINI_API_KEY


# ── GPU function: transcribe one audio with both models ───────────────────────
@app.function(
    image=image,
    gpu="any",
    volumes={"/model-cache": model_volume},
    secrets=[hf_secret],
    timeout=120,
)
def transcribe_both(audio_dict: dict) -> dict:
    import torch
    from qwen_asr import Qwen3ASRModel

    if not hasattr(transcribe_both, "models"):
        hf_token = os.environ["HF_TOKEN"]
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16

        transcribe_both.models = {}
        for model_id in [
            "Qwen/Qwen3-ASR-0.6B",
            "ghananlpcommunity/qwen3-asr-0.6b-ghana-twi-ewe-dagbani",
        ]:
            transcribe_both.models[model_id] = Qwen3ASRModel.from_pretrained(
                model_id, dtype=dtype, device_map="cuda:0", token=hf_token,
            )

    models = transcribe_both.models
    audio_bytes = audio_dict["bytes"]  # decode=False gives {"bytes": ..., "path": ...}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name

    try:
        txt_base = models["Qwen/Qwen3-ASR-0.6B"].transcribe(audio=path)[0].text.strip()
        txt_ft   = models["ghananlpcommunity/qwen3-asr-0.6b-ghana-twi-ewe-dagbani"].transcribe(audio=path)[0].text.strip()
        return {"transcription_base": txt_base, "transcription_ft": txt_ft}
    finally:
        os.unlink(path)


# ── CPU function: emotion prediction with Gemini ──────────────────────────────
@app.function(
    image=image,
    secrets=[gemini_secret],
    timeout=30,
)
def predict_emotions(transcription: str, lang: str) -> dict:
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    labels_str = ", ".join(EMOTION_LABELS)
    prompt = (
        f"Language: {lang}\nTranscription: {transcription}\n"
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
        raw = raw.split("```")[1].replace("json", "")
    s, e = raw.find("{"), raw.rfind("}") + 1
    return json.loads(raw[s:e])


# ── Orchestration function (with secrets) ─────────────────────────────────────
@app.function(
    image=image,
    secrets=[hf_secret, gemini_secret],
    timeout=3600,
)
def main():
    from tqdm import tqdm

    # 1. Load all source datasets – decode=False to get raw bytes
    print("Loading source datasets (decode=False)…")
    all_samples = []
    for lang, ds_id in AUDIO_DATASETS.items():
        ds = load_dataset(ds_id, split="train", token=os.environ["HF_TOKEN"])
        # Cast the audio column to raw bytes (no torchcodec needed)
        ds = ds.cast_column("audio", Audio(decode=False))
        for i, row in enumerate(ds):
            all_samples.append((lang, i, row))

    # 2. Resume: skip samples already in the precomputed dataset
    print("Checking existing results…")
    try:
        existing_ds = load_dataset(PRE_COMP_DATASET, split="train", token=os.environ["HF_TOKEN"])
        existing_set = {(r["language"], r["original_index"]) for r in existing_ds}
    except Exception:
        existing_set = set()

    to_process = [(lang, idx, row) for lang, idx, row in all_samples
                  if (lang, idx) not in existing_set]

    if not to_process:
        print("✅ All samples already processed. Nothing to do.")
        return

    print(f"Processing {len(to_process)} new samples…")

    # 3. Stage 1: Transcribe all (GPU) in parallel
    print("Step 1/2: Transcribing with both models (GPU)…")
    # Each row's audio is now a dict with 'bytes' + 'sampling_rate'
    audio_inputs = [row["audio"] for (_, _, row) in to_process]
    transcriptions = list(transcribe_both.map(audio_inputs, return_exceptions=True))

    # 4. Prepare sentiment tasks
    sentiment_tasks = []
    for (lang, idx, row), trans in zip(to_process, transcriptions):
        if isinstance(trans, Exception):
            print(f"Transcription failed for {lang} #{idx}: {trans}")
            continue
        sentiment_tasks.append((trans["transcription_base"], lang, idx, "base"))
        sentiment_tasks.append((trans["transcription_ft"], lang, idx, "ft"))

    if not sentiment_tasks:
        print("No transcriptions succeeded. Exiting.")
        return

    # 5. Stage 2: Predict emotions (CPU) in parallel
    print("Step 2/2: Predicting emotions (CPU)…")
    starmap_inputs = [(txt, lang) for (txt, lang, _, _) in sentiment_tasks]
    sentiments = list(predict_emotions.starmap(starmap_inputs, return_exceptions=True))

    # 6. Assemble final results
    new_rows = []
    sent_idx = 0
    for (lang, idx, row), trans in zip(to_process, transcriptions):
        if isinstance(trans, Exception):
            continue
        if sent_idx + 1 >= len(sentiments):
            break
        s_base = sentiments[sent_idx]
        s_ft   = sentiments[sent_idx+1]
        sent_idx += 2

        if isinstance(s_base, Exception):
            print(f"Sentiment (base) failed for {lang} #{idx}: {s_base}")
            continue
        if isinstance(s_ft, Exception):
            print(f"Sentiment (ft) failed for {lang} #{idx}: {s_ft}")
            continue

        new_rows.append({
            "language": lang,
            "original_index": idx,
            "text": str(row["text"]).strip(),
            "transcription_base": trans["transcription_base"],
            "sentiment_base": s_base["sentiment"],
            "confidence_base": s_base["confidence"],
            "reasoning_base": s_base["reasoning"],
            "transcription_ft": trans["transcription_ft"],
            "sentiment_ft": s_ft["sentiment"],
            "confidence_ft": s_ft["confidence"],
            "reasoning_ft": s_ft["reasoning"],
        })

    if new_rows:
        print(f"Pushing {len(new_rows)} rows to {PRE_COMP_DATASET}…")
        Dataset.from_list(new_rows).push_to_hub(
            PRE_COMP_DATASET,
            token=os.environ["HF_TOKEN"],
            private=True,
        )
        print("✅ Done")
    else:
        print("✅ No new rows to push")
