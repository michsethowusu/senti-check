"""
precompute_modal.py
Run on Modal:  modal run precompute_modal.py

1) Transcribes all audio using BOTH Qwen models (GPU) – parallel via .map
2) Predicts emotions for all transcriptions (CPU) – parallel via .map
3) Pushes results to a private HF dataset with resume capability.
"""

import os
import json
import numpy as np
import soundfile as sf
import tempfile
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
import modal

load_dotenv()   # local .env (only for non‑secret vars like PRE_COMP_DATASET)

# ── Configuration (non‑secret) ────────────────────────────────────────────────
PRE_COMP_DATASET = os.environ.get("PRE_COMP_DATASET", "ghananlpcommunity/unicef-asr-precomputed")

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

# Shared image (GPU + CPU)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0", "torchaudio>=2.2.0",
        "transformers>=4.41.0", "accelerate>=0.30.0",
        "soundfile", "librosa", "qwen-asr",
        "google-genai", "datasets", "huggingface-hub",
    )
)

# Volume where model weights are cached (same one your service uses)
model_volume = modal.Volume.from_name("unicef-asr-models", create_if_missing=True)

# Secrets (must exist in Modal dashboard)
hf_secret = modal.Secret.from_name("unicef-hf-token")         # contains HF_TOKEN
gemini_secret = modal.Secret.from_name("unicef-gemini-key")   # contains GEMINI_API_KEY


# ── GPU function: transcribe one audio with both models ───────────────────────
@app.function(
    image=image,
    gpu="any",                     # uses T4/A100 as available
    volumes={"/model-cache": model_volume},
    secrets=[hf_secret],
    timeout=120,
)
def transcribe_both(audio_dict: dict) -> dict:
    """
    Input: HF audio dict with 'array' and 'sampling_rate'.
    Returns dict with 'transcription_base' and 'transcription_ft'.
    Models are loaded once per container (lazy) and cached on volume.
    """
    import torch
    from qwen_asr import Qwen3ASRModel

    # Lazy load both models (only on first call in this container)
    if not hasattr(transcribe_both, "models"):
        hf_token = os.environ["HF_TOKEN"]
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16

        transcribe_both.models = {}
        for model_id in [
            "Qwen/Qwen3-ASR-0.6B",
            "ghananlpcommunity/qwen3-asr-0.6b-ghana-twi-ewe-dagbani",
        ]:
            model = Qwen3ASRModel.from_pretrained(
                model_id,
                dtype=dtype,
                device_map="cuda:0",
                token=hf_token,
            )
            transcribe_both.models[model_id] = model

    models = transcribe_both.models
    arr = np.array(audio_dict["array"], dtype=np.float32)
    sr = audio_dict["sampling_rate"]

    # qwen-asr requires a file path, so we write a temporary WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, arr, sr)
        path = tmp.name

    try:
        txt_base = models["Qwen/Qwen3-ASR-0.6B"].transcribe(audio=path)[0].text.strip()
        txt_ft   = models["ghananlpcommunity/qwen3-asr-0.6b-ghana-multilang"].transcribe(audio=path)[0].text.strip()
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
    """
    Calls Gemini to pick the closest emotion label.
    Returns dict with 'sentiment', 'confidence', 'reasoning'.
    """
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


# ── Local entrypoint: orchestrates everything ─────────────────────────────────
@app.local_entrypoint()
def main():
    import time
    from tqdm import tqdm

    # 1. Load all source datasets and flatten to list of (lang, idx, row)
    print("Loading source datasets…")
    all_samples = []
    for lang, ds_id in AUDIO_DATASETS.items():
        ds = load_dataset(ds_id, split="train", token=os.environ["HF_TOKEN"])
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
    audio_inputs = [row["audio"] for (_, _, row) in to_process]
    # .map automatically parallelizes across multiple GPU containers
    transcriptions = list(transcribe_both.map(audio_inputs, return_exceptions=True))

    # Prepare data for stage 2 – collect (transcription_base, lang, idx, transcription_ft)
    sentiment_tasks = []
    for (lang, idx, row), trans in zip(to_process, transcriptions):
        if isinstance(trans, Exception):
            print(f"Transcription failed for {lang} #{idx}: {trans}")
            continue
        # We'll predict emotion for both transcripts
        sentiment_tasks.append((trans["transcription_base"], lang, idx, "base"))
        sentiment_tasks.append((trans["transcription_ft"], lang, idx, "ft"))

    if not sentiment_tasks:
        print("No transcriptions succeeded. Exiting.")
        return

    # 4. Stage 2: Predict emotions (CPU) in parallel
    print("Step 2/2: Predicting emotions (CPU)…")
    # Prepare inputs for starmap
    starmap_inputs = [(txt, lang) for (txt, lang, _, _) in sentiment_tasks]
    sentiments = list(predict_emotions.starmap(starmap_inputs, return_exceptions=True))

    # 5. Assemble final results
    new_rows = []
    # We need to match sentiments back to samples
    # Create mapping from original samples to their base/ft sentiment indices
    sample_to_sent = {}   # (lang, idx) -> {"base": sentiment_result, "ft": sentiment_result}
    sent_idx = 0
    for (lang, idx, row), trans in zip(to_process, transcriptions):
        if isinstance(trans, Exception):
            continue
        if sent_idx + 1 >= len(sentiments):
            break
        s_base = sentiments[sent_idx]
        s_ft   = sentiments[sent_idx+1]
        sent_idx += 2

        # Skip if either sentiment failed
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
