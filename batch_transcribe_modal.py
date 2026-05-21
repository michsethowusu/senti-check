"""
precompute_modal.py
Run on Modal:  modal run precompute_modal.py::main

1) Transcribes all audio using BOTH Qwen models (GPU) – parallel via .map
2) Predicts emotions for all transcriptions (CPU) – sequential, one by one
3) Pushes results to a private HF dataset with resume capability.
"""

import os
import json
import tempfile
from datasets import load_dataset, Dataset, Audio
import modal

# ── Configuration ─────────────────────────────────────────────────────────────
PRE_COMP_DATASET = "ghananlpcommunity/unicef-asr-precomputed"

AUDIO_DATASETS = {
    "TWI":     "ghananlpcommunity/ghana-nlp-health-UNICEF-asr-twi",
    "EWE":     "ghananlpcommunity/ghana-nlp-health-UNICEF-asr-ewe",
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
hf_secret     = modal.Secret.from_name("unicef-hf-token")
gemini_secret = modal.Secret.from_name("unicef-gemini-key")


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
    import transformers
    transformers.logging.set_verbosity_error()

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
    audio_bytes = audio_dict["bytes"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name

    try:
        txt_base = models["Qwen/Qwen3-ASR-0.6B"].transcribe(audio=path)[0].text.strip()
        txt_ft   = models["ghananlpcommunity/qwen3-asr-0.6b-ghana-twi-ewe-dagbani"].transcribe(audio=path)[0].text.strip()
        return {"transcription_base": txt_base, "transcription_ft": txt_ft}
    finally:
        os.unlink(path)


# ── CPU function: emotion prediction with Gemini (with retry) ─────────────────
@app.function(
    image=image,
    secrets=[gemini_secret],
    timeout=300,
)
def predict_emotions(transcription: str, lang: str) -> dict:
    import time
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    labels_str = ", ".join(EMOTION_LABELS)
    prompt = (
        f"Language: {lang}\nTranscription: {transcription}\n"
        f"Choose the closest emotion from: {labels_str}\n"
        "Respond ONLY with JSON: {\"sentiment\":\"...\", \"confidence\":0.xx, \"reasoning\":\"...\"}"
    )

    for attempt in range(5):
        try:
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
        except Exception as ex:
            if "429" in str(ex) and attempt < 4:
                wait = 20 * (attempt + 1)  # 20s, 40s, 60s, 80s
                print(f"    Rate limited, retrying in {wait}s (attempt {attempt+1}/5)…")
                time.sleep(wait)
            else:
                raise


# ── Orchestration ─────────────────────────────────────────────────────────────
@app.function(
    image=image,
    secrets=[hf_secret, gemini_secret],
    volumes={"/model-cache": model_volume},
    timeout=7200,
)
def main():
    # ── 1. Load source datasets ───────────────────────────────────────────────
    print("Loading source datasets…")
    all_samples = []
    for lang, ds_id in AUDIO_DATASETS.items():
        ds = load_dataset(ds_id, split="train", token=os.environ["HF_TOKEN"])
        ds = ds.cast_column("audio", Audio(decode=False))
        for i, row in enumerate(ds):
            all_samples.append((lang, i, row))
    print(f"Total samples: {len(all_samples)}")

    # ── 2. Resume: skip already processed ────────────────────────────────────
    print("Checking existing results…")
    try:
        existing_ds = load_dataset(PRE_COMP_DATASET, split="train", token=os.environ["HF_TOKEN"])
        existing_set = {(r["language"], r["original_index"]) for r in existing_ds}
        print(f"  {len(existing_set)} already processed, skipping…")
    except Exception:
        existing_set = set()

    to_process = [
        (lang, idx, row) for lang, idx, row in all_samples
        if (lang, idx) not in existing_set
    ]

    if not to_process:
        print("✅ All samples already processed.")
        return

    print(f"Will process {len(to_process)} samples…")

    # ── 3. Transcribe all (GPU, parallel) ────────────────────────────────────
    print("Step 1/2: Transcribing with both models (GPU)…")
    audio_inputs = [row["audio"] for (_, _, row) in to_process]
    transcriptions = list(transcribe_both.map(audio_inputs, return_exceptions=True))

    successes = sum(1 for t in transcriptions if not isinstance(t, Exception))
    print(f"Transcription done: {successes}/{len(transcriptions)} succeeded")

    # ── 4. Save transcriptions to volume immediately (safety net) ────────────
    valid_trans = []
    for (lang, idx, row), trans in zip(to_process, transcriptions):
        if isinstance(trans, Exception):
            print(f"  Transcription failed for {lang} #{idx}: {trans}")
            continue
        valid_trans.append((lang, idx, row, trans))

    backup_path = "/model-cache/transcriptions_backup.json"
    try:
        with open(backup_path) as f:
            existing_backup = json.load(f)
    except FileNotFoundError:
        existing_backup = []

    new_backup_entries = [
        {
            "language": lang,
            "original_index": idx,
            "text": str(row.get("text", "")).strip(),
            "transcription_base": trans["transcription_base"],
            "transcription_ft":   trans["transcription_ft"],
        }
        for (lang, idx, row, trans) in valid_trans
    ]
    existing_backup.extend(new_backup_entries)

    with open(backup_path, "w") as f:
        json.dump(existing_backup, f, ensure_ascii=False, indent=2)
    model_volume.commit()
    print(f"💾 {len(new_backup_entries)} transcriptions saved to volume → {backup_path}")

    # ── 5. Predict emotions (sequential, one by one) ──────────────────────────
    print("Step 2/2: Predicting emotions (sequential)…")
    total_calls = len(valid_trans) * 2
    sentiments = []

    for i, (lang, idx, row, trans) in enumerate(valid_trans):
        # base model transcription
        print(f"  [{i*2+1}/{total_calls}] {lang} #{idx} — base…")
        try:
            s_base = predict_emotions.remote(trans["transcription_base"], lang)
        except Exception as ex:
            print(f"    ❌ base failed: {ex}")
            s_base = None

        # fine-tuned model transcription
        print(f"  [{i*2+2}/{total_calls}] {lang} #{idx} — ft…")
        try:
            s_ft = predict_emotions.remote(trans["transcription_ft"], lang)
        except Exception as ex:
            print(f"    ❌ ft failed: {ex}")
            s_ft = None

        sentiments.append((s_base, s_ft))

    # ── 6. Assemble final rows ────────────────────────────────────────────────
    new_rows = []
    for (lang, idx, row, trans), (s_base, s_ft) in zip(valid_trans, sentiments):
        new_rows.append({
            "language":         lang,
            "original_index":   idx,
            "text":             str(row.get("text", "")).strip(),
            "transcription_base":  trans["transcription_base"],
            "sentiment_base":      s_base.get("sentiment")   if s_base else None,
            "confidence_base":     s_base.get("confidence")  if s_base else None,
            "reasoning_base":      s_base.get("reasoning")   if s_base else None,
            "transcription_ft":    trans["transcription_ft"],
            "sentiment_ft":        s_ft.get("sentiment")     if s_ft else None,
            "confidence_ft":       s_ft.get("confidence")    if s_ft else None,
            "reasoning_ft":        s_ft.get("reasoning")     if s_ft else None,
        })

    # ── 7. Push to HF ─────────────────────────────────────────────────────────
    if new_rows:
        print(f"Pushing {len(new_rows)} rows to {PRE_COMP_DATASET}…")
        Dataset.from_list(new_rows).push_to_hub(
            PRE_COMP_DATASET,
            token=os.environ["HF_TOKEN"],
            private=True,
        )
        print(f"✅ Done — {len(new_rows)} rows pushed")
    else:
        print("⚠️ No rows to push — check errors above")
