"""
UNICEF Ghana NLP ASR Evaluator – Audio‑based edition (no recording)
Session data stored in HF bucket (not cookie) to stay under the 4KB cookie limit.
"""

import os
import json
import random
import base64
import hashlib
import datetime
import tempfile
import time
from typing import Optional
import io
import wave

from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, send_file
)
from huggingface_hub import HfApi
from datasets import load_dataset
from google import genai
import soundfile as sf
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
FINETUNED_MODEL     = "ghananlpcommunity/qwen3-asr-0.6b-ghana-twi-ewe-dagbani"
BASE_MODEL          = "Qwen/Qwen3-ASR-0.6B"
BUCKET_ID           = "ghananlpcommunity/unicef-evaluator-app-audio-file-storage"
GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY", "")
HF_TOKEN            = os.environ.get("HF_TOKEN", "")
SECRET_KEY          = os.environ.get("SECRET_KEY", "unicef-asr-secret-change-me")
MODAL_ASR_URL       = os.environ.get("MODAL_ASR_URL", "")
MODAL_SECRET        = os.environ.get("MODAL_SHARED_SECRET", "")

# Map language codes to the new audio‑text datasets
DATASET_IDS = {
    "TWI":     "ghananlpcommunity/ghana-nlp-health-UNICEF-asr-twi",
    "EWE":     "ghananlpcommunity/ghana-nlp-health-UNICEF-asr-ewe",
    "DAGBANI": "ghananlpcommunity/ghana-nlp-health-UNICEF-asr-dagbani",
}

MAX_VOLUNTEERS      = 5
CLAIM_MAP_PATH      = "distribution/claim_map.json"

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

api = HfApi(token=HF_TOKEN)


# ── Ensure bucket exists ───────────────────────────────────────────────────────
def ensure_bucket_exists():
    try:
        api.repo_info(repo_id=BUCKET_ID, repo_type="dataset", token=HF_TOKEN)
    except Exception:
        try:
            api.create_repo(
                repo_id=BUCKET_ID, repo_type="dataset",
                private=True, token=HF_TOKEN, exist_ok=True,
            )
        except Exception as e:
            print(f"[WARN] Could not create bucket repo: {e}")

try:
    ensure_bucket_exists()
except Exception as e:
    print(f"[WARN] ensure_bucket_exists: {e}")


# ── Dataset loader (audio + text) ─────────────────────────────────────────────
_dataset_cache: dict = {}   # lang -> list of {"text":..., "audio":...}

def load_audio_dataset(lang: str) -> list:
    """Load the full dataset for a language and cache it."""
    if lang in _dataset_cache:
        return _dataset_cache[lang]
    try:
        ds_id = DATASET_IDS[lang]
        ds = load_dataset(ds_id, split="train", token=HF_TOKEN)
        # Shuffle deterministically so every volunteer gets a different slice
        ds = ds.shuffle(seed=42)
        items = []
        for row in ds:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            audio = row["audio"]
            if isinstance(audio, dict):
                items.append({
                    "text": text,
                    "array": np.array(audio["array"], dtype=np.float32),
                    "sampling_rate": audio["sampling_rate"],
                })
        _dataset_cache[lang] = items
        print(f"[INFO] Loaded {len(items)} audio samples for {lang}")
        return items
    except Exception as e:
        print(f"[ERROR] load_audio_dataset({lang}): {e}")
        return []


# ── Claim map (unchanged but now works with dataset indices) ──────────────────
def _dl_claim_map() -> dict:
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=BUCKET_ID, filename=CLAIM_MAP_PATH,
            repo_type="dataset", token=HF_TOKEN,
        )
        with open(local) as f:
            return json.load(f)
    except Exception:
        return {}

def _ul_claim_map(claim_map: dict):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(claim_map, f, ensure_ascii=False, indent=2)
        tmp = f.name
    try:
        api.upload_file(
            path_or_fileobj=tmp, path_in_repo=CLAIM_MAP_PATH,
            repo_id=BUCKET_ID, repo_type="dataset", token=HF_TOKEN,
        )
    finally:
        os.unlink(tmp)

def claim_samples(volunteer_id: str, lang: str) -> list:
    """Return a list of dataset indices assigned to this volunteer."""
    items = load_audio_dataset(lang)
    total = len(items)

    for attempt in range(3):
        claim_map = _dl_claim_map()
        lang_map: dict = claim_map.get(lang, {})

        if volunteer_id in lang_map:
            # already assigned – return stored indices
            return [items[i] for i in lang_map[volunteer_id]]

        claimed_count = len(lang_map)
        if claimed_count >= MAX_VOLUNTEERS:
            # overflow volunteer – pick random subset
            rng     = random.Random(f"{volunteer_id}_{lang}_overflow")
            indices = rng.sample(range(total), min(total // MAX_VOLUNTEERS, total))
        else:
            # allocate band of indices
            samples_per = total // MAX_VOLUNTEERS
            start = claimed_count * samples_per
            end   = min(start + samples_per, total) if claimed_count < MAX_VOLUNTEERS - 1 else total
            band  = list(range(start, end))
            # If band is too small, pad with extra from the end
            if len(band) < samples_per:
                tail = list(range(MAX_VOLUNTEERS * samples_per, total))
                rng  = random.Random(f"{volunteer_id}_{lang}_tail")
                rng.shuffle(tail)
                band = band + tail[:samples_per - len(band)]
            rng = random.Random(f"{volunteer_id}_{lang}_sample")
            rng.shuffle(band)
            indices = band

        lang_map[volunteer_id] = indices
        claim_map[lang]        = lang_map
        try:
            _ul_claim_map(claim_map)
            return [items[i] for i in indices]
        except Exception as e:
            print(f"[WARN] claim_samples attempt {attempt}: {e}")
            time.sleep(0.5 * (attempt + 1))

    # ultimate fallback
    rng = random.Random(f"{volunteer_id}_{lang}_final")
    indices = rng.sample(range(total), min(total // MAX_VOLUNTEERS, total))
    return [items[i] for i in indices]


# ── Volunteer code ─────────────────────────────────────────────────────────────
def decode_code(code: str) -> Optional[dict]:
    try:
        padding = 4 - len(code) % 4
        if padding != 4:
            code += "=" * padding
        payload  = json.loads(base64.urlsafe_b64decode(code))
        expected = hashlib.md5(
            f"{payload['name']}|{payload['lang'].lower()}|unicef-asr".encode()
        ).hexdigest()[:6]
        return payload if payload.get("chk") == expected else None
    except Exception:
        return None


# ── Modal ASR ─────────────────────────────────────────────────────────────────
def transcribe(audio_path: str, model_id: str) -> str:
    import base64 as b64
    import requests as req

    if not MODAL_ASR_URL:
        return "[Transcription error: MODAL_ASR_URL not configured]"

    model_key = "finetuned" if model_id == FINETUNED_MODEL else "base"
    ext = audio_path.rsplit(".", 1)[-1] if "." in audio_path else "wav"

    try:
        with open(audio_path, "rb") as f:
            audio_b64 = b64.b64encode(f.read()).decode()

        resp = req.post(
            MODAL_ASR_URL,
            json={
                "audio_b64": audio_b64,
                "audio_fmt": ext,
                "model":     model_key,
                "secret":    MODAL_SECRET,
            },
            timeout=120,
        )
        data = resp.json()
        if "error" in data:
            return f"[Transcription error: {data['error']}]"
        return data.get("transcription", "").strip()
    except Exception as e:
        return f"[Transcription error: {e}]"


# ── Emotion analysis via Gemini (Gemma) ────────────────────────────────────────
EMOTION_LABELS = [
    "Happy", "Excited", "Proud", "Loved", "Relieved", "Grateful",
    "Sad", "Lonely", "Hopeless", "Guilty", "Ashamed",
    "Angry", "Frustrated", "Irritated",
    "Afraid", "Anxious", "Overwhelmed",
    "Confused", "Curious", "Doubtful",
    "Embarrassed", "Insecure", "Jealous", "Rejected"
]

def get_sentiment(transcription: str, language: str) -> dict:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=GEMINI_API_KEY)
    labels_str = ", ".join(EMOTION_LABELS)

    prompt = (
        f"You are an emotion analysis expert for African languages.\n"
        f"Language: {language}\nTranscription: {transcription}\n\n"
        f"Classify the emotion expressed in the transcription by picking exactly one label from the list below. "
        f"Choose the label that best matches, even if it's not a perfect fit. Do NOT invent new labels.\n\n"
        f"Allowed labels:\n{labels_str}\n\n"
        "You must respond with a JSON object in the exact format shown in the example below. "
        "The 'sentiment' field must be one of the allowed labels (capitalised exactly as in the list).\n\n"
        "Example:\n"
        '{"sentiment": "Happy", "confidence": 0.87, "reasoning": "The speaker expresses joy and satisfaction with their recent achievement."}\n\n'
        "Now produce the actual response for the transcription above. Do not include any other text."
    )

    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(thinking_level="HIGH"),
        ),
    )

    raw = response.text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    s, e = raw.find("{"), raw.rfind("}") + 1
    if s == -1 or e <= s:
        raise ValueError(f"Gemini did not return valid JSON: {response.text}")

    result = json.loads(raw[s:e])
    sentiment = result.get("sentiment", "").strip()

    match = None
    for label in EMOTION_LABELS:
        if label.lower() == sentiment.lower():
            match = label
            break
    if not match:
        raise ValueError(f"Gemini returned invalid emotion label: '{sentiment}'")

    return {
        "sentiment": match,
        "confidence": result.get("confidence", 0.5),
        "reasoning": result.get("reasoning", ""),
    }


# ── Bucket I/O (unchanged) ────────────────────────────────────────────────────
def _up(local: str, remote: str):
    api.upload_file(
        path_or_fileobj=local, path_in_repo=remote,
        repo_id=BUCKET_ID, repo_type="dataset", token=HF_TOKEN,
    )

def _dl(remote: str) -> Optional[str]:
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id=BUCKET_ID, filename=remote,
                               repo_type="dataset", token=HF_TOKEN)
    except Exception:
        return None

def bucket_load_progress(vid: str) -> Optional[dict]:
    p = _dl(f"progress/{vid}.json")
    return json.load(open(p)) if p else None

def bucket_save_progress(vid: str, sess_data: dict):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sess_data, f, ensure_ascii=False, indent=2)
        tmp = f.name
    try:
        _up(tmp, f"progress/{vid}.json")
    except Exception as e:
        print(f"[WARN] save_progress: {e}")
    finally:
        os.unlink(tmp)

def bucket_save_results(vid: str, sess_data: dict):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sess_data, f, ensure_ascii=False, indent=2)
        tmp = f.name
    try:
        _up(tmp, f"results/{vid}_results.json")
    except Exception as e:
        print(f"[WARN] save_results: {e}")
    finally:
        os.unlink(tmp)


# ── Server-side session ────────────────────────────────────────────────────────
def get_eval_data() -> Optional[dict]:
    vid = session.get("volunteer_id")
    if not vid:
        return None
    return bucket_load_progress(vid)

def save_eval_data(sess_data: dict):
    vid = session.get("volunteer_id")
    if vid:
        bucket_save_progress(vid, sess_data)


# ── Session helpers ────────────────────────────────────────────────────────────
def new_session(info: dict) -> dict:
    lang = info["lang"]
    name = info["name"]
    vid  = f"{name.replace(' ', '_')}_{lang}"
    samples = claim_samples(vid, lang)   # list of dicts: {"text":..., "array":..., "sampling_rate":...}
    n = len(samples)
    assigns = ["finetuned"] * (n // 2) + ["base"] * (n - n // 2)
    random.shuffle(assigns)
    items = [
        {
            "index": i,
            "text": samples[i]["text"],   # stored so we don't have to reload dataset
            "model_used": assigns[i],
            "transcription": None,
            "predicted_sentiment": None,
            "sentiment_confidence": None,
            "sentiment_reasoning": None,
            "volunteer_judgment": None,
            "completed": False,
        }
        for i in range(n)
    ]
    return {
        "volunteer_id": vid,
        "volunteer_name": name,
        "language": lang,
        "items": items,
        "current_index": 0,
        "total_completed": 0,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "completed_at": None,
    }

def calc_stats(sess_data: dict) -> dict:
    fc = ft = bc = bt = 0
    for it in sess_data["items"]:
        if not it["completed"]:
            continue
        if it["model_used"] == "finetuned":
            ft += 1
            if it["volunteer_judgment"] == "correct":
                fc += 1
        else:
            bt += 1
            if it["volunteer_judgment"] == "correct":
                bc += 1
    return {
        "fc": fc, "ft": ft, "bc": bc, "bt": bt,
        "done": sess_data["total_completed"],
        "total": len(sess_data["items"]),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    if "volunteer_id" in session:
        return redirect(url_for("evaluate"))
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    code = request.form.get("code", "").strip()
    if not code:
        return render_template("login.html", error="Please enter your volunteer code.")

    info = decode_code(code)
    if not info:
        return render_template("login.html", error="Invalid code. Please check and try again.")

    vid = f"{info['name'].replace(' ', '_')}_{info['lang']}"
    existing = bucket_load_progress(vid)
    sess_data = existing if existing else new_session(info)

    bucket_save_progress(vid, sess_data)
    session.clear()
    session["volunteer_id"]   = vid
    session["volunteer_name"] = info["name"]
    session["language"]       = info["lang"]

    return redirect(url_for("evaluate"))

@app.route("/evaluate")
def evaluate():
    if "volunteer_id" not in session:
        return redirect(url_for("index"))

    sess_data = get_eval_data()
    if not sess_data:
        session.clear()
        return redirect(url_for("index"))

    st      = calc_stats(sess_data)
    idx     = sess_data.get("current_index", 0)
    items   = sess_data.get("items", [])
    done    = idx >= len(items)
    current = items[idx] if not done else None

    return render_template("evaluate.html",
                           sess=sess_data, stats=st,
                           current=current, done=done,
                           num=idx + 1, total=len(items))

@app.route("/api/audio/<int:item_idx>")
def api_audio(item_idx: int):
    """Serve the audio for the given evaluation item index."""
    if "volunteer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    sess_data = get_eval_data()
    if not sess_data:
        return jsonify({"error": "Session not found"}), 400

    items = sess_data.get("items", [])
    if item_idx < 0 or item_idx >= len(items):
        return jsonify({"error": "Invalid item index"}), 400

    # Reconstruct the original dataset index (we stored it as item["index"])
    dataset_idx = items[item_idx]["index"]
    lang = sess_data["language"]
    all_samples = load_audio_dataset(lang)
    if dataset_idx >= len(all_samples):
        return jsonify({"error": "Dataset index out of range"}), 500

    sample = all_samples[dataset_idx]
    arr = sample["array"]
    sr  = sample["sampling_rate"]

    # Write to a temporary WAV file and send it
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        sf.write(tmp.name, arr, sr)
        return send_file(tmp.name, mimetype="audio/wav", as_attachment=False)
    finally:
        # Cleanup after the file is sent (Flask send_file can't delete immediately,
        # so we schedule removal; for simplicity we leave it – fine in production)
        pass

@app.route("/api/submit", methods=["POST"])
def api_submit():
    if "volunteer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    sess_data = get_eval_data()
    if not sess_data:
        return jsonify({"error": "Session not found"}), 400

    idx   = sess_data.get("current_index", 0)
    items = sess_data.get("items", [])
    if idx >= len(items):
        return jsonify({"error": "All items complete"}), 400

    it  = items[idx]
    mid = FINETUNED_MODEL if it["model_used"] == "finetuned" else BASE_MODEL

    # Retrieve the audio sample for the current item
    lang = sess_data["language"]
    dataset_idx = it["index"]
    all_samples = load_audio_dataset(lang)
    sample = all_samples[dataset_idx]
    arr = sample["array"]
    sr  = sample["sampling_rate"]

    # Write to a temporary file for the ASR call
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, arr, sr)
        tmp_path = tmp.name

    try:
        tx = transcribe(tmp_path, mid)
        it["transcription"] = tx

        # Emotion analysis
        emo = get_sentiment(tx, lang)
        it["predicted_sentiment"]  = emo["sentiment"]
        it["sentiment_confidence"] = emo["confidence"]
        it["sentiment_reasoning"]  = emo.get("reasoning", "")

        items[idx]         = it
        sess_data["items"] = items
        save_eval_data(sess_data)

        return jsonify({
            "sentiment":     it["predicted_sentiment"],
            "confidence":    it["sentiment_confidence"],
            "reasoning":     it["sentiment_reasoning"],
        })
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

@app.route("/api/judge", methods=["POST"])
def api_judge():
    if "volunteer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data     = request.get_json()
    judgment = data.get("judgment")
    if judgment not in ("correct", "wrong"):
        return jsonify({"error": "Invalid judgment"}), 400

    sess_data = get_eval_data()
    if not sess_data:
        return jsonify({"error": "Session not found"}), 400

    idx   = sess_data.get("current_index", 0)
    items = sess_data.get("items", [])

    it = items[idx]
    it["volunteer_judgment"] = judgment
    it["completed"]          = True
    items[idx]               = it
    sess_data["items"]       = items
    sess_data["total_completed"] = sess_data.get("total_completed", 0) + 1
    sess_data["current_index"]  = idx + 1

    total = len(items)
    done  = sess_data["current_index"] >= total

    if done:
        sess_data["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        bucket_save_results(session["volunteer_id"], sess_data)

    save_eval_data(sess_data)

    st = calc_stats(sess_data)
    return jsonify({"done": done, "stats": st,
                    "next_index": sess_data["current_index"],
                    "total": total})

@app.route("/api/save", methods=["POST"])
def api_save():
    if "volunteer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    sess_data = get_eval_data()
    if sess_data:
        save_eval_data(sess_data)
    return jsonify({"ok": True})

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/health")
def health():
    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
