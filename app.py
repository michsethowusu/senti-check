"""
UNICEF Ghana NLP ASR Evaluator – Flask edition
Compatible with Render (free tier) – deploy via GitHub.
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

from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for
)
from huggingface_hub import HfApi

# ── Config ────────────────────────────────────────────────────────────────────
FINETUNED_MODEL     = "ghananlpcommunity/qwen3-asr-0.6b-ghana-multilang"
BASE_MODEL          = "Qwen/Qwen3-ASR-0.6B"
BUCKET_ID           = "michsethowusu/unicef-evaluator-app-audio-file-storage"
NVIDIA_API_KEY      = os.environ.get("NVIDIA_API_KEY", "")
HF_TOKEN            = os.environ.get("HF_TOKEN", "")
SECRET_KEY          = os.environ.get("SECRET_KEY", "unicef-asr-secret-change-me")

DATASET_IDS = {
    "TWI":     "ghananlpcommunity/youth-conversations-tw",
    "EWE":     "ghananlpcommunity/youth-conversations-ee",
    "DAGBANI": "ghananlpcommunity/youth-conversations-dag",
}

TEXTS_PER_VOLUNTEER = 100
MAX_VOLUNTEERS      = 5
CLAIM_MAP_PATH      = "distribution/claim_map.json"

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB uploads

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


# ── Dataset loader ─────────────────────────────────────────────────────────────
_sentence_cache: dict = {}

def load_sentences(lang: str) -> list:
    if lang in _sentence_cache:
        return _sentence_cache[lang]
    try:
        from datasets import load_dataset
        ds_id = DATASET_IDS[lang]
        ds    = load_dataset(ds_id, split="train", token=HF_TOKEN)
        sents = [str(r["text"]).strip() for r in ds if str(r.get("text", "")).strip()]
    except Exception as e:
        print(f"[ERROR] load_sentences({lang}): {e}")
        sents = [f"Dataset load failed – check HF_TOKEN and dataset access: {e}"]
    rng = random.Random(42)
    rng.shuffle(sents)
    _sentence_cache[lang] = sents
    print(f"[INFO] Loaded {len(sents)} sentences for {lang}")
    return sents


# ── Claim map ──────────────────────────────────────────────────────────────────
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

def claim_sentences(volunteer_id: str, lang: str) -> list:
    sentences = load_sentences(lang)
    total     = len(sentences)
    band_size = total // MAX_VOLUNTEERS

    for attempt in range(3):
        claim_map = _dl_claim_map()
        lang_map: dict = claim_map.get(lang, {})

        if volunteer_id in lang_map:
            return [sentences[i % total] for i in lang_map[volunteer_id]]

        claimed_count = len(lang_map)
        if claimed_count >= MAX_VOLUNTEERS:
            rng     = random.Random(f"{volunteer_id}_{lang}_overflow")
            indices = rng.sample(range(total), min(TEXTS_PER_VOLUNTEER, total))
        else:
            band_start = claimed_count * band_size
            band_end   = min(band_start + band_size, MAX_VOLUNTEERS * band_size)
            band       = list(range(band_start, band_end))
            if len(band) < TEXTS_PER_VOLUNTEER:
                tail = list(range(MAX_VOLUNTEERS * band_size, total))
                rng  = random.Random(f"{volunteer_id}_{lang}_tail")
                rng.shuffle(tail)
                band = band + tail[:TEXTS_PER_VOLUNTEER - len(band)]
            rng = random.Random(f"{volunteer_id}_{lang}_sample")
            rng.shuffle(band)
            indices = band[:TEXTS_PER_VOLUNTEER]

        lang_map[volunteer_id] = indices
        claim_map[lang]        = lang_map
        try:
            _ul_claim_map(claim_map)
            return [sentences[i % total] for i in indices]
        except Exception as e:
            print(f"[WARN] claim_sentences attempt {attempt}: {e}")
            time.sleep(0.5 * (attempt + 1))

    return [sentences[i % total] for i in indices]


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


# ── Modal ASR (GPU inference via Modal web endpoint) ───────────────────────────
MODAL_ASR_URL   = os.environ.get("MODAL_ASR_URL", "")   # set in Render env vars
MODAL_SECRET    = os.environ.get("MODAL_SHARED_SECRET", "")

def transcribe(audio_path: str, model_id: str) -> str:
    """
    Send audio to the Modal GPU endpoint for transcription.
    model_id is the full HF model string; we map it to "finetuned" / "base"
    so the Modal service knows which loaded pipeline to use.
    """
    import base64 as b64
    import requests as req

    if not MODAL_ASR_URL:
        return "[Transcription error: MODAL_ASR_URL not configured]"

    model_key = "finetuned" if model_id == FINETUNED_MODEL else "base"
    ext = audio_path.rsplit(".", 1)[-1] if "." in audio_path else "webm"

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


# ── Sentiment via NVIDIA ───────────────────────────────────────────────────────
def get_sentiment(transcription: str, language: str) -> dict:
    import requests as req
    if not NVIDIA_API_KEY:
        return {"sentiment": "neutral", "confidence": 0.5, "reasoning": "No NVIDIA_API_KEY set."}
    prompt = (
        f"You are a sentiment analysis expert for African languages.\n"
        f"Language: {language}\nTranscription: {transcription}\n\n"
        "Analyse the sentiment. Respond ONLY with a JSON object with keys:\n"
        "'sentiment' (positive/negative/neutral), 'confidence' (0.0–1.0), "
        "'reasoning' (1 sentence in English)."
    )
    try:
        r   = req.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"},
            json={"model": "deepseek-ai/deepseek-r1",
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 200, "temperature": 0.1},
            timeout=30,
        )
        txt = r.json()["choices"][0]["message"]["content"]
        txt = txt.replace("```json", "").replace("```", "").strip()
        s, e = txt.find("{"), txt.rfind("}") + 1
        return json.loads(txt[s:e])
    except Exception as ex:
        return {"sentiment": "neutral", "confidence": 0.5, "reasoning": str(ex)}


# ── Bucket I/O ─────────────────────────────────────────────────────────────────
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

def bucket_upload_audio(vid: str, idx: int, path: str):
    try:
        _up(path, f"audio/{vid}/item_{idx:03d}.wav")
    except Exception as e:
        print(f"[WARN] upload_audio: {e}")


# ── Session helpers ────────────────────────────────────────────────────────────
def new_session(info: dict) -> dict:
    lang = info["lang"]
    name = info["name"]
    vid  = f"{name.replace(' ', '_')}_{lang}"
    sents = claim_sentences(vid, lang)
    n     = len(sents)
    assigns = ["finetuned"] * (n // 2) + ["base"] * (n - n // 2)
    random.shuffle(assigns)
    items = [
        {
            "index": i, "text": sents[i], "model_used": assigns[i],
            "transcription": None, "predicted_sentiment": None,
            "sentiment_confidence": None, "volunteer_judgment": None,
            "audio_uploaded": False, "completed": False,
        }
        for i in range(n)
    ]
    return {
        "volunteer_id": vid, "volunteer_name": name, "language": lang,
        "items": items, "current_index": 0, "total_completed": 0,
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

    vid      = f"{info['name'].replace(' ', '_')}_{info['lang']}"
    existing = bucket_load_progress(vid)
    sess_data = existing if existing else new_session(info)

    # Store in Flask session
    session["volunteer_id"]   = vid
    session["volunteer_name"] = info["name"]
    session["language"]       = info["lang"]
    session["eval_data"]      = sess_data   # full state in cookie (encrypted)

    return redirect(url_for("evaluate"))

@app.route("/evaluate")
def evaluate():
    if "volunteer_id" not in session:
        return redirect(url_for("index"))
    sess_data = session.get("eval_data", {})
    st        = calc_stats(sess_data)
    idx       = sess_data.get("current_index", 0)
    items     = sess_data.get("items", [])
    done      = idx >= len(items)
    current   = items[idx] if not done else None
    return render_template("evaluate.html",
                           sess=sess_data, stats=st,
                           current=current, done=done,
                           num=idx + 1, total=len(items))

@app.route("/api/submit", methods=["POST"])
def api_submit():
    """Receives audio blob, transcribes, runs sentiment."""
    if "volunteer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio received"}), 400

    sess_data = session.get("eval_data", {})
    idx       = sess_data.get("current_index", 0)
    items     = sess_data.get("items", [])
    if idx >= len(items):
        return jsonify({"error": "All items complete"}), 400

    it  = items[idx]
    mid = FINETUNED_MODEL if it["model_used"] == "finetuned" else BASE_MODEL

    # Save audio to temp file
    suffix = ".webm"
    if audio_file.filename and "." in audio_file.filename:
        suffix = "." + audio_file.filename.rsplit(".", 1)[-1]

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        tx = transcribe(tmp_path, mid)
        it["transcription"] = tx

        # Upload audio to bucket (fire-and-forget)
        bucket_upload_audio(session["volunteer_id"], idx, tmp_path)
        it["audio_uploaded"] = True

        sv = get_sentiment(tx, sess_data["language"])
        it["predicted_sentiment"]  = sv.get("sentiment", "neutral")
        it["sentiment_confidence"] = sv.get("confidence", 0.5)
        it["sentiment_reasoning"]  = sv.get("reasoning", "")

        items[idx] = it
        sess_data["items"] = items
        session["eval_data"] = sess_data
        session.modified = True

        return jsonify({
            "transcription": tx,
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
    """Records volunteer judgment (correct/wrong) and advances."""
    if "volunteer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data      = request.get_json()
    judgment  = data.get("judgment")  # "correct" | "wrong"
    if judgment not in ("correct", "wrong"):
        return jsonify({"error": "Invalid judgment"}), 400

    sess_data = session.get("eval_data", {})
    idx       = sess_data.get("current_index", 0)
    items     = sess_data.get("items", [])

    it = items[idx]
    it["volunteer_judgment"] = judgment
    it["completed"]          = True
    items[idx]               = it
    sess_data["items"]       = items
    sess_data["total_completed"] = sess_data.get("total_completed", 0) + 1
    sess_data["current_index"]  = idx + 1

    bucket_save_progress(session["volunteer_id"], sess_data)

    total = len(items)
    done  = sess_data["current_index"] >= total

    if done:
        sess_data["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        bucket_save_results(session["volunteer_id"], sess_data)

    session["eval_data"] = sess_data
    session.modified = True

    st = calc_stats(sess_data)
    return jsonify({"done": done, "stats": st,
                    "next_index": sess_data["current_index"],
                    "total": total})

@app.route("/api/save", methods=["POST"])
def api_save():
    if "volunteer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    sess_data = session.get("eval_data", {})
    bucket_save_progress(session["volunteer_id"], sess_data)
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
