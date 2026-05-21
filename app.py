"""
UNICEF Ghana NLP ASR Evaluator – Precomputed emotions (local JSON)
Volunteers see the original text + pre‑predicted emotion and judge.
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
BUCKET_ID           = "ghananlpcommunity/unicef-evaluator-app-audio-file-storage"
JSON_DATA_FILE      = "transcriptions_with_emotions.json"   # local file in repo root
HF_TOKEN            = os.environ.get("HF_TOKEN", "")
SECRET_KEY          = os.environ.get("SECRET_KEY", "unicef-asr-secret-change-me")

MAX_VOLUNTEERS      = 5
CLAIM_MAP_PATH      = "distribution/claim_map.json"

app = Flask(__name__)
app.secret_key = SECRET_KEY

api = HfApi(token=HF_TOKEN)


# ── Load precomputed dataset from local JSON (once) ───────────────────────────
_precomputed_cache = None

def get_precomputed() -> list:
    """Load the precomputed emotions from local JSON file. Returns list of dicts."""
    global _precomputed_cache
    if _precomputed_cache is None:
        with open(JSON_DATA_FILE, "r", encoding="utf-8") as f:
            _precomputed_cache = json.load(f)
        print(f"[INFO] Loaded {len(_precomputed_cache)} samples from {JSON_DATA_FILE}")
    return _precomputed_cache


# ── Claim map helpers ─────────────────────────────────────────────────────────
def _dl_claim_map():
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=BUCKET_ID, filename=CLAIM_MAP_PATH,
            repo_type="dataset", token=HF_TOKEN,
        )
        with open(local) as f:
            return json.load(f)
    except:
        return {}

def _ul_claim_map(m):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
        tmp = f.name
    try:
        api.upload_file(
            path_or_fileobj=tmp, path_in_repo=CLAIM_MAP_PATH,
            repo_id=BUCKET_ID, repo_type="dataset", token=HF_TOKEN,
        )
    finally:
        os.unlink(tmp)

def claim_samples(volunteer_id, lang):
    ds = get_precomputed()
    lang_indices = [i for i, row in enumerate(ds) if row["language"] == lang]
    total = len(lang_indices)

    for attempt in range(3):
        claim_map = _dl_claim_map()
        lang_map = claim_map.get(lang, {})
        if volunteer_id in lang_map:
            return [lang_indices[i] for i in lang_map[volunteer_id]]

        claimed = len(lang_map)
        if claimed >= MAX_VOLUNTEERS:
            rng = random.Random(f"{volunteer_id}_{lang}_overflow")
            my_idx = rng.sample(range(total), min(total // MAX_VOLUNTEERS, total))
        else:
            per_person = total // MAX_VOLUNTEERS
            start = claimed * per_person
            end = min(start + per_person, total) if claimed < MAX_VOLUNTEERS - 1 else total
            band = list(range(start, end))
            if len(band) < per_person:
                tail = list(range(MAX_VOLUNTEERS * per_person, total))
                rng = random.Random(f"{volunteer_id}_{lang}_tail")
                rng.shuffle(tail)
                band += tail[:per_person - len(band)]
            rng = random.Random(f"{volunteer_id}_{lang}_sample")
            rng.shuffle(band)
            my_idx = band

        lang_map[volunteer_id] = my_idx
        claim_map[lang] = lang_map
        try:
            _ul_claim_map(claim_map)
            return [lang_indices[i] for i in my_idx]
        except Exception as e:
            print(f"claim error: {e}")
            time.sleep(0.5)

    # fallback
    rng = random.Random(f"{volunteer_id}_{lang}_final")
    my_idx = rng.sample(range(total), min(total // MAX_VOLUNTEERS, total))
    return [lang_indices[i] for i in my_idx]


# ── Volunteer code ─────────────────────────────────────────────────────────────
def decode_code(code):
    try:
        padding = 4 - len(code) % 4
        if padding != 4:
            code += "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(code))
        expected = hashlib.md5(
            f"{payload['name']}|{payload['lang'].lower()}|unicef-asr".encode()
        ).hexdigest()[:6]
        return payload if payload.get("chk") == expected else None
    except:
        return None


# ── Bucket I/O ─────────────────────────────────────────────────────────────────
def _up(local, remote):
    api.upload_file(
        path_or_fileobj=local, path_in_repo=remote,
        repo_id=BUCKET_ID, repo_type="dataset", token=HF_TOKEN,
    )

def _dl(remote):
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(
            repo_id=BUCKET_ID, filename=remote,
            repo_type="dataset", token=HF_TOKEN,
        )
    except:
        return None

def bucket_load_progress(vid):
    p = _dl(f"progress/{vid}.json")
    return json.load(open(p)) if p else None

def bucket_save_progress(vid, data):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        tmp = f.name
    try:
        _up(tmp, f"progress/{vid}.json")
    finally:
        os.unlink(tmp)

def bucket_save_results(vid, data):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        tmp = f.name
    try:
        _up(tmp, f"results/{vid}_results.json")
    finally:
        os.unlink(tmp)

def get_eval_data():
    vid = session.get("volunteer_id")
    return bucket_load_progress(vid) if vid else None

def save_eval_data(data):
    vid = session.get("volunteer_id")
    if vid:
        bucket_save_progress(vid, data)


# ── Session helpers ────────────────────────────────────────────────────────────
def new_session(info):
    lang = info["lang"]
    name = info["name"]
    vid = f"{name.replace(' ', '_')}_{lang}"
    ds = get_precomputed()
    indices = claim_samples(vid, lang)
    items = []
    for idx in indices:
        row = ds[idx]
        use_ft = random.choice([True, False])          # 50% base / 50% fine‑tuned
        sentiment = row["sentiment_ft"] if use_ft else row["sentiment_base"]
        confidence = row["confidence_ft"] if use_ft else row["confidence_base"]
        reasoning = row["reasoning_ft"] if use_ft else row["reasoning_base"]
        items.append({
            "index": idx,
            "text": row["text"],                       # original ground‑truth text
            "model_used": "finetuned" if use_ft else "base",
            "predicted_sentiment": sentiment,
            "sentiment_confidence": confidence,
            "sentiment_reasoning": reasoning,
            "volunteer_judgment": None,
            "completed": False,
        })
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

def calc_stats(sess_data):
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
@app.route("/")
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
        return render_template("login.html", error="Invalid code.")
    vid = f"{info['name'].replace(' ', '_')}_{info['lang']}"
    sess_data = bucket_load_progress(vid) or new_session(info)
    bucket_save_progress(vid, sess_data)
    session.clear()
    session["volunteer_id"] = vid
    session["volunteer_name"] = info["name"]
    session["language"] = info["lang"]
    return redirect(url_for("evaluate"))

@app.route("/evaluate")
def evaluate():
    if "volunteer_id" not in session:
        return redirect(url_for("index"))
    sess_data = get_eval_data()
    if not sess_data:
        session.clear()
        return redirect(url_for("index"))
    st = calc_stats(sess_data)
    idx = sess_data.get("current_index", 0)
    items = sess_data.get("items", [])
    done = idx >= len(items)
    current = items[idx] if not done else None
    return render_template("evaluate.html",
                           sess=sess_data, stats=st,
                           current=current, done=done,
                           num=idx + 1, total=len(items))

@app.route("/api/judge", methods=["POST"])
def api_judge():
    if "volunteer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json()
    judgment = data.get("judgment")
    if judgment not in ("correct", "wrong"):
        return jsonify({"error": "Invalid judgment"}), 400
    sess_data = get_eval_data()
    if not sess_data:
        return jsonify({"error": "Session not found"}), 400
    idx = sess_data["current_index"]
    items = sess_data["items"]
    it = items[idx]
    it["volunteer_judgment"] = judgment
    it["completed"] = True
    items[idx] = it
    sess_data["items"] = items
    sess_data["total_completed"] += 1
    sess_data["current_index"] += 1

    total = len(items)
    done = sess_data["current_index"] >= total
    if done:
        sess_data["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        bucket_save_results(session["volunteer_id"], sess_data)
    save_eval_data(sess_data)
    st = calc_stats(sess_data)
    return jsonify({"done": done, "stats": st,
                    "next_index": sess_data["current_index"], "total": total})

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
