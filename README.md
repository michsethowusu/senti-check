# UNICEF Ghana NLP – ASR Evaluator (Flask + Modal GPU)

A volunteer-facing web app for evaluating Ghanaian-language ASR models (Twi, Ewe, Dagbani).
Compares `ghananlpcommunity/qwen3-asr-0.6b-ghana-multilang` (fine-tuned) vs `Qwen/Qwen3-ASR-0.6B` (base).

## Architecture

```
Browser  ──── Flask on Render (web UI, sessions, HF bucket I/O)
                      │
                      │  POST audio (base64 JSON)
                      ▼
              Modal GPU endpoint  (T4 GPU, both models loaded)
                      │
                      │  { "transcription": "..." }
                      ▼
              Flask  ──── NVIDIA NIM (DeepSeek sentiment)
                      │
                      └── HuggingFace bucket (audio, results, progress)
```

- **Render** runs Flask – no GPU needed, free tier is fine.
- **Modal** runs both ASR models on a T4 GPU. You pay per second of GPU use (~$0.00059/s on T4). A 10-second transcription costs less than $0.01.
- Models are cached in a Modal Volume so cold starts after the first deploy are fast.

---

## Repo structure

```
├── app.py               # Flask app (all routes + logic)
├── modal_asr.py         # Modal GPU service (deploy separately)
├── generate_codes.py    # Local script to mint volunteer codes
├── requirements.txt     # Flask deps only – no torch/transformers
├── Procfile
├── render.yaml
├── .gitignore
├── templates/
│   ├── login.html
│   └── evaluate.html
└── static/css/main.css
```

---

## Step 1 – Set up Modal

```bash
pip install modal
modal setup          # authenticates your account in the browser
```

### Create Modal secrets

In the [Modal dashboard](https://modal.com/secrets) create two secrets:

| Secret name               | Key             | Value                              |
|---------------------------|-----------------|------------------------------------|
| `unicef-hf-token`         | `HF_TOKEN`      | Your HuggingFace token             |
| `unicef-asr-shared-secret`| `SHARED_SECRET` | Any long random string you choose  |

### Pre-download models into the Volume (one-time)

```bash
modal run modal_asr.py::download_models
```

This takes ~5 min on first run. After that models live in the Volume forever.

### Deploy the Modal service

```bash
modal deploy modal_asr.py
```

Modal prints the endpoint URL, e.g.:
```
https://your-workspace--unicef-asr-transcribe.modal.run
```

**Copy this URL** – you'll need it for Render.

---

## Step 2 – Deploy Flask to Render

### Push to GitHub

```bash
git init
git add .
git commit -m "initial"
git remote add origin https://github.com/YOUR_ORG/unicef-asr-evaluator.git
git push -u origin main
```

### Create Web Service on Render

1. [render.com](https://render.com) → **New → Web Service** → connect your repo
2. Render reads `render.yaml` automatically
3. Add these **env vars** in the Render dashboard:

| Key                   | Value                                             |
|-----------------------|---------------------------------------------------|
| `HF_TOKEN`            | Your HuggingFace token                            |
| `NVIDIA_API_KEY`      | NVIDIA NIM key (for DeepSeek sentiment)           |
| `MODAL_ASR_URL`       | The URL printed by `modal deploy`                 |
| `MODAL_SHARED_SECRET` | Same random string you put in the Modal secret    |

4. Click **Deploy**

---

## Step 3 – Generate volunteer codes

Edit `generate_codes.py` with your volunteers' names and languages, then:

```bash
python generate_codes.py
```

Share codes with volunteers. Each code is signed with a checksum so it can't be guessed.

---

## Cost estimate

| Component | Cost |
|-----------|------|
| Render (free tier) | $0/mo |
| Modal T4 GPU | ~$0.00059/s active · $0 when idle |
| 5 volunteers × 100 recordings × ~10s each | ≈ $0.30 total |
| NVIDIA NIM (DeepSeek) | Free tier / pay-per-token |

Modal containers stay warm for 5 min after the last request (`container_idle_timeout=300`), so back-to-back recordings by the same volunteer don't incur cold starts.

---

## Data stored in HuggingFace bucket

```
ghananlpcommunity/unicef-evaluator-app-audio-file-storage/
├── distribution/claim_map.json          # which volunteer got which sentences
├── progress/<volunteer_id>.json         # saved after every judgment
├── results/<volunteer_id>_results.json  # written on completion
└── audio/<volunteer_id>/item_NNN.wav    # uploaded recordings
```
