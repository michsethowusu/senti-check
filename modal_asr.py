"""
modal_asr.py  –  Modal GPU transcription service (auto‑caching via HF cache)
Deploy with:  modal deploy modal_asr.py

Exposes a web endpoint that Flask calls for ASR inference.
Model weights are automatically cached in the Modal Volume
by setting HF_HOME to the volume mount point.
"""

import os
import io
import base64
import modal

# ── Modal app ─────────────────────────────────────────────────────────────────
app = modal.App("unicef-ghana-asr")

# Volume for HuggingFace cache (survives container restarts)
model_volume = modal.Volume.from_name("unicef-asr-models", create_if_missing=True)

FINETUNED_MODEL = "ghananlpcommunity/qwen3-asr-0.6b-ghana-twi-ewe-dagbani"
BASE_MODEL      = "Qwen/Qwen3-ASR-0.6B"
MODEL_CACHE_DIR = "/model-cache"          # will be used as HF_HOME

HF_TOKEN_SECRET = modal.Secret.from_name("unicef-hf-token")
SHARED_SECRET   = modal.Secret.from_name("unicef-asr-shared-secret")

# ── Image ─────────────────────────────────────────────────────────────────────
asr_image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"HF_HOME": MODEL_CACHE_DIR})     # 👈 all HF downloads go to the Volume
    .pip_install(
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "transformers>=4.41.0",
        "huggingface-hub>=0.23.0",
        "accelerate>=0.30.0",
        "soundfile",
        "librosa",
        "fastapi[standard]",
        "qwen-asr",
    )
)


# ── (Optional) Explicit download function – not needed anymore, but kept for reference ──
@app.function(
    image=asr_image,
    volumes={MODEL_CACHE_DIR: model_volume},
    secrets=[HF_TOKEN_SECRET],
    timeout=900,
)
def download_models():
    """
    You can still run this if you want to pre‑populate the cache, but it’s
    no longer required – the models will be downloaded automatically on first use.
    """
    from huggingface_hub import snapshot_download
    hf_token = os.environ["HF_TOKEN"]

    for model_id in [FINETUNED_MODEL, BASE_MODEL]:
        print(f"[INFO] Downloading {model_id}")
        snapshot_download(repo_id=model_id, token=hf_token)
        print(f"[INFO] Done: {model_id}")

    model_volume.commit()
    print("[INFO] All models saved to Volume.")


# ── ASR inference class ────────────────────────────────────────────────────────
@app.cls(
    image=asr_image,
    gpu="A100-40GB",
    volumes={MODEL_CACHE_DIR: model_volume},   # mount the cache volume
    secrets=[HF_TOKEN_SECRET, SHARED_SECRET],
    timeout=180,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=4)
class ASRService:

    @modal.enter()
    def load_models(self):
        """Load both models (downloads & caches automatically in /model-cache)."""
        import torch
        from qwen_asr import Qwen3ASRModel

        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        hf_token = os.environ.get("HF_TOKEN", "")

        self._models = {}
        for model_id in [FINETUNED_MODEL, BASE_MODEL]:
            print(f"[INFO] Loading {model_id} (HF cache: {MODEL_CACHE_DIR})")
            model = Qwen3ASRModel.from_pretrained(
                model_id,                      # 👈 HF repo id – downloads what’s missing
                dtype=dtype,
                device_map="cuda:0",
                token=hf_token,
            )
            self._models[model_id] = model
            print(f"[INFO] Loaded {model_id}")

    @modal.fastapi_endpoint(method="POST")
    def transcribe(self, request_data: dict) -> dict:
        """
        POST body (JSON):
          {
            "audio_b64": "<base64-encoded audio bytes>",
            "audio_fmt": "webm",
            "model":     "finetuned" | "base",
            "secret":    "<SHARED_SECRET value>"
          }
        Returns:
          { "transcription": "..." }   or   { "error": "..." }
        """
        import tempfile, base64 as b64

        expected = os.environ.get("SHARED_SECRET", "")
        if expected and request_data.get("secret") != expected:
            return {"error": "Unauthorized"}

        model_key = request_data.get("model", "finetuned")
        model_id  = FINETUNED_MODEL if model_key == "finetuned" else BASE_MODEL

        audio_b64 = request_data.get("audio_b64", "")
        audio_fmt = request_data.get("audio_fmt", "webm")
        if not audio_b64:
            return {"error": "No audio provided"}

        try:
            audio_bytes = b64.b64decode(audio_b64)
        except Exception as e:
            return {"error": f"base64 decode failed: {e}"}

        suffix = f".{audio_fmt}" if audio_fmt else ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            model  = self._models[model_id]
            result = model.transcribe(audio=tmp_path)
            text   = result[0].text.strip() if result else ""
            return {"transcription": text}
        except Exception as e:
            return {"error": f"Transcription failed: {e}"}
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
