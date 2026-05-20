"""
modal_asr.py  –  Modal GPU transcription service
Deploy with:  modal deploy modal_asr.py

Exposes a web endpoint that Flask calls for ASR inference.
Both models are cached in a Modal Volume so cold starts
after the first deploy are fast (no re-download).
"""

import os
import io
import base64
import modal

# ── Modal app ─────────────────────────────────────────────────────────────────
app = modal.App("unicef-ghana-asr")

# Volume caches model weights across container restarts
model_volume = modal.Volume.from_name("unicef-asr-models", create_if_missing=True)

FINETUNED_MODEL = "ghananlpcommunity/qwen3-asr-0.6b-ghana-multilang"
BASE_MODEL      = "Qwen/Qwen3-ASR-0.6B"
MODEL_CACHE_DIR = "/model-cache"

HF_TOKEN_SECRET = modal.Secret.from_name("unicef-hf-token")   # set up in Modal dashboard
SHARED_SECRET   = modal.Secret.from_name("unicef-asr-shared-secret")  # protects the endpoint

# ── Image ──────────────────────────────────────────────────────────────────────
# Bake heavy deps into the image so they're available on every cold start
asr_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "transformers>=4.41.0",
        "huggingface-hub>=0.23.0",
        "accelerate>=0.30.0",
        "soundfile",
        "librosa",
        "fastapi[standard]",
    )
)


# ── Model download function (run once, stored in Volume) ──────────────────────
@app.function(
    image=asr_image,
    volumes={MODEL_CACHE_DIR: model_volume},
    secrets=[HF_TOKEN_SECRET],
    timeout=900,
)
def download_models():
    """
    Download both models into the Volume.
    Run once manually:  modal run modal_asr.py::download_models
    """
    from huggingface_hub import snapshot_download
    hf_token = os.environ["HF_TOKEN"]

    for model_id in [FINETUNED_MODEL, BASE_MODEL]:
        local_dir = f"{MODEL_CACHE_DIR}/{model_id.replace('/', '__')}"
        print(f"[INFO] Downloading {model_id} → {local_dir}")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            token=hf_token,
        )
        print(f"[INFO] Done: {model_id}")

    model_volume.commit()
    print("[INFO] All models saved to Volume.")


# ── ASR inference class ────────────────────────────────────────────────────────
@app.cls(
    image=asr_image,
    gpu="T4",                           # T4 ~$0.00059/s – cheap for 0.6B model
    volumes={MODEL_CACHE_DIR: model_volume},
    secrets=[HF_TOKEN_SECRET, SHARED_SECRET],
    timeout=180,
    scaledown_window=300,               # keep warm for 5 min between requests
)
@modal.concurrent(max_inputs=4)          # Allow up to 4 concurrent transcriptions (class-level)
class ASRService:

    @modal.enter()
    def load_models(self):
        """Called once when the container starts – loads both models into GPU memory."""
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline

        hf_token = os.environ.get("HF_TOKEN", "")
        dev      = "cuda"
        dtype    = torch.float16

        self._pipes = {}
        for model_id in [FINETUNED_MODEL, BASE_MODEL]:
            local_dir = f"{MODEL_CACHE_DIR}/{model_id.replace('/', '__')}"
            # Fall back to HF hub if volume doesn't have it yet
            load_from = local_dir if os.path.isdir(local_dir) else model_id
            print(f"[INFO] Loading {model_id} from {load_from}")
            m    = AutoModelForSpeechSeq2Seq.from_pretrained(
                       load_from, torch_dtype=dtype, low_cpu_mem_usage=True,
                       token=hf_token).to(dev)
            proc = AutoProcessor.from_pretrained(load_from, token=hf_token)
            self._pipes[model_id] = hf_pipeline(
                "automatic-speech-recognition",
                model=m, tokenizer=proc.tokenizer,
                feature_extractor=proc.feature_extractor,
                torch_dtype=dtype, device=dev,
            )
            print(f"[INFO] Loaded {model_id}")

    @modal.fastapi_endpoint(method="POST")   # Replaces deprecated @modal.web_endpoint
    def transcribe(self, request_data: dict) -> dict:
        """
        POST body (JSON):
          {
            "audio_b64": "<base64-encoded audio bytes>",
            "audio_fmt": "webm",          # extension hint
            "model":     "finetuned" | "base",
            "secret":    "<SHARED_SECRET value>"
          }

        Returns:
          { "transcription": "..." }   or   { "error": "..." }
        """
        import tempfile, base64 as b64

        # ── Auth ──
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

        # Write to temp file
        suffix = f".{audio_fmt}" if audio_fmt else ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            pipe   = self._pipes[model_id]
            result = pipe(tmp_path, return_timestamps=False)
            text   = result["text"].strip()
            return {"transcription": text}
        except Exception as e:
            return {"error": f"Transcription failed: {e}"}
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
