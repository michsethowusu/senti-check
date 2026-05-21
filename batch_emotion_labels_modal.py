"""
fill_emotions_async.py  –  Gemini API, async version
Reads transcriptions_backup.json, predicts emotions via Gemini,
saves progress incrementally. Resumes on restart. Stays under RPM limit.
"""

import asyncio
import json
import os
import random
import time

from google import genai
from dotenv import load_dotenv

try:
    from tqdm.asyncio import tqdm as atqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

INPUT_FILE  = "transcriptions_backup.json"
OUTPUT_FILE = "transcriptions_with_emotions.json"

# ── Rate limit config ─────────────────────────────────────────────────────────
# Gemini free tier: 15 RPM / paid: 1000 RPM
# Change MAX_RPM to match your quota. BUCKET_CAPACITY controls burst size.
MAX_RPM         = 20            # stay 1 under to have a safety margin
BUCKET_CAPACITY = 4             # max burst before throttling kicks in
REFILL_RATE     = MAX_RPM / 60  # tokens per second

# Max tasks allowed in-flight at once (prevents spawning 1000 coroutines)
MAX_CONCURRENT  = 15

EMOTION_LABELS = [
    "Happy", "Excited", "Proud", "Loved", "Relieved", "Grateful",
    "Sad", "Lonely", "Hopeless", "Guilty", "Ashamed",
    "Angry", "Frustrated", "Irritated",
    "Afraid", "Anxious", "Overwhelmed",
    "Confused", "Curious", "Doubtful",
    "Embarrassed", "Insecure", "Jealous", "Rejected",
]


# ── Token bucket ──────────────────────────────────────────────────────────────
class TokenBucket:
    """
    Async token bucket rate limiter.
    acquire() yields until a token is available — never busy-spins.
    """

    def __init__(self, capacity: float, refill_rate: float):
        self.capacity    = capacity
        self.refill_rate = refill_rate
        self.tokens      = float(capacity)
        self.last_refill = time.monotonic()
        self._lock       = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            while True:
                now     = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(
                    self.capacity,
                    self.tokens + elapsed * self.refill_rate,
                )
                self.last_refill = now

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return

                wait = (1.0 - self.tokens) / self.refill_rate
                await asyncio.sleep(wait)


# ── Emotion prediction ────────────────────────────────────────────────────────
async def predict_emotion(
    client: genai.Client,
    bucket: TokenBucket,
    transcription: str,
    lang: str,
) -> dict:
    await bucket.acquire()

    labels_str = ", ".join(EMOTION_LABELS)
    prompt = (
        f"Language: {lang}\nTranscription: {transcription}\n"
        f"Choose the closest emotion from: {labels_str}\n"
        'Respond ONLY with JSON: {"sentiment":"...", "confidence":0.xx, "reasoning":"..."}'
    )

    # run_in_executor lets us use the synchronous Gemini SDK without blocking
    # the event loop — keeps all other coroutines running during the API call
    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model="gemma-4-31b-it",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                thinking_config=genai.types.ThinkingConfig(thinking_level="HIGH"),
            ),
        ),
    )

    raw = resp.text.strip()
    if "```" in raw:
        raw = raw.split("```")[1].replace("json", "").strip()
    s, e = raw.find("{"), raw.rfind("}") + 1
    if s == -1 or e == 0:
        raise ValueError(f"No JSON found in response: {raw[:120]}")

    result = json.loads(raw[s:e])
    if result.get("sentiment") not in EMOTION_LABELS:
        raise ValueError(f"Invalid sentiment label: {result.get('sentiment')!r}")
    return result


async def predict_with_retry(
    client: genai.Client,
    bucket: TokenBucket,
    transcription: str,
    lang: str,
    max_retries: int = 5,
) -> dict:
    last_exc = None
    for attempt in range(max_retries):
        try:
            return await predict_emotion(client, bucket, transcription, lang)
        except Exception as exc:
            msg = str(exc).upper()
            is_transient = any(
                code in msg
                for code in (
                    "500", "502", "503", "429",
                    "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL",
                    "TIMED OUT", "TIMEOUT",
                )
            )
            if is_transient:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"\n  ⚠ Transient error (attempt {attempt + 1}/{max_retries}), "
                      f"retrying in {wait:.1f}s: {exc}")
                await asyncio.sleep(wait)
                last_exc = exc
            else:
                raise
    return {"error": str(last_exc)}


# ── File handling ─────────────────────────────────────────────────────────────
def load_samples() -> list:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError("Input file must be a JSON array")

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            samples = json.load(f)
        print(f"Resuming from {OUTPUT_FILE} ({len(samples)} samples)")
    else:
        print(f"Starting fresh from {INPUT_FILE} ({len(samples)} samples)")
    return samples


def save_samples(samples: list):
    tmp = OUTPUT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUTPUT_FILE)    # atomic on POSIX — safe to Ctrl+C


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    samples   = load_samples()
    client    = genai.Client(api_key=GEMINI_API_KEY)
    save_lock = asyncio.Lock()

    pending = []
    for idx, s in enumerate(samples):
        lang = s.get("language", "unknown")
        if s.get("transcription_base") and not s.get("sentiment_base"):
            pending.append((idx, "base", s["transcription_base"].strip(), lang))
        if s.get("transcription_ft") and not s.get("sentiment_ft"):
            pending.append((idx, "ft", s["transcription_ft"].strip(), lang))

    if not pending:
        print("✅ All emotions already present.")
        return

    print(f"Predictions needed: {len(pending)}")

    bucket    = TokenBucket(BUCKET_CAPACITY, REFILL_RATE)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def process_one(idx: int, model: str, txt: str, lang: str):
        async with semaphore:
            try:
                res = await predict_with_retry(client, bucket, txt, lang)
            except Exception as exc:
                res = {"error": str(exc)}

        if "error" not in res:
            samples[idx][f"sentiment_{model}"]  = res["sentiment"]
            samples[idx][f"confidence_{model}"] = res["confidence"]
            samples[idx][f"reasoning_{model}"]  = res["reasoning"]
        else:
            print(f"  ✗ Sample {idx} [{model}] failed: {res['error']}")

        async with save_lock:
            save_samples(samples)

    tasks = [
        asyncio.create_task(process_one(idx, model, txt, lang))
        for idx, model, txt, lang in pending
    ]

    if HAS_TQDM:
        for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Emotions"):
            await coro
    else:
        done = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            done += 1
            if done % 20 == 0:
                print(f"  Progress: {done}/{len(tasks)}")

    print(f"✅ Done – results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹ Interrupted – progress already saved incrementally.")
