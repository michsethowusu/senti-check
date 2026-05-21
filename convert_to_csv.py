"""
convert_to_csv.py
-----------------
Converts transcriptions_with_emotions.json → transcriptions_with_emotions.csv
Drops any row where ANY of the required fields is None/null/empty string.

Required fields (must all be present and non-empty):
  language, text,
  sentiment_base, confidence_base, reasoning_base,
  sentiment_ft,   confidence_ft,   reasoning_ft

Run: python convert_to_csv.py
"""

import json
import csv
import sys

INPUT_FILE  = "transcriptions_with_emotions.json"
OUTPUT_FILE = "transcriptions_with_emotions.csv"

REQUIRED_FIELDS = [
    "language",
    "text",
    "sentiment_base",
    "confidence_base",
    "reasoning_base",
    "sentiment_ft",
    "confidence_ft",
    "reasoning_ft",
]

# All columns written to CSV (optional fields kept if present)
ALL_COLUMNS = [
    "language",
    "original_index",
    "text",
    "transcription_base",
    "transcription_ft",
    "sentiment_base",
    "confidence_base",
    "reasoning_base",
    "sentiment_ft",
    "confidence_ft",
    "reasoning_ft",
]

def is_missing(value):
    """True if value is None, empty string, or whitespace-only."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False

def main():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total       = len(data)
    kept        = []
    dropped     = []

    for i, row in enumerate(data):
        missing = [field for field in REQUIRED_FIELDS if is_missing(row.get(field))]
        if missing:
            dropped.append((i, row.get("original_index", i), missing))
        else:
            kept.append(row)

    print(f"\nTotal rows  : {total}")
    print(f"Kept        : {len(kept)}")
    print(f"Dropped     : {len(dropped)}")

    if dropped:
        print("\nDropped rows (original_index → missing fields):")
        for idx, orig_idx, fields in dropped:
            print(f"  row {idx:>4}  original_index={orig_idx}  missing: {', '.join(fields)}")

    if not kept:
        print("\nNo rows to write. Exiting.")
        sys.exit(1)

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in kept:
            writer.writerow({col: row.get(col, "") for col in ALL_COLUMNS})

    print(f"\nCSV written to {OUTPUT_FILE}  ({len(kept)} rows)")

if __name__ == "__main__":
    main()
