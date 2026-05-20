"""
generate_codes.py  –  Run locally to create volunteer login codes.

Usage:
    python generate_codes.py

Then share the printed codes with your volunteers.
"""

import base64, hashlib, json

def make_code(name: str, lang: str) -> str:
    lang = lang.upper()
    chk  = hashlib.md5(f"{name}|{lang.lower()}|unicef-asr".encode()).hexdigest()[:6]
    payload = json.dumps({"name": name, "lang": lang, "chk": chk}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).rstrip(b"=").decode()

if __name__ == "__main__":
    volunteers = [
        # ("Full Name", "LANGUAGE")   – LANGUAGE must be TWI, EWE, or DAGBANI
        ("Kofi Mensah",   "TWI"),
        ("Ama Owusu",     "TWI"),
        ("Esi Amoah",     "EWE"),
        ("Mawuli Agbeko", "EWE"),
        ("Alhassan Baba", "DAGBANI"),
    ]

    print("\n=== UNICEF Ghana ASR – Volunteer Codes ===\n")
    for name, lang in volunteers:
        code = make_code(name, lang)
        print(f"  {name:20s}  [{lang}]  →  {code}")
    print()
