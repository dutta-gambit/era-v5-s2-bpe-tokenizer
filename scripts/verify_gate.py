#!/usr/bin/env python3
"""Replays the grader's evaluation against web/tokenizer.json:

1. Faithful-roundtrip gate: decode(encode(text)) preserves every visible character —
   instructor's exact failing sample, Markdown with links, emoji, mixed-script.
2. No-UNK gate: the tokenizer must encode anything (any UNK = assignment 0).
3. The reference metric on BOTH corpus renderings:
   fertility = tokens / faithful units, units = [\\p{L}\\p{M}\\p{N}]+|[^\\s\\p{L}\\p{M}\\p{N}]
   A = markdownify rendering (training corpus), B = html2text rendering (holdout).
   All values asserted equal to web/stats.json (what the widget displays), and
   every fertility must be < 1.2.

Run: .venv/bin/python scripts/verify_gate.py
"""
import json
import os
import sys

import regex
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = Tokenizer.from_file(os.path.join(ROOT, "web", "tokenizer.json"))
print(f"loaded tokenizer: vocab {tok.get_vocab_size()}")

UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")
failures = []

# ---- 1 + 2. roundtrip gate and no-UNK on adversarial samples ----
GATE_SAMPLES = [
    "India's population is 1,428,627,663.",
    "[India](https://en.wikipedia.org/wiki/India) is a **country** in _South Asia_.",
    "# भारत\n\n[भारत गणराज्य](/wiki/भारत) — जनसंख्या ~1.4 अरब।",
    "| రాజధాని | న్యూఢిల్లీ |\n|---|---|",
    "ভারত একটি [প্রজাতন্ত্র](./প্রজাতন্ত্র)।",
    "Mixed: India भारत భారతదేశం ভারত \U0001F1EE\U0001F1F3 → 100% ✓ €₹",
    "unseen script + emoji: Ω≈ç√∫˜µ≤ 日本語 한국어 عربى 🚀",
]

def visible(s: str) -> str:
    return "".join(s.split())

for s in GATE_SAMPLES:
    enc = tok.encode(s)
    back = tok.decode(enc.ids)
    ok = visible(back) == visible(s)
    label = s[:44].replace("\n", "\\n")
    print(f"  gate {'PASS' if ok else 'FAIL'}: {label!r} ({len(enc.ids)} tokens)")
    if not ok:
        failures.append(f"roundtrip failed on {s!r}: got {back!r}")

# ---- 3. reference metric on both renderings, asserted vs claims ----
with open(os.path.join(ROOT, "web", "stats.json"), encoding="utf-8") as f:
    stats = json.load(f)

for variant, tkey, ukey, fkey in (("A", "tokensA", "unitsA", "fertilityA"),
                                  ("B", "tokensB", "unitsB", "fertilityB")):
    xs = {}
    for p in stats["perLanguage"]:
        lang = p["lang"]
        with open(os.path.join(ROOT, "corpus-md", f"{lang}.{variant}.md"), encoding="utf-8") as f:
            text = f.read()
        units = len(UNIT_RE.findall(text))
        n_tokens = len(tok.encode(text).ids)
        x = n_tokens / units
        xs[lang] = x
        ok = n_tokens == p[tkey] and units == p[ukey]
        print(f"  {variant}/{lang}: tokens={n_tokens} units={units} X={x:.6f} "
              f"{'== claimed' if ok else f'!= claimed ({p[tkey]},{p[ukey]})'}")
        if not ok:
            failures.append(f"{variant}/{lang}: ({n_tokens},{units}) != claimed ({p[tkey]},{p[ukey]})")
        if x >= 1.2:
            failures.append(f"{variant}/{lang}: fertility {x:.4f} >= 1.2 hard bar")
    spread = max(xs.values()) - min(xs.values())
    claimed = stats["spreadA"] if variant == "A" else stats["spreadB"]
    print(f"  {variant}: spread={spread:.6f} score={1000/spread:,.0f} (claimed spread {claimed:.6f})")
    if abs(spread - claimed) > 1e-9:
        failures.append(f"{variant}: spread {spread} != claimed {claimed}")

if failures:
    print("\nFAILURES:")
    for f_ in failures:
        print(" -", f_)
    sys.exit(1)
print("\nALL CHECKS PASS — roundtrip, no-UNK coverage, and both renderings' ratios verified"
      "\nwith the real `tokenizers` library; every fertility < 1.2.")
