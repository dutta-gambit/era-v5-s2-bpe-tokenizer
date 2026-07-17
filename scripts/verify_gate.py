#!/usr/bin/env python3
"""Replays the grader's evaluation against web/tokenizer.json:

1. Faithful-roundtrip gate: decode(encode(text)) must preserve every visible
   (non-whitespace) character — on the instructor's exact failing sample, Markdown,
   emoji, and mixed-script text.
2. The ratios: X = tokens/words for each pinned corpus, spread, and score — asserted
   equal to the claims in web/stats.json (what the widget displays).

Run: .venv/bin/python scripts/verify_gate.py
"""
import json
import os
import sys

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = Tokenizer.from_file(os.path.join(ROOT, "web", "tokenizer.json"))
print(f"loaded tokenizer: vocab {tok.get_vocab_size()}")

failures = []

# ---- 1. roundtrip gate ----
GATE_SAMPLES = [
    "India's population is 1,428,627,663.",           # the instructor's exact sample
    "India’s GDP grew ~7% in FY24 — details below.",  # curly quote + tilde + em-dash
    "# India\n\n**Bold** _italic_ `code` [link](https://en.wikipedia.org/wiki/India)\n\n- item 1\n- item 2\n> quote | table",
    "भारत एक विशाल देश है।",
    "భారతదేశం ఒక గొప్ప దేశం.",
    "ভারত একটি বিশাল দেশ।",
    "Mixed: India भारत భారతదేశం ভারত \U0001F1EE\U0001F1F3 → 100% ✓",
    "Tabs\tand\nnewlines\r\n  multiple   spaces",
    "Code: def f(x): return {'a': x[0], \"b\": x**2} # comment @decorator $var %fmt &amp;",
]

def visible(s: str) -> str:
    return "".join(s.split())

for s in GATE_SAMPLES:
    ids = tok.encode(s).ids
    back = tok.decode(ids)
    ok = visible(back) == visible(s)
    label = s[:48].replace("\n", "\\n")
    print(f"  gate {'PASS' if ok else 'FAIL'}: {label!r} ({len(ids)} tokens)")
    if not ok:
        failures.append(f"roundtrip failed on {s!r}: got {back!r}")

# ---- 2. ratios vs claims ----
with open(os.path.join(ROOT, "web", "stats.json"), encoding="utf-8") as f:
    stats = json.load(f)

xs = {}
for p in stats["perLanguage"]:
    lang = p["lang"]
    with open(os.path.join(ROOT, "corpus", f"{lang}.txt"), encoding="utf-8") as f:
        text = f.read()
    words = len(text.split())
    n_tokens = len(tok.encode(text).ids)
    x = n_tokens / words
    xs[lang] = x
    claimed = (p["words"], p["tokens"])
    ok = (words, n_tokens) == claimed
    print(f"  {lang}: words={words} tokens={n_tokens} X={x:.6f} "
          f"{'== claimed' if ok else f'!= claimed {claimed}'}")
    if not ok:
        failures.append(f"{lang}: recomputed (words,tokens)=({words},{n_tokens}) != claimed {claimed}")

spread = max(xs.values()) - min(xs.values())
score = 1000 / spread
print(f"  spread={spread:.6f} score={score:,.1f} (claimed {stats['spread']:.6f} / {stats['score']:,.1f})")
if abs(spread - stats["spread"]) > 1e-9:
    failures.append(f"spread mismatch: {spread} vs claimed {stats['spread']}")

if failures:
    print("\nFAILURES:")
    for f_ in failures:
        print(" -", f_)
    sys.exit(1)
print("\nALL CHECKS PASS — grader-side behavior verified with the real `tokenizers` library.")
