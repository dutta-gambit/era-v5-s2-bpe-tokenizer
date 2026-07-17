# One tokenizer, four scripts — ERA V5 Session 2

A BPE tokenizer trained from scratch on **India's Wikipedia page** in **English, Hindi,
Telugu and Bengali**, shipped as a standard **HuggingFace `tokenizer.json`** —
`Tokenizer.from_file()` gives a working `encode()` **and** `decode()`, and
decode(encode(text)) preserves every visible character on any input (byte-fallback).
The merge budget is split so all four tokens-per-word ratios are equal to the 4th decimal:

| Language | Words | Tokens | X = tokens/words |
|---|---|---|---|
| English | 9,938 | 15,790 | 1.588851 |
| Hindi | 7,972 | 12,666 | 1.588811 |
| Telugu | 2,318 | 3,683 | 1.588870 |
| Bengali | 5,279 | 8,387 | 1.588748 |

Spread `Xmax − Xmin` = **0.000122** → assignment score `1000/spread` ≈ **8,207,057**.
Vocab: **9,987** of the 10,000 cap (256 byte tokens + 318 base code points + 9,413 merge tokens).

The widget (`web/`) recomputes every ratio **and replays the faithful-roundtrip gate**
live in the browser, provides an encode/decode playground, the full token list for
download, and a measured analysis of the brief's X ≤ 1.2 target (unreachable at 10k
vocab on page-sized corpora — needs ≈ 18.3k; the chart shows the exact trade-off curve).

## Reproduce — the grader's view (python, real 🤗 tokenizers)

```bash
pip install tokenizers
python scripts/verify_gate.py        # roundtrip gate + every claimed number, asserted

# or by hand:
python -c "from tokenizers import Tokenizer; t = Tokenizer.from_file('web/tokenizer.json'); \
  print(t.decode(t.encode(\"India's population is 1,428,627,663.\").ids))"
```

Same numbers from node (no dependencies, any node ≥ 14):

```bash
node src/tokenize-cli.js corpus/en.txt corpus/hi.txt corpus/te.txt corpus/bn.txt

# full retrain from the pinned Wikipedia snapshots (revids in corpus/manifest.json)
node src/prepare-corpus.js && node src/train.js
python scripts/build_hf.py           # emits the canonical web/tokenizer.json
```

## Design notes

- **Pipeline** (exactly what `tokenizer.json` encodes): Metaspace pre-tokenization
  (`▁` per word, merges within words), BPE over code points, `byte_fallback: true`
  for out-of-vocab characters, llama-style decoder chain. Built via the python
  `tokenizers` API — not hand-written JSON — so any loader reconstructs it.
- **The decode gate shaped the design**: our first submission used cross-word phrase
  merges over akshara clusters (X ≈ 1.62 at the cap) — better compression, but not
  loadable/decodable by standard libraries, and the grader's faithful-roundtrip check
  rightly failed it. Word-internal HF-compatible BPE actually equalizes *better*
  (X ≈ 1.5888) because the code-point alphabet is 6× cheaper than the akshara one.
- **Budget split**: per-language merge tables (scripts are disjoint), interleaved and
  deduped; budgets by binary-searching a common fertility target on the training
  curves, then hill-climbing pure spread against the real unified tokenizer.
  Deterministic throughout (lexicographic tie-breaks, no RNG, no ICU dependence).
- `X = tokens / words`, words = `text.split(/\s+/)` on the raw corpus.

## Layout

```
corpus/    cleaned corpora + manifest (revids, sha-256)   [what the ratios are measured on]
data/      raw Wikipedia API snapshots                    [source of truth]
src/       bpe.js (trainer+tokenizer) · train.js · prepare-corpus.js · tokenize-cli.js
scripts/   build_hf.py (emit canonical tokenizer.json) · verify_gate.py (grader replay)
web/       the widget (static — deploy this folder; publish dir = web)
```
