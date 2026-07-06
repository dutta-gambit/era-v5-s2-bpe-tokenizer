# One tokenizer, four scripts — ERA V5 Session 2

A 10,000-token BPE tokenizer trained from scratch (vanilla JS, zero dependencies) on
**India's Wikipedia page** in **English, Hindi, Telugu and Bengali**, with the merge
budget split so that all four tokens-per-word ratios are equal to the 4th decimal:

| Language | Words | Tokens | X = tokens/words |
|---|---|---|---|
| English | 9,938 | 16,099 | 1.619944 |
| Hindi | 7,972 | 12,915 | 1.620045 |
| Telugu | 2,318 | 3,755 | 1.619931 |
| Bengali | 5,279 | 8,552 | 1.620004 |

Spread `Xmax − Xmin` = **0.000114** → assignment score `1000/spread` ≈ **8,757,865**.
Vocab: **9,999** of the 10,000 cap (2,236 base akshara/char symbols + 7,763 unique merge tokens).

The widget (`web/`) recomputes every number **live in the browser** with the same
`bpe.js` that trained the tokenizer, provides a playground, the full token list for
download, and a measured analysis of the brief's X ≤ 1.2 target (unreachable at 10k
vocab on page-sized corpora — needs ≈ 16.9k; the chart shows the exact trade-off curve).

## Reproduce

```bash
# ratios straight from the shipped tokenizer (matches the widget; any node >= 14)
node src/tokenize-cli.js corpus/en.txt corpus/hi.txt corpus/te.txt corpus/bn.txt

# full pipeline from the pinned Wikipedia snapshots (data/raw_*.json, revision ids inside)
node src/prepare-corpus.js   # -> corpus/*.txt + manifest (sha-256, revids)
node src/train.js            # -> web/tokenizer.json, stats.json, analysis.json, downloads
```

## Design notes

- **Pipeline**: whitespace runs → single space; every word prefixed with `▁` (U+2581);
  symbols = **akshara clusters** via a table-driven segmenter defined in `bpe.js`
  (deliberately not `Intl.Segmenter`: ICU versions disagree on Indic conjuncts, which
  changed counts by ±1 between engines — ours is bit-identical everywhere);
  merges applied greedily in rank order over the whole stream (leftmost first).
- **Cross-word merges allowed** (SentencePiece `split_by_whitespace=false` style):
  on 2–10k-word corpora, word-internal BPE exhausts all frequency-≥2 merges at
  fertility 1.28 (en) … 2.16 (te) — the 1.2 target needs phrase tokens or ~17k vocab.
- **Budget split**: per-language merge tables (scripts are disjoint), interleaved and
  deduped; budgets chosen by binary-searching a common fertility target on the
  training curves, then hill-climbing against the real unified tokenizer to minimize
  the spread. Everything is deterministic (lexicographic tie-breaks, no RNG).
- `X = tokens / words`, words = `text.split(/\s+/)` on the raw corpus. Definitions are
  embedded in `tokenizer.json` and printed by the CLI.

## Layout

```
corpus/    cleaned corpora + manifest (revids, sha-256)   [what the ratios are measured on]
data/      raw Wikipedia API snapshots                    [source of truth]
src/       bpe.js (trainer+tokenizer) · train.js · prepare-corpus.js · tokenize-cli.js
web/       the widget (static, self-contained — deploy this folder)
```
