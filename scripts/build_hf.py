#!/usr/bin/env python3
"""Builds the canonical web/tokenizer.json (HuggingFace `tokenizers` format) from
build/hf-input.json (vocab list + merges produced by src/train.js).

Constructed via the real library — not hand-written JSON — so anything that does
Tokenizer.from_file("tokenizer.json") gets a working encode() AND decode().
Pipeline: Metaspace pre-tokenization (▁ per word) → BPE merges (byte_fallback for
out-of-vocab characters) → llama-style decoder chain that restores visible text.
"""
import json
import os

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(ROOT, "build", "hf-input.json"), encoding="utf-8") as f:
    src = json.load(f)

vocab = {tok: i for i, tok in enumerate(src["vocab"])}
merges = [(a, b) for a, b in src["merges"]]

tok = Tokenizer(models.BPE(
    vocab=vocab,
    merges=merges,
    byte_fallback=True,
    unk_token=None,
    fuse_unk=False,
))
# fold all whitespace runs to single spaces BEFORE Metaspace: markdown is full of
# newlines/tabs, which Metaspace would otherwise leave to byte-fallback as <0x0A>,
# inflating token counts. Whitespace is not part of the faithful-roundtrip gate.
tok.normalizer = normalizers.Sequence([
    normalizers.Replace(Regex(r"\s+"), " "),
    normalizers.Strip(left=True, right=True),
])
tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
tok.decoder = decoders.Sequence([
    decoders.Replace("▁", " "),
    decoders.ByteFallback(),
    decoders.Fuse(),
    decoders.Strip(content=" ", left=1, right=0),
])

out = os.path.join(ROOT, "web", "tokenizer.json")
tok.save(out)
print(f"saved {out}: vocab {tok.get_vocab_size()} (cap 10,000)")
assert tok.get_vocab_size() <= 10000, "vocab cap exceeded!"

# smoke: the instructor's exact failing sample must roundtrip
sample = "India's population is 1,428,627,663."
ids = tok.encode(sample).ids
back = tok.decode(ids)
vis = lambda s: "".join(s.split())
print(f"sample tokens: {len(ids)}; roundtrip: {back!r}")
assert vis(back) == vis(sample), "ROUNDTRIP GATE FAILED"
print("roundtrip gate: PASS")
