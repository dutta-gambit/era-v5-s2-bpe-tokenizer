#!/usr/bin/env python3
"""HTML -> Markdown corpora, two independent renderings per language.

Variant A (markdownify) is the TRAINING corpus; variant B (html2text) is the HOLDOUT —
a different converter's rendering of the same articles, standing in for the
instructor's unknown "secret sauce" cleaning. Links/URLs are preserved in both
(the reference solution evaluates faithful markdown; "you can't throw away text").

Faithful units (the reference denominator): [\\p{L}\\p{M}\\p{N}]+|[^\\s\\p{L}\\p{M}\\p{N}]
— each letter/mark/number run is one unit, each other visible character is one unit.
"""
import hashlib
import json
import os
import re

import html2text
import regex
from markdownify import markdownify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")

def strip_nontext(html: str) -> str:
    html = re.sub(r"<head\b.*?</head>", "", html, flags=re.S)
    html = re.sub(r"<style\b[^>]*>.*?</style>", "", html, flags=re.S)
    html = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.S)
    html = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    return html

def tidy(md: str) -> str:
    md = md.replace("\ufeff", "")  # BOM/ZWNBSP: invisible, and JS/python disagree on \s membership
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = "\n".join(line.rstrip() for line in md.splitlines())
    return md.strip() + "\n"

os.makedirs(os.path.join(ROOT, "corpus-md"), exist_ok=True)
manifest = []
for lang in ["en", "hi", "te", "bn"]:
    with open(os.path.join(ROOT, "data", "html", f"{lang}.html"), encoding="utf-8") as f:
        html = strip_nontext(f.read())

    md_a = tidy(markdownify(html, heading_style="ATX"))

    h = html2text.HTML2Text()
    h.body_width = 0
    h.ignore_links = False
    h.ignore_images = False
    h.unicode_snob = True
    h.mark_code = False
    md_b = tidy(h.handle(html))

    for variant, md in (("A", md_a), ("B", md_b)):
        path = os.path.join(ROOT, "corpus-md", f"{lang}.{variant}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        units = len(UNIT_RE.findall(md))
        words = len(md.split())
        manifest.append({
            "lang": lang, "variant": variant, "chars": len(md),
            "words": words, "faithfulUnits": units,
            "sha256": hashlib.sha256(md.encode()).hexdigest(),
        })
        print(f"{lang}.{variant}: {len(md):,} chars, {words:,} words, {units:,} faithful units")

with open(os.path.join(ROOT, "corpus-md", "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print("corpus-md/manifest.json written")
