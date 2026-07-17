#!/usr/bin/env node
/* Grader CLI: tokenize any file(s) with the shipped tokenizer and print the ratios,
 * plus a faithful-roundtrip check on every file.
 *
 *   node src/tokenize-cli.js corpus/en.txt corpus/hi.txt corpus/te.txt corpus/bn.txt
 *   node src/tokenize-cli.js --tokens somefile.txt      # also dump the tokens
 *
 * Reads web/tokenizer.json (HuggingFace format) and mirrors its pipeline exactly:
 * Metaspace pre-tokenization + BPE with byte_fallback. scripts/verify_gate.py proves
 * this CLI and the python `tokenizers` library agree bit-for-bit. Any node >= 14.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const BPE = require('./bpe.js');

const args = process.argv.slice(2);
const dumpTokens = args.includes('--tokens');
const files = args.filter((a) => a !== '--tokens');
if (files.length === 0) {
  console.error('usage: node src/tokenize-cli.js [--tokens] <file> [file...]');
  process.exit(1);
}

const hf = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'web', 'tokenizer.json'), 'utf8'));
const merges = hf.model.merges.map((m) =>
  Array.isArray(m) ? m : [m.slice(0, m.indexOf(' ')), m.slice(m.indexOf(' ') + 1)]);
const vocabKeys = Object.keys(hf.model.vocab);
const tok = BPE.makeWordTokenizer(merges, vocabKeys);
console.error(`tokenizer: ${vocabKeys.length} vocab tokens, ${merges.length} merge rules (HF format)`);

const visible = (s) => s.replace(/\s+/g, '');
const results = [];
for (const f of files) {
  const text = fs.readFileSync(f, 'utf8');
  const words = BPE.countWords(text);
  const tokens = BPE.encode(tok, text);
  const roundtrip = visible(BPE.decode(tokens)) === visible(text);
  results.push({ file: f, words, tokens: tokens.length, ratio: tokens.length / words, roundtrip });
  if (dumpTokens) fs.writeFileSync(f + '.tokens', tokens.join('\n'));
}
for (const r of results) {
  console.log(`${r.file}: words=${r.words} tokens=${r.tokens} X=${r.ratio.toFixed(6)} roundtrip=${r.roundtrip ? 'faithful' : 'LOSSY'}`);
}
if (results.length > 1) {
  const xs = results.map((r) => r.ratio);
  const spread = Math.max(...xs) - Math.min(...xs);
  console.log(`spread=${spread.toFixed(6)} score=1000/spread=${(1000 / spread).toFixed(1)}`);
}
if (results.some((r) => !r.roundtrip)) process.exit(1);
