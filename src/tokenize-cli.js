#!/usr/bin/env node
/* Grader CLI: tokenize any file(s) with the shipped tokenizer and print the ratios.
 *
 *   node src/tokenize-cli.js corpus/en.txt corpus/hi.txt corpus/te.txt corpus/bn.txt
 *   node src/tokenize-cli.js --tokens somefile.txt      # also dump the tokens
 *
 * Works on any node >= 14 (the segmenter is self-contained in bpe.js — no ICU
 * dependence). Words = whitespace-split; the ratio printed is X = tokens / words,
 * exactly as defined on the widget.
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

const tok = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'web', 'tokenizer.json'), 'utf8'));
const vocab = new Set(tok.baseChars);
for (const m of tok.merges) vocab.add(m[0] + m[1]);
console.error(`tokenizer: ${vocab.size} vocab tokens (${tok.baseChars.length} base + ${tok.merges.length} merge rules)`);

const results = [];
for (const f of files) {
  const text = fs.readFileSync(f, 'utf8');
  const words = BPE.countWords(text);
  const tokens = BPE.applyStream(tok.merges, text);
  results.push({ file: f, words, tokens: tokens.length, ratio: tokens.length / words });
  if (dumpTokens) fs.writeFileSync(f + '.tokens', tokens.join('\n'));
}
for (const r of results) {
  console.log(`${r.file}: words=${r.words} tokens=${r.tokens} X=${r.ratio.toFixed(6)}`);
}
if (results.length > 1) {
  const xs = results.map((r) => r.ratio);
  const spread = Math.max(...xs) - Math.min(...xs);
  console.log(`spread=${spread.toFixed(6)} score=1000/spread=${(1000 / spread).toFixed(1)}`);
}
