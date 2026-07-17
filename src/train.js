/* Trains the 4-language BPE tokenizer and optimizes the merge budget split.
 *
 * SHIPPED PIPELINE (HuggingFace-compatible, decode-gate safe):
 *   Metaspace pre-tokenization (every word becomes ▁word), BPE merges WITHIN words
 *   over code points, byte_fallback for anything outside the vocab. This is exactly
 *   what tokenizers.Tokenizer.from_file() reconstructs from web/tokenizer.json, so
 *   the grader gets encode() AND decode() with a faithful visible-character roundtrip.
 *
 * The four scripts are (nearly) disjoint, so each language gets its own merge table;
 * the shipped tokenizer is the interleaved union. Budgets are chosen by binary-searching
 * a common fertility target on the training curves, then hill-climbing against the real
 * unified tokenizer to minimize spread = Xmax − Xmin (score = 1000/spread).
 *
 * Outputs build/hf-input.json (vocab + merges); scripts/build_hf.py turns that into the
 * canonical web/tokenizer.json via the python `tokenizers` library, and
 * scripts/verify_gate.py replays the grader's checks against it.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const BPE = require('./bpe.js');

const ROOT = path.join(__dirname, '..');
const LANGS = ['en', 'hi', 'te', 'bn'];
const VOCAB_CAP = 10000;
const N_BYTE_TOKENS = 256;
const MAX_TRAIN = 7000;

const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, 'corpus', 'manifest.json'), 'utf8'));
const texts = {};
for (const l of LANGS) texts[l] = fs.readFileSync(path.join(ROOT, 'corpus', `${l}.txt`), 'utf8');

/* ---- 1. per-language training (word-internal, ▁-prefixed, code points) ---- */
const trained = {};
for (const l of LANGS) {
  const t0 = Date.now();
  trained[l] = BPE.trainWords(texts[l], MAX_TRAIN, null, true);
  const tr = trained[l];
  console.log(`${l}: ${tr.merges.length} merges in ${((Date.now() - t0) / 1000).toFixed(1)}s ` +
    `(words ${tr.words}, curve fertility ${(tr.tokenTotals[tr.tokenTotals.length - 1] / tr.words).toFixed(4)} at max)`);
}

/* base symbols = every code point of every ▁-prefixed word */
const base = new Set([BPE.MARK]);
for (const l of LANGS) {
  for (const c of Array.from(texts[l])) if (!/\s/.test(c)) base.add(c);
}
console.log(`base code points (incl. ▁): ${base.size}; byte tokens: ${N_BYTE_TOKENS}`);
const MERGE_BUDGET = VOCAB_CAP - base.size - N_BYTE_TOKENS;

/* fertility of language l after k merges, from its own curve */
function curveFert(l, k) {
  const tr = trained[l];
  if (k <= 0) return tr.tokens0 / tr.words;
  const kk = Math.min(k, tr.tokenTotals.length);
  return tr.tokenTotals[kk - 1] / tr.words;
}
function needK(l, T) {
  const tr = trained[l];
  if (tr.tokens0 / tr.words <= T) return 0;
  let lo = 1, hi = tr.tokenTotals.length;
  if (curveFert(l, hi) > T) return Infinity;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (curveFert(l, mid) <= T) hi = mid; else lo = mid + 1; }
  return lo;
}

/* ---- 2. binary-search the common target T (curve approximation) ---- */
let tLo = 0.5, tHi = 3.0;
for (let it = 0; it < 60; it++) {
  const mid = (tLo + tHi) / 2;
  const need = LANGS.reduce((s, l) => s + needK(l, mid), 0);
  if (need <= MERGE_BUDGET && need !== Infinity) tHi = mid; else tLo = mid;
}
const T = tHi;
const k = {};
for (const l of LANGS) k[l] = needK(l, T);
console.log(`curve target T = ${T.toFixed(4)}, budgets`, k,
  `(sum ${LANGS.reduce((s, l) => s + k[l], 0)} / ${MERGE_BUDGET})`);

/* ---- 3. spend leftover on the worst language (per curves) ---- */
let left = MERGE_BUDGET - LANGS.reduce((s, l) => s + k[l], 0);
while (left > 0) {
  let worst = null, wf = -1;
  for (const l of LANGS) {
    if (k[l] >= trained[l].tokenTotals.length) continue;
    const f = curveFert(l, k[l]);
    if (f > wf) { wf = f; worst = l; }
  }
  if (!worst) break;
  k[worst]++; left--;
}

/* ---- 4. unified tokenizer + real evaluation + hill climb (pure spread) ---- */
function buildUnified(budgets) {
  const seen = new Set();
  const merges = [];
  const maxK = Math.max(...LANGS.map((l) => budgets[l]));
  for (let r = 0; r < maxK; r++) {
    for (const l of LANGS) {
      if (r >= budgets[l]) continue;
      const m = trained[l].merges[r];
      const key = m[0] + BPE.SEP + m[1];
      if (!seen.has(key)) { seen.add(key); merges.push(m); }
    }
  }
  const vocab = new Set(base);
  for (const m of merges) vocab.add(m[0] + m[1]);
  return { merges, vocabSize: vocab.size + N_BYTE_TOKENS };
}

function evaluate(budgets) {
  const u = buildUnified(budgets);
  if (u.vocabSize > VOCAB_CAP) return { spread: Infinity, maxF: Infinity, u };
  const tok = BPE.makeWordTokenizer(u.merges, []); // corpus chars are all in base — no fallback needed here
  const per = {};
  for (const l of LANGS) per[l] = BPE.statsWords(tok, texts[l]);
  const fvals = LANGS.map((l) => per[l].fertility);
  return { spread: Math.max(...fvals) - Math.min(...fvals), maxF: Math.max(...fvals), per, u };
}

let cur = evaluate(k);
console.log('initial real spread:', cur.spread.toExponential(3), 'vocab:', cur.u.vocabSize,
  'fertilities:', LANGS.map((l) => cur.per[l].fertility.toFixed(4)).join(' '));

function argmax() { return LANGS.reduce((a, l) => (cur.per[l].fertility > cur.per[a].fertility ? l : a)); }
function argmin() { return LANGS.reduce((a, l) => (cur.per[l].fertility < cur.per[a].fertility ? l : a)); }
function cost(e) { return e.u.vocabSize > VOCAB_CAP ? Infinity : e.spread; }

const steps = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1];
let evals = 0;
for (let pass = 0; pass < 8; pass++) {
  let improvedAny = false;
  for (const step of steps) {
    let improved = true;
    while (improved) {
      improved = false;
      const hi = argmax(), lo = argmin();
      if (hi !== lo && k[lo] >= step) {
        k[lo] -= step; k[hi] += step;
        const trial = evaluate(k); evals++;
        if (cost(trial) < cost(cur) - 1e-12) { cur = trial; improved = improvedAny = true; continue; }
        k[lo] += step; k[hi] -= step;
      }
      k[hi] += step;
      const trial2 = evaluate(k); evals++;
      if (cost(trial2) < cost(cur) - 1e-12) { cur = trial2; improved = improvedAny = true; continue; }
      k[hi] -= step;
    }
    if (step <= 8) {
      let polished = true;
      while (polished) {
        polished = false;
        for (const from of LANGS) {
          for (const to of LANGS) {
            if (from === to || k[from] < step) continue;
            k[from] -= step; k[to] += step;
            const t = evaluate(k); evals++;
            if (cost(t) < cost(cur) - 1e-12) { cur = t; polished = improvedAny = true; }
            else { k[from] += step; k[to] -= step; }
          }
        }
      }
    }
  }
  if (!improvedAny) break;
}
console.log(`hill climb done (${evals} evaluations)`);

const per = cur.per;
const fsSorted = LANGS.map((l) => ({ lang: l, x: per[l].fertility })).sort((a, b) => b.x - a.x);
const spread = cur.spread;
console.log('final budgets', k, 'vocab', cur.u.vocabSize);
for (const l of LANGS) console.log(`  X_${l} = ${per[l].tokens}/${per[l].words} = ${per[l].fertility.toFixed(6)}`);
console.log(`spread = ${spread.toFixed(6)}  score = ${(1000 / spread).toFixed(1)}`);

/* ---- 5. emit artifacts ---- */
const WEB = path.join(ROOT, 'web');
const BUILD = path.join(ROOT, 'build');
fs.mkdirSync(path.join(WEB, 'corpus'), { recursive: true });
fs.mkdirSync(path.join(WEB, 'download'), { recursive: true });
fs.mkdirSync(BUILD, { recursive: true });

const unified = cur.u;
/* vocab id order: 256 byte tokens, then base code points (sorted), then merge results in rank order */
const vocabList = [];
for (let b = 0; b < 256; b++) vocabList.push('<0x' + b.toString(16).toUpperCase().padStart(2, '0') + '>');
const baseArr = Array.from(base).sort();
vocabList.push(...baseArr);
const seenTok = new Set(vocabList);
for (const m of unified.merges) {
  const s = m[0] + m[1];
  if (!seenTok.has(s)) { seenTok.add(s); vocabList.push(s); }
}

fs.writeFileSync(path.join(BUILD, 'hf-input.json'), JSON.stringify({
  vocab: vocabList,
  merges: unified.merges,
}));

fs.writeFileSync(path.join(WEB, 'stats.json'), JSON.stringify({
  builtAt: new Date().toISOString(),
  vocabCap: VOCAB_CAP,
  vocabSize: vocabList.length,
  baseSymbols: base.size,
  byteTokens: N_BYTE_TOKENS,
  mergeRules: unified.merges.length,
  budgets: k,
  perLanguage: LANGS.map((l) => ({
    ...manifest.find((m) => m.lang === l),
    words: per[l].words,
    tokens: per[l].tokens,
    fertility: per[l].fertility,
    budget: k[l],
  })),
  sorted: fsSorted,
  spread,
  score: 1000 / spread,
}, null, 2));

/* ---- 6. constraint analysis for the widget ---- */
const grid = [];
for (let t = 1.0; t <= 2.61; t += 0.05) {
  const T2 = Math.round(t * 100) / 100;
  const need = LANGS.reduce((s, l) => {
    const nk = needK(l, T2);
    return nk === Infinity ? Infinity : s + nk;
  }, 0);
  grid.push({ target: T2, vocabNeeded: need === Infinity ? null : need + base.size + N_BYTE_TOKENS });
}
fs.writeFileSync(path.join(WEB, 'analysis.json'), JSON.stringify({
  note: 'vocabNeeded = byte tokens + base code points + sum of per-language merges to reach the target fertility (curve-exact; ignores a small cross-language dedupe credit)',
  baseSymbols: base.size,
  byteTokens: N_BYTE_TOKENS,
  grid,
  vocabNeededFor12: grid.find((g) => g.target === 1.2),
  minCommonFertilityAt10k: cur.maxF,
  streamGraphemeReference: {
    fertilityAt10k: 1.6194,
    note: 'our previous submission: SentencePiece-style cross-word merges over akshara clusters reached X ≈ 1.62 — but that pipeline has no HF-loadable decode, failing the roundtrip gate. The shipped word-internal pipeline trades a slightly different X for a grader-runnable encode/decode.',
  },
}, null, 2));
console.log(`analysis: vocab needed for 1.2 = ${grid.find((g) => g.target === 1.2)?.vocabNeeded}`);

for (const l of LANGS) fs.copyFileSync(path.join(ROOT, 'corpus', `${l}.txt`), path.join(WEB, 'corpus', `${l}.txt`));
fs.copyFileSync(path.join(ROOT, 'corpus', 'manifest.json'), path.join(WEB, 'corpus', 'manifest.json'));
fs.writeFileSync(path.join(WEB, 'download', 'tokens.txt'), vocabList.join('\n'));
fs.writeFileSync(path.join(WEB, 'download', 'merges.txt'), unified.merges.map((m) => `${m[0]} ${m[1]}`).join('\n'));
fs.copyFileSync(path.join(ROOT, 'src', 'bpe.js'), path.join(WEB, 'bpe.js'));

console.log(`written: build/hf-input.json (${vocabList.length} tokens), web/stats.json, web/analysis.json, downloads`);
console.log('next: .venv/bin/python scripts/build_hf.py && .venv/bin/python scripts/verify_gate.py');
