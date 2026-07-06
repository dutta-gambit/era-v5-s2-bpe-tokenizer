/* Trains the 4-language BPE tokenizer and optimizes the merge budget split.
 *
 * The four scripts are (nearly) disjoint, so each language gets its own BPE merge
 * table; the shipped tokenizer is the interleaved union. A language's fertility
 * Xi = tokens/words on its corpus falls as its merge budget grows, so we:
 *   1. train each language far past what it will get (fertility-vs-merges curves),
 *   2. binary-search the lowest common fertility target T every language can reach
 *      within the total vocab budget (10,000 incl. base symbols),
 *   3. spend leftover budget on whichever language currently has max fertility,
 *   4. hill-climb budgets against the REAL unified tokenizer (dedupe + cross-script
 *      interactions make it differ slightly from the per-language curves),
 * minimizing spread = Xmax − Xmin, which the assignment scores as 1000/spread.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const BPE = require('./bpe.js');

const ROOT = path.join(__dirname, '..');
const LANGS = ['en', 'hi', 'te', 'bn'];
const VOCAB_CAP = 10000;
const MAX_TRAIN = 6500;

const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, 'corpus', 'manifest.json'), 'utf8'));
const texts = {};
for (const l of LANGS) texts[l] = fs.readFileSync(path.join(ROOT, 'corpus', `${l}.txt`), 'utf8');

/* ---- 1. per-language training ---- */
const trained = {};
for (const l of LANGS) {
  const t0 = Date.now();
  trained[l] = BPE.trainStream(texts[l], MAX_TRAIN);
  const tr = trained[l];
  console.log(`${l}: ${tr.merges.length} merges in ${((Date.now() - t0) / 1000).toFixed(1)}s ` +
    `(words ${tr.words}, curve fertility ${(tr.tokenTotals[tr.tokenTotals.length - 1] / tr.words).toFixed(4)} at max)`);
}

const base = BPE.baseChars(LANGS.map((l) => texts[l]));
console.log(`base symbols (union incl. word marker): ${base.size}`);
const MERGE_BUDGET = VOCAB_CAP - base.size;

/* fertility of language l after k merges, from its own (chunk-approx) curve */
function curveFert(l, k) {
  const tr = trained[l];
  if (k <= 0) return tr.tokens0 / tr.words;
  const kk = Math.min(k, tr.tokenTotals.length);
  return tr.tokenTotals[kk - 1] / tr.words;
}
/* merges needed to reach fertility <= T (Infinity if unreachable) */
function needK(l, T) {
  const tr = trained[l];
  if (tr.tokens0 / tr.words <= T) return 0;
  let lo = 1, hi = tr.tokenTotals.length;
  if (curveFert(l, hi) > T) return Infinity;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (curveFert(l, mid) <= T) hi = mid; else lo = mid + 1; }
  return lo;
}

/* ---- 2. binary-search the common target T (curve approximation) ----
 * The naive sum ignores cross-language rule dedupe (Latin/digit chains inside the
 * Indic pages re-learn English rules), so T may land slightly above 1.2 here; the
 * hill climb below works against the REAL unified vocab and recovers the slack. */
let tLo = 0.05, tHi = 2.5;
for (let it = 0; it < 60; it++) {
  const mid = (tLo + tHi) / 2;
  const need = LANGS.reduce((s, l) => s + needK(l, mid), 0);
  if (need <= MERGE_BUDGET && need !== Infinity) tHi = mid; else tLo = mid;
}
const T = tHi;
if (T > 1.2) console.warn(`curve-level T ${T.toFixed(4)} > 1.2 — relying on dedupe slack in the hill climb`);
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

/* ---- 4. unified tokenizer + real evaluation + hill climb ---- */
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
  return { merges, vocabSize: vocab.size };
}

function evaluate(budgets) {
  const u = buildUnified(budgets);
  if (u.vocabSize > VOCAB_CAP) return { spread: Infinity, maxF: Infinity, u };
  const per = {};
  for (const l of LANGS) per[l] = BPE.stats(u.merges, texts[l]);
  const fvals = LANGS.map((l) => per[l].fertility);
  return { spread: Math.max(...fvals) - Math.min(...fvals), maxF: Math.max(...fvals), per, u };
}

let cur = evaluate(k);
console.log('initial real spread:', cur.spread.toExponential(3), 'vocab:', cur.u.vocabSize,
  'maxF:', cur.maxF.toFixed(4),
  'fertilities:', LANGS.map((l) => cur.per[l].fertility.toFixed(4)).join(' '));

function argmax() { return LANGS.reduce((a, l) => (cur.per[l].fertility > cur.per[a].fertility ? l : a)); }
function argmin() { return LANGS.reduce((a, l) => (cur.per[l].fertility < cur.per[a].fertility ? l : a)); }

/* Objective: the assignment scores 1000/spread, so minimize spread, full stop.
 * (X <= 1.2 is provably unreachable at this cap — see analysis.json — so it cannot
 * gate the search; the vocab cap is the only hard constraint.) */
function cost(e) {
  return e.u.vocabSize > VOCAB_CAP ? Infinity : e.spread;
}

const steps = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1];
let evals = 0;
for (let pass = 0; pass < 8; pass++) {
  let improvedAny = false;
  for (const step of steps) {
    let improved = true;
    while (improved) {
      improved = false;
      const hi = argmax(), lo = argmin();
      // (a) move budget min -> max
      if (hi !== lo && k[lo] >= step) {
        k[lo] -= step; k[hi] += step;
        const trial = evaluate(k); evals++;
        if (cost(trial) < cost(cur) - 1e-12) { cur = trial; improved = improvedAny = true; continue; }
        k[lo] += step; k[hi] -= step;
      }
      // (b) pure addition to max (dedupe slack can allow it)
      k[hi] += step;
      const trial2 = evaluate(k); evals++;
      if (cost(trial2) < cost(cur) - 1e-12) {
        cur = trial2; improved = improvedAny = true; continue;
      }
      k[hi] -= step;
    }
    // (c) fine polish at small steps: all ordered pairs, catches min<->max wedges
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
if (cur.maxF > 1.2) {
  console.warn(`note: max fertility ${cur.maxF.toFixed(4)} > 1.2 — expected; see analysis.json`);
}

const per = cur.per;
const fsSorted = LANGS.map((l) => ({ lang: l, x: per[l].fertility })).sort((a, b) => b.x - a.x);
const spread = cur.spread;
console.log('final budgets', k, 'vocab', cur.u.vocabSize);
for (const l of LANGS) console.log(`  X_${l} = ${per[l].tokens}/${per[l].words} = ${per[l].fertility.toFixed(6)}`);
console.log(`spread = ${spread.toFixed(6)}  score = ${(1000 / spread).toFixed(1)}`);

/* ---- 5. emit artifacts ---- */
const WEB = path.join(ROOT, 'web');
fs.mkdirSync(path.join(WEB, 'corpus'), { recursive: true });
fs.mkdirSync(path.join(WEB, 'download'), { recursive: true });

const unified = cur.u;
const baseArr = Array.from(base).sort();
const seenTok = new Set(baseArr);
const vocabList = baseArr.slice();
for (const m of unified.merges) {
  const s = m[0] + m[1];
  if (!seenTok.has(s)) { seenTok.add(s); vocabList.push(s); }
}

fs.writeFileSync(path.join(WEB, 'tokenizer.json'), JSON.stringify({
  format: 'bpe-v1',
  pipeline: 'collapse whitespace runs to single spaces; prefix every word with U+2581; split into akshara/grapheme clusters (table-driven segmenter defined in bpe.js — independent of ICU/browser version); apply merges greedily in rank order (leftmost first) over the whole stream',
  wordDefinition: 'a word = maximal run of non-whitespace in the raw text (text.split(/\\s+/))',
  vocabSize: unified.vocabSize,
  baseChars: baseArr,
  merges: unified.merges,
}));

fs.writeFileSync(path.join(WEB, 'stats.json'), JSON.stringify({
  builtAt: new Date().toISOString(),
  vocabCap: VOCAB_CAP,
  vocabSize: unified.vocabSize,
  baseSymbols: base.size,
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

/* ---- 6. constraint analysis for the widget: what would Xi <= 1.2 cost? ---- */
const grid = [];
for (let t = 1.0; t <= 2.21; t += 0.05) {
  const T2 = Math.round(t * 100) / 100;
  const need = LANGS.reduce((s, l) => {
    const nk = needK(l, T2);
    return nk === Infinity ? Infinity : s + nk;
  }, 0);
  grid.push({ target: T2, vocabNeeded: need === Infinity ? null : need + base.size });
}
console.log('analysis: word-internal variant floors (for the "why" panel)...');
const wordInternal = {};
const baseCp = (() => { // code-point base for the classic variant
  const s = new Set();
  for (const l of LANGS) for (const c of Array.from(texts[l])) if (!/\s/.test(c)) s.add(c);
  return s.size;
})();
for (const l of LANGS) {
  const tw = BPE.trainWords(texts[l], 9000);
  let k12 = Infinity;
  for (let i = 0; i < tw.tokenTotals.length; i++) {
    if (tw.tokenTotals[i] / tw.words <= 1.2) { k12 = i + 1; break; }
  }
  // count>=2 floor = last merge whose application reduced the total by >= 2
  let floorIdx = tw.tokenTotals.length;
  for (let i = 1; i < tw.tokenTotals.length; i++) {
    if (tw.tokenTotals[i - 1] - tw.tokenTotals[i] < 2) { floorIdx = i; break; }
  }
  wordInternal[l] = {
    mergesTo12: k12,
    countGe2Floor: {
      merges: floorIdx,
      fertility: (floorIdx > 0 ? tw.tokenTotals[floorIdx - 1] : tw.tokens0) / tw.words,
    },
  };
}
const wiTotal = LANGS.reduce((s, l) => s + wordInternal[l].mergesTo12, 0);
fs.writeFileSync(path.join(WEB, 'analysis.json'), JSON.stringify({
  note: 'vocabNeeded = base symbols + sum of per-language merges to reach the target fertility (curve-exact; ignores ~51 slots of cross-language rule dedupe)',
  baseSymbols: base.size,
  grid,
  vocabNeededFor12: grid.find((g) => g.target === 1.2),
  minCommonFertilityAt10k: cur.maxF,
  wordInternal: {
    baseCodePoints: baseCp,
    perLanguage: wordInternal,
    vocabNeededFor12: baseCp + wiTotal,
  },
}, null, 2));
console.log(`analysis.json: grapheme-BPE vocab needed for 1.2 = ${grid.find((g) => g.target === 1.2)?.vocabNeeded}; ` +
  `word-internal = ${baseCp + wiTotal}`);

for (const l of LANGS) fs.copyFileSync(path.join(ROOT, 'corpus', `${l}.txt`), path.join(WEB, 'corpus', `${l}.txt`));
fs.copyFileSync(path.join(ROOT, 'corpus', 'manifest.json'), path.join(WEB, 'corpus', 'manifest.json'));
fs.writeFileSync(path.join(WEB, 'download', 'tokens.txt'), vocabList.join('\n'));
fs.writeFileSync(path.join(WEB, 'download', 'merges.txt'), unified.merges.map((m) => `${m[0]} ${m[1]}`).join('\n'));
fs.copyFileSync(path.join(ROOT, 'src', 'bpe.js'), path.join(WEB, 'bpe.js'));

console.log(`written: web/tokenizer.json, web/stats.json, web/download/tokens.txt (${vocabList.length} tokens), merges.txt`);
if (vocabList.length !== unified.vocabSize) {
  console.warn(`WARNING: vocab list length ${vocabList.length} != vocabSize ${unified.vocabSize}`);
}
