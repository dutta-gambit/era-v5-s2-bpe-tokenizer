/* Byte-Pair Encoding, from scratch. Dual-mode: node module + browser global (window.BPE).
 *
 * Conventions:
 *  - A "word" (the ratio denominator) is a maximal run of non-whitespace (text.split(/\s+/)).
 *  - Symbols are akshara clusters via the table-driven segmenter below (NOT
 *    Intl.Segmenter — ICU versions disagree on Indic conjuncts), so one Indic akshara
 *    is one base symbol and results are identical on every engine.
 *  - SentencePiece-style: whitespace runs normalize to the marker U+2581 (shown as an
 *    underscore-like block), one before every word. Merges MAY cross word boundaries
 *    (SentencePiece's split_by_whitespace=false), so frequent phrases can become single
 *    tokens. This is a deliberate, documented choice: with word-internal merges only,
 *    Xi <= 1.2 for all four corpora is unsatisfiable at 10,000 vocab (see
 *    src/analyze-wordbound.js and the widget's "why" panel).
 *  - Deterministic: ties on pair frequency break lexicographically; application replays
 *    merges in rank order, leftmost occurrence first.
 *
 * trainWords implements the classic word-internal variant — kept for the infeasibility
 * analysis only; it is not the shipped tokenizer.
 */
(function () {
  'use strict';
  var SEP = '';  // pair-key separator; never occurs in any corpus
  var MARK = '▁'; // the SentencePiece space marker

  /* Akshara-cluster segmentation — the honest base unit for Indic scripts, where one
   * user-perceived character (consonant + virama + consonant + vowel sign ...) is built
   * from 2-4 code points. Deliberately implemented HERE, not via Intl.Segmenter: ICU
   * versions disagree on Indic conjunct clusters (Unicode 15.1 GB9c), which changed
   * token counts by ±1 between node and Chrome. This table-driven segmenter is
   * engine-independent and fully auditable.
   *
   * Rules: a cluster starts at any base code point; EXTEND code points (matras, nukta,
   * anusvara/visarga, viramas, combining accents, ZWJ/ZWNJ, variation selectors) attach
   * to the current cluster; and a base code point ALSO attaches when the cluster's last
   * non-joiner code point is a virama (conjunct formation, e.g. \u0C37\u0C4D\u0C1F). */
  var EXTEND_RANGES = [
    [0x0300, 0x036F], // combining diacritics (Latin)
    [0x0900, 0x0903], [0x093A, 0x093C], [0x093E, 0x094F], [0x0951, 0x0957], [0x0962, 0x0963], // Devanagari signs/matras/virama
    [0x0980, 0x0983], [0x09BC, 0x09BC], [0x09BE, 0x09C4], [0x09C7, 0x09C8], [0x09CB, 0x09CD], [0x09D7, 0x09D7], [0x09E2, 0x09E3], // Bengali
    [0x0C00, 0x0C04], [0x0C3C, 0x0C3C], [0x0C3E, 0x0C44], [0x0C46, 0x0C48], [0x0C4A, 0x0C4D], [0x0C55, 0x0C56], [0x0C62, 0x0C63], // Telugu
    [0x200C, 0x200D], // ZWNJ / ZWJ
    [0xFE00, 0xFE0F], // variation selectors
  ];
  var VIRAMA = { 0x094D: 1, 0x09CD: 1, 0x0C4D: 1 };
  function isExtend(cp) {
    for (var i = 0; i < EXTEND_RANGES.length; i++) {
      if (cp >= EXTEND_RANGES[i][0] && cp <= EXTEND_RANGES[i][1]) return true;
    }
    return false;
  }
  function graphemes(str) {
    var cps = Array.from(str);
    var out = [];
    var cur = '';
    var lastSolid = 0; // last non-ZWJ/ZWNJ code point of the current cluster
    for (var i = 0; i < cps.length; i++) {
      var cp = cps[i].codePointAt(0);
      if (cur === '') { cur = cps[i]; lastSolid = cp; continue; }
      if (isExtend(cp)) {
        cur += cps[i];
        if (cp !== 0x200C && cp !== 0x200D) lastSolid = cp;
      } else if (VIRAMA[lastSolid]) {
        cur += cps[i]; // conjunct: consonant joins the cluster after a virama
        lastSolid = cp;
      } else {
        out.push(cur);
        cur = cps[i];
        lastSolid = cp;
      }
    }
    if (cur !== '') out.push(cur);
    return out;
  }

  /* text -> symbol stream: collapse whitespace, prefix every word with MARK */
  function textToSymbols(text) {
    var norm = text.trim().replace(/\s+/g, ' ');
    if (!norm) return [];
    return graphemes(MARK + norm.replace(/ /g, MARK));
  }

  function countWords(text) {
    return text.split(/\s+/).filter(Boolean).length;
  }

  /* ---------------- training ---------------- */

  /* Shipped variant: stream with MARK, chunked at sentence enders purely to keep the
   * pair bookkeeping cheap (chunk boundaries only hide a handful of cross-sentence
   * pairs during TRAINING; the shipped tokenizer applies merges over the whole stream).
   * Merges run down to pair count 1 (memorization is legitimate when the evaluation
   * corpus IS the training corpus; ties break lexicographically for determinism). Returns:
   *   merges:      [[left, right], ...] in learned order
   *   tokenTotals: tokenTotals[k] = training-chunk token total after k+1 merges
   *   tokens0:     token count before any merge
   *   words:       word count of the corpus
   */
  function trainStream(text, maxMerges, onProgress) {
    var norm = text.trim().replace(/\s+/g, ' ');
    var chunks = norm.split(/(?<=[.।?!])\s+/u).filter(Boolean); // ., danda, ?, !
    var freq = new Map();
    for (var i = 0; i < chunks.length; i++) {
      freq.set(chunks[i], (freq.get(chunks[i]) || 0) + 1);
    }
    var types = [];
    freq.forEach(function (f, w) { types.push({ sym: textToSymbols(w), f: f }); });
    return coreTrain(types, countWords(text), maxMerges, onProgress);
  }

  /* Classic word-internal variant (analysis only). */
  function trainWords(text, maxMerges, onProgress) {
    var words = text.split(/\s+/).filter(Boolean);
    var freq = new Map();
    for (var i = 0; i < words.length; i++) {
      freq.set(words[i], (freq.get(words[i]) || 0) + 1);
    }
    var types = [];
    freq.forEach(function (f, w) { types.push({ sym: Array.from(w), f: f }); });
    return coreTrain(types, words.length, maxMerges, onProgress);
  }

  /* shared trainer over weighted symbol-sequence types */
  function coreTrain(types, wordCount, maxMerges, onProgress) {
    var total = 0, i;
    for (i = 0; i < types.length; i++) total += types[i].f * types[i].sym.length;
    var tokens0 = total;

    var pairCount = new Map();   // key -> weighted count
    var pairTypes = new Map();   // key -> Set of type indices containing it

    function addTypePairs(idx) {
      var t = types[idx], s = t.sym, f = t.f;
      for (var j = 0; j + 1 < s.length; j++) {
        var key = s[j] + SEP + s[j + 1];
        pairCount.set(key, (pairCount.get(key) || 0) + f);
        var set = pairTypes.get(key);
        if (!set) { set = new Set(); pairTypes.set(key, set); }
        set.add(idx);
      }
    }
    function removeTypePairs(idx) {
      var t = types[idx], s = t.sym, f = t.f;
      for (var j = 0; j + 1 < s.length; j++) {
        var key = s[j] + SEP + s[j + 1];
        var c = (pairCount.get(key) || 0) - f;
        if (c <= 0) { pairCount.delete(key); pairTypes.delete(key); }
        else {
          pairCount.set(key, c);
          var set = pairTypes.get(key);
          if (set) set.delete(idx);
        }
      }
    }
    for (i = 0; i < types.length; i++) addTypePairs(i);

    var merges = [];
    var tokenTotals = [];

    for (var k = 0; k < maxMerges; k++) {
      var bestKey = null, bestCount = 0;
      pairCount.forEach(function (c, key) {
        if (c > bestCount || (c === bestCount && bestKey !== null && key < bestKey)) {
          bestCount = c; bestKey = key;
        }
      });
      if (bestKey === null) break; // no adjacent pairs left at all

      var p = bestKey.split(SEP);
      var a = p[0], b = p[1], merged = a + b;
      merges.push([a, b]);

      var affected = Array.from(pairTypes.get(bestKey));
      for (var ai = 0; ai < affected.length; ai++) {
        var idx = affected[ai];
        var t = types[idx];
        removeTypePairs(idx);
        var s = t.sym, ns = [], occ = 0;
        for (var j = 0; j < s.length; j++) {
          if (j + 1 < s.length && s[j] === a && s[j + 1] === b) { ns.push(merged); j++; occ++; }
          else ns.push(s[j]);
        }
        t.sym = ns;
        total -= t.f * occ;
        addTypePairs(idx);
      }
      tokenTotals.push(total);
      if (onProgress && (k + 1) % 500 === 0) onProgress(k + 1, total);
    }
    return { merges: merges, tokenTotals: tokenTotals, tokens0: tokens0, words: wordCount };
  }

  /* ---------------- application (exact replay over the full stream) ---------------- */

  /* Apply `merges` (in rank order, leftmost occurrence first) to the whole symbol
   * stream of `text`. Returns the token array. This IS the shipped tokenizer: every
   * reported number comes from this function.
   */
  function applyStream(merges, text) {
    var sym = textToSymbols(text);
    var n = sym.length;
    if (n === 0) return [];
    var next = new Int32Array(n), prev = new Int32Array(n);
    var alive = new Uint8Array(n);
    for (var i = 0; i < n; i++) { next[i] = i + 1; prev[i] = i - 1; alive[i] = 1; }

    var pairPos = new Map(); // key -> Set of left positions
    function reg(i) {
      var j = next[i];
      if (i < 0 || j >= n) return;
      var key = sym[i] + SEP + sym[j];
      var set = pairPos.get(key);
      if (!set) { set = new Set(); pairPos.set(key, set); }
      set.add(i);
    }
    function unreg(i) {
      var j = next[i];
      if (i < 0 || j >= n) return;
      var key = sym[i] + SEP + sym[j];
      var set = pairPos.get(key);
      if (set) { set.delete(i); if (set.size === 0) pairPos.delete(key); }
    }
    for (i = 0; i + 1 < n; i++) reg(i);

    for (var r = 0; r < merges.length; r++) {
      var a = merges[r][0], b = merges[r][1];
      var key = a + SEP + b;
      var set = pairPos.get(key);
      if (!set || set.size === 0) continue;
      var positions = Array.from(set).sort(function (x, y) { return x - y; });
      for (var pi = 0; pi < positions.length; pi++) {
        var pos = positions[pi];
        if (!alive[pos] || sym[pos] !== a) continue;
        var j = next[pos];
        if (j >= n || !alive[j] || sym[j] !== b) continue;
        // deregister the three affected pairs
        var pl = prev[pos], jn = next[j];
        if (pl >= 0) unreg(pl);
        unreg(pos);          // (pos, j)
        unreg(j);            // (j, jn) — uses next[j], still valid
        // merge
        sym[pos] = a + b;
        alive[j] = 0;
        next[pos] = jn;
        if (jn < n) prev[jn] = pos;
        // re-register the two new pairs
        if (pl >= 0) reg(pl);
        reg(pos);
      }
    }

    var out = [];
    for (i = 0; i >= 0 && i < n; i = next[i]) if (alive[i]) out.push(sym[i]);
    return out;
  }

  /* words, tokens, fertility — the assignment's Xi, computed by the shipped pipeline */
  function stats(merges, text) {
    var words = countWords(text);
    var tokens = applyStream(merges, text).length;
    return { words: words, tokens: tokens, fertility: tokens / words };
  }

  function baseChars(texts) {
    var set = new Set([MARK]);
    for (var i = 0; i < texts.length; i++) {
      var arr = graphemes(texts[i].replace(/\s+/g, ' '));
      for (var j = 0; j < arr.length; j++) if (!/^\s$/.test(arr[j])) set.add(arr[j]);
    }
    return set;
  }

  var api = {
    trainStream: trainStream,
    trainWords: trainWords,
    applyStream: applyStream,
    stats: stats,
    baseChars: baseChars,
    textToSymbols: textToSymbols,
    graphemes: graphemes,
    countWords: countWords,
    SEP: SEP,
    MARK: MARK,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') window.BPE = api;
})();
