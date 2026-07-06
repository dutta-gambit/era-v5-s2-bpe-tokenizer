/* Widget logic. Every number shown is either loaded from the build artifacts
 * (stats.json / analysis.json / tokenizer.json) or recomputed live in this browser
 * with the same bpe.js that trained the tokenizer. */
(function () {
  'use strict';

  var LANGS = ['en', 'hi', 'te', 'bn'];
  var NATIVE = { en: 'India', hi: 'भारत', te: 'భారతదేశం', bn: 'ভারত' };
  var state = { stats: null, analysis: null, tokenizer: null, corpora: {}, manifest: null };

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function fmt(n) { return n.toLocaleString('en-IN'); }
  function f6(x) { return x.toFixed(6); }

  /* ---------------- data loading ---------------- */
  function fetchJSON(u) { return fetch(u).then(function (r) { return r.json(); }); }
  function fetchText(u) { return fetch(u).then(function (r) { return r.text(); }); }

  Promise.all([
    fetchJSON('stats.json'), fetchJSON('analysis.json'), fetchJSON('tokenizer.json'),
    fetchJSON('corpus/manifest.json'),
    fetchText('corpus/en.txt'), fetchText('corpus/hi.txt'),
    fetchText('corpus/te.txt'), fetchText('corpus/bn.txt'),
  ]).then(function (r) {
    state.stats = r[0]; state.analysis = r[1]; state.tokenizer = r[2]; state.manifest = r[3];
    state.corpora = { en: r[4], hi: r[5], te: r[6], bn: r[7] };
    renderHero(); renderStats(); renderScore(); renderConstraint(); renderDownloads();
    initPlayground();
    runVerification();
  }).catch(function (e) {
    $('verify-status').textContent = 'failed to load artifacts: ' + e;
  });

  function perLang(l) {
    return state.stats.perLanguage.find(function (p) { return p.lang === l; });
  }

  /* ---------------- hero ---------------- */
  function renderHero() {
    var s = state.stats;
    var root = $('hero-cards');
    root.innerHTML = '';
    LANGS.forEach(function (l) {
      var p = perLang(l);
      var c = el('div', 'hero-card');
      c.appendChild(el('div', 'hc-label', p.language + ' · X'));
      c.appendChild(el('div', 'hc-value', p.fertility.toFixed(4)));
      c.appendChild(el('div', 'hc-sub', fmt(p.tokens) + ' tokens / ' + fmt(p.words) + ' words'));
      root.appendChild(c);
    });
    var spread = el('div', 'hero-card');
    spread.appendChild(el('div', 'hc-label', 'spread X<sub>max</sub> − X<sub>min</sub>'));
    spread.appendChild(el('div', 'hc-value', s.spread.toFixed(6)));
    spread.appendChild(el('div', 'hc-sub', 'vocab ' + fmt(s.vocabSize) + ' / ' + fmt(s.vocabCap)));
    root.appendChild(spread);
    var score = el('div', 'hero-card score');
    score.appendChild(el('div', 'hc-label', 'self score · 1000 ÷ spread'));
    score.appendChild(el('div', 'hc-value', '≈ ' + Math.round(s.score).toLocaleString('en-IN')));
    score.appendChild(el('div', 'hc-sub', 'recomputed live below'));
    root.appendChild(score);
  }

  /* ---------------- stats table ---------------- */
  function renderStats() {
    var s = state.stats;
    var wrap = el('div', 'table-scroll');
    var t = el('table', 'data-table');
    t.appendChild(el('thead', null,
      '<tr><th>Language</th><th>Wikipedia revision</th><th>Words</th><th>Unique words</th>' +
      '<th>Tokens</th><th>X = tokens/words</th><th>Merge budget</th></tr>'));
    var tb = el('tbody');
    LANGS.forEach(function (l) {
      var p = perLang(l);
      var uniq = new Set(state.corpora[l].split(/\s+/).filter(Boolean)).size;
      var tr = el('tr');
      tr.appendChild(el('td', 'lang', p.language + '<span class="native">' + NATIVE[l] + '</span>'));
      tr.appendChild(el('td', null, p.revid + ' · ' + p.revTimestamp.slice(0, 10)));
      tr.appendChild(el('td', null, fmt(p.words)));
      tr.appendChild(el('td', null, fmt(uniq)));
      tr.appendChild(el('td', null, fmt(p.tokens)));
      var td = el('td', 'hl', f6(p.fertility));
      tr.appendChild(td);
      tr.appendChild(el('td', null, fmt(p.budget) + ' merges'));
      tb.appendChild(tr);
    });
    var trTot = el('tr');
    var totW = LANGS.reduce(function (a, l) { return a + perLang(l).words; }, 0);
    var totT = LANGS.reduce(function (a, l) { return a + perLang(l).tokens; }, 0);
    trTot.appendChild(el('td', 'lang', 'All four'));
    trTot.appendChild(el('td', null, '—'));
    trTot.appendChild(el('td', null, fmt(totW)));
    trTot.appendChild(el('td', null, '—'));
    trTot.appendChild(el('td', null, fmt(totT)));
    trTot.appendChild(el('td', null, '—'));
    trTot.appendChild(el('td', null, fmt(s.baseSymbols) + ' base + ' + fmt(s.mergeRules) + ' rules'));
    tb.appendChild(trTot);
    t.appendChild(tb);
    wrap.appendChild(t);
    $('stats-table').appendChild(wrap);
  }

  function renderScore() {
    var s = state.stats;
    var sorted = s.sorted; // [{lang, x}] desc
    var line1 = 'sorted:  ' + sorted.map(function (e) {
      return 'X<sub>' + e.lang + '</sub> = ' + f6(e.x);
    }).join(' <span class="op">≥</span> ');
    var xmax = sorted[0], xmin = sorted[sorted.length - 1];
    var line2 = 'X<sub>max</sub> − X<sub>min</sub> <span class="op">=</span> ' + f6(xmax.x) +
      ' <span class="op">−</span> ' + f6(xmin.x) + ' <span class="op">=</span> ' + s.spread.toFixed(6);
    var line3 = 'score <span class="op">=</span> 1000 <span class="op">÷</span> ' + s.spread.toFixed(6) +
      ' <span class="op">=</span> <span class="big">' + Math.round(s.score).toLocaleString('en-IN') + '</span>';
    var box = el('div', 'score-box', line1 + '<br>' + line2 + '<br>' + line3);
    $('score-calc').appendChild(box);
  }

  /* ---------------- verification ---------------- */
  function runVerification() {
    var btn = $('verify-btn'), status = $('verify-status'), out = $('verify-table');
    btn.disabled = true;
    out.innerHTML = '';
    status.textContent = 'tokenizing all four corpora in this browser…';
    // vocab check straight from tokenizer.json
    var vocab = new Set(state.tokenizer.baseChars);
    state.tokenizer.merges.forEach(function (m) { vocab.add(m[0] + m[1]); });

    setTimeout(function () {
      var t0 = performance.now();
      var rows = [];
      var allOk = true;
      LANGS.forEach(function (l) {
        var claimed = perLang(l);
        var re = window.BPE.stats(state.tokenizer.merges, state.corpora[l]);
        var ok = re.tokens === claimed.tokens && re.words === claimed.words;
        allOk = allOk && ok;
        rows.push({ l: l, claimed: claimed, re: re, ok: ok });
      });
      var ms = Math.round(performance.now() - t0);

      var wrap = el('div', 'table-scroll');
      var t = el('table', 'data-table');
      t.appendChild(el('thead', null,
        '<tr><th>Language</th><th>Claimed tokens</th><th>Recomputed</th>' +
        '<th>Claimed X</th><th>Recomputed X</th><th>Match</th></tr>'));
      var tb = el('tbody');
      rows.forEach(function (r) {
        var tr = el('tr');
        tr.appendChild(el('td', 'lang', perLang(r.l).language));
        tr.appendChild(el('td', null, fmt(r.claimed.tokens)));
        tr.appendChild(el('td', null, fmt(r.re.tokens)));
        tr.appendChild(el('td', null, f6(r.claimed.fertility)));
        tr.appendChild(el('td', null, f6(r.re.fertility)));
        tr.appendChild(el('td', r.ok ? 'ok' : 'miss', r.ok ? '✓' : '✗'));
        tb.appendChild(tr);
      });
      t.appendChild(tb);
      wrap.appendChild(t);
      out.appendChild(wrap);

      var fvals = rows.map(function (r) { return r.re.fertility; });
      var spread = Math.max.apply(null, fvals) - Math.min.apply(null, fvals);
      var vOk = vocab.size <= 10000;
      var verdict = el('div', 'verdict ' + (allOk && vOk ? 'pass' : 'fail'));
      verdict.innerHTML = (allOk && vOk)
        ? 'VERIFIED in your browser (' + ms + ' ms) — vocab ' + fmt(vocab.size) + ' ≤ 10,000 · ' +
          'recomputed spread ' + spread.toFixed(6) + ' · score ≈ <b>' +
          Math.round(1000 / spread).toLocaleString('en-IN') + '</b>'
        : 'MISMATCH — recomputed values differ from the claims (segmenter difference?). The recomputed column is the truth on this device.';
      out.appendChild(verdict);
      status.textContent = 'done in ' + ms + ' ms · ' + fmt(vocab.size) + ' vocab tokens checked';
      btn.disabled = false;
    }, 30);
  }
  $('verify-btn').addEventListener('click', runVerification);

  /* ---------------- playground ---------------- */
  function initPlayground() {
    var input = $('play-input');
    input.value = 'India is a union of 28 states. भारत एक संघ है। భారతదేశం ఒక సమాఖ్య. ভারত একটি সংঘ।';
    var render = function () {
      var text = input.value;
      var box = $('play-tokens');
      var metrics = $('play-metrics');
      box.innerHTML = ''; metrics.innerHTML = '';
      if (!text.trim()) return;
      var toks = window.BPE.applyStream(state.tokenizer.merges, text);
      var words = window.BPE.countWords(text);
      var mk = function (label, value, cls) {
        var m = el('div', 'metric');
        m.appendChild(el('span', 'label', label));
        m.appendChild(el('span', 'value' + (cls ? ' ' + cls : ''), value));
        return m;
      };
      metrics.appendChild(mk('words', fmt(words)));
      metrics.appendChild(mk('tokens', fmt(toks.length)));
      metrics.appendChild(mk('tokens / word', (toks.length / words).toFixed(4), 'accent'));
      var pal = [28, 145, 210, 45, 90, 260, 320, 0, 180, 60];
      toks.forEach(function (tk) {
        var h = 0;
        for (var i = 0; i < tk.length; i++) h = (h * 31 + tk.charCodeAt(i)) >>> 0;
        var hue = pal[h % pal.length];
        var span = el('span', 'tok');
        span.style.borderColor = 'hsl(' + hue + ', 45%, 32%)';
        span.style.color = 'hsl(' + hue + ', 65%, 72%)';
        span.innerHTML = tk.replace(/▁/g, '<span class="wm">▁</span>');
        box.appendChild(span);
      });
    };
    input.addEventListener('input', render);
    render();
  }

  /* ---------------- constraint section ---------------- */
  function renderConstraint() {
    var a = state.analysis, s = state.stats;
    // facts (numbers computed live where cheap)
    var tokens0 = 0, targetTokens = 0, totalWords = 0;
    LANGS.forEach(function (l) {
      tokens0 += window.BPE.textToSymbols(state.corpora[l]).length;
      totalWords += perLang(l).words;
    });
    targetTokens = Math.round(1.2 * totalWords);
    var teUniq = new Set(state.corpora.te.split(/\s+/).filter(Boolean)).size;
    var facts = $('constraint-facts');
    [
      ['hot', 'Reaching X ≤ 1.2 for all four needs ≈ <b>' + fmt(a.vocabNeededFor12.vocabNeeded) +
        ' tokens</b> with this (grapheme) BPE, or ≈ <b>' + fmt(a.wordInternal.vocabNeededFor12) +
        '</b> with classic word-internal BPE — about 1.7× the 10,000 cap. Read it off the chart.'],
      [null, '<b>Why:</b> these are page-sized corpora. Telugu has <b>' + fmt(perLang('te').words) +
        ' words</b> and <b>' + fmt(teUniq) + '</b> of them are unique — once every repeated pattern is merged, ' +
        'each further token saved costs one whole vocab slot (a frequency-1 merge).'],
      [null, '<b>The mass balance:</b> the four corpora are <span class="mono">' + fmt(tokens0) +
        '</span> base symbols and must compress to ≤ <span class="mono">' + fmt(targetTokens) +
        '</span> tokens for fertility 1.2 — the top ~' + fmt(s.mergeRules) +
        ' pair frequencies don\'t add up to the required <span class="mono">' + fmt(tokens0 - targetTokens) +
        '</span> removals. Zipf\'s tail is the wall.'],
      ['cool', 'On a <b>large corpus</b> (millions of words) frequency-2+ merges stay productive far longer and ' +
        'X ≤ 1.2 at 10k vocab is routine — the collision here is with corpus <em>size</em>, not with BPE.'],
      ['hot', 'At 10,000 tokens the minimum achievable worst-language fertility is <b>' +
        a.minCommonFertilityAt10k.toFixed(4) + '</b> (water-filling across the four training curves). ' +
        'Equalizing all four languages there is simultaneously the best possible X<sub>max</sub> <em>and</em> ' +
        'the maximum possible score.'],
    ].forEach(function (f) {
      facts.appendChild(el('div', 'fact' + (f[0] ? ' ' + f[0] : ''), f[1]));
    });
    $('constraint-note').innerHTML =
      '<b>If a strict X ≤ 1.2 is required</b>, the honest routes are: a ~17k vocab, bigger corpora ' +
      '(more pages per language), or leaving BPE for a whole-word dictionary tokenizer (longest match) — ' +
      'which trivially hits fertility ≈ 1.0 here but is no longer BPE and would make the score formula ' +
      'degenerate (spread → 0). We kept the tokenizer genuinely BPE, the corpora exactly as specified, ' +
      'and the trade-off documented.';

    drawChart();
  }

  function drawChart() {
    var a = state.analysis, s = state.stats;
    var cv = $('constraint-chart');
    var box = cv.parentNode;
    var W = box.clientWidth || 480, H = 300;
    var dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = '100%'; cv.style.height = 'auto';
    var ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#12100d';
    ctx.fillRect(0, 0, W, H);

    var pts = a.grid.filter(function (g) { return g.vocabNeeded !== null; });
    var xMin = 1.0, xMax = 2.2;
    var yMax = 20000;
    var L = 52, R = 14, T = 16, B = 34;
    var iw = W - L - R, ih = H - T - B;
    var X = function (t) { return L + (t - xMin) / (xMax - xMin) * iw; };
    var Y = function (v) { return T + (1 - Math.min(v, yMax) / yMax) * ih; };

    ctx.font = '10px ui-monospace, Menlo, monospace';
    // y gridlines
    for (var v = 0; v <= yMax; v += 5000) {
      ctx.strokeStyle = '#211c17'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - R, Y(v)); ctx.stroke();
      ctx.fillStyle = '#6f6355';
      ctx.fillText((v / 1000) + 'k', 14, Y(v) + 3);
    }
    // x ticks
    for (var t = 1.0; t <= 2.21; t += 0.2) {
      var tx = X(t);
      ctx.fillStyle = '#6f6355';
      ctx.fillText(t.toFixed(1), tx - 8, H - 12);
    }
    ctx.fillText('fertility target (all languages)', W - R - 170, H - 12);

    // cap line at 10k
    ctx.strokeStyle = '#3ecf8e'; ctx.setLineDash([5, 4]); ctx.globalAlpha = 0.8;
    ctx.beginPath(); ctx.moveTo(L, Y(10000)); ctx.lineTo(W - R, Y(10000)); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
    ctx.fillStyle = '#3ecf8e';
    ctx.fillText('vocab cap 10,000', L + 6, Y(10000) - 6);

    // curve
    ctx.strokeStyle = '#ff9d45'; ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach(function (g, i) {
      var x = X(g.target), y = Y(g.vocabNeeded);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // markers
    function dot(x, y, color, label, above) {
      ctx.beginPath(); ctx.arc(x, y, 4, 0, 6.2832);
      ctx.fillStyle = color; ctx.fill();
      ctx.fillText(label, Math.min(x + 8, W - 150), above ? y - 8 : y + 14);
    }
    var here = a.minCommonFertilityAt10k;
    dot(X(1.2), Y(a.vocabNeededFor12.vocabNeeded), '#ff6b6b',
      '1.2 needs ' + fmt(a.vocabNeededFor12.vocabNeeded), false);
    dot(X(here), Y(10000), '#3ecf8e',
      'this tokenizer: X = ' + here.toFixed(4) + ' @ ' + fmt(s.vocabSize), true);
  }

  /* ---------------- downloads & repro ---------------- */
  function renderDownloads() {
    var s = state.stats;
    var mkItem = function (href, name, desc) {
      var a = el('a', 'dl-item');
      a.href = href; a.setAttribute('download', '');
      a.appendChild(el('span', 'name', name));
      a.appendChild(el('span', 'desc', desc));
      return a;
    };
    var dt = $('dl-tokenizer');
    dt.appendChild(mkItem('download/tokens.txt', 'tokens.txt', 'all ' + fmt(s.vocabSize) + ' vocab tokens, one per line'));
    dt.appendChild(mkItem('download/merges.txt', 'merges.txt', fmt(s.mergeRules) + ' merge rules in rank order'));
    dt.appendChild(mkItem('tokenizer.json', 'tokenizer.json', 'base symbols + merges + pipeline spec'));
    dt.appendChild(mkItem('bpe.js', 'bpe.js', 'trainer + tokenizer, the code this page runs'));

    var dc = $('dl-corpora');
    state.manifest.forEach(function (m) {
      dc.appendChild(mkItem('corpus/' + m.lang + '.txt', m.lang + '.txt',
        m.language + ' · rev ' + m.revid + ' · ' + fmt(m.words) + ' words'));
    });
    dc.appendChild(mkItem('corpus/manifest.json', 'manifest.json', 'revision ids · timestamps · sha-256'));

    $('repro').innerHTML =
      '<b># exact numbers, straight from the shipped tokenizer</b>\n' +
      'git clone https://github.com/dutta-gambit/era-v5-s2-bpe-tokenizer\n' +
      'cd era-v5-s2-bpe-tokenizer        <b># any node ≥ 14 — no ICU dependence</b>\n' +
      'node src/tokenize-cli.js corpus/en.txt corpus/hi.txt corpus/te.txt corpus/bn.txt\n\n' +
      '<b># or retrain everything from the pinned snapshots</b>\n' +
      'node src/prepare-corpus.js && node src/train.js';
  }
})();
