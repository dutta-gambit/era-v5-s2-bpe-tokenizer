/* Builds corpus/{lang}.txt from data/raw_{lang}.json (Wikipedia TextExtracts snapshots).
 *
 * Cleaning policy (documented in the widget):
 *  1. Cut the article at its first back-matter section (See also / Notes / References /
 *     Bibliography / External links and their hi/te/bn equivalents) — citation boilerplate
 *     is not prose and pollutes the Indic corpora with English strings.
 *  2. Strip "== Heading ==" markers but keep the heading words (they are real text).
 *  3. Normalize all whitespace runs to single spaces (word counting is whitespace-split,
 *     so this only affects file size, not counts).
 */
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const ROOT = path.join(__dirname, '..');
const BACK_MATTER = {
  en: ['See also', 'Notes', 'References', 'Bibliography', 'External links'],
  hi: ['इन्हें भी देखें', 'सन्दर्भ', 'टिप्पणी सूची', 'बाहरी कड़ियाँ'],
  te: ['చిత్రమాలిక', 'గమనికలు', 'మూలాలు', 'ఉపయుక్త గ్రంథాలు', 'వెలుపలి లంకెలు', 'పాద పీఠిక'],
  bn: ['আরও দেখুন', 'টীকা', 'গ্রন্থপঞ্জি', 'তথ্যসূত্র', 'বহিঃসংযোগ'],
};
const LANG_NAMES = { en: 'English', hi: 'Hindi', te: 'Telugu', bn: 'Bengali' };

fs.mkdirSync(path.join(ROOT, 'corpus'), { recursive: true });
const manifest = [];

for (const lang of ['en', 'hi', 'te', 'bn']) {
  const raw = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', `raw_${lang}.json`), 'utf8'));
  const page = raw.query.pages[0];
  let text = page.extract;

  // 1. cut at first back-matter heading
  let cutAt = text.length;
  for (const h of BACK_MATTER[lang]) {
    const idx = text.indexOf(`== ${h} ==`);
    if (idx !== -1 && idx < cutAt) cutAt = idx;
  }
  text = text.slice(0, cutAt);

  // 2. strip heading markers, keep the words
  text = text.replace(/^=+ (.+?) =+$/gm, '$1');

  // 3. normalize whitespace
  text = text.replace(/\s+/g, ' ').trim();

  const words = text.split(/\s+/).filter(Boolean);
  fs.writeFileSync(path.join(ROOT, 'corpus', `${lang}.txt`), text);
  manifest.push({
    lang,
    language: LANG_NAMES[lang],
    title: page.title,
    source: `https://${lang}.wikipedia.org/wiki/${encodeURIComponent(page.title)}`,
    revid: page.revisions[0].revid,
    revTimestamp: page.revisions[0].timestamp,
    chars: text.length,
    words: words.length,
    sha256: crypto.createHash('sha256').update(text).digest('hex'),
  });
  console.log(`${lang}: ${words.length} words, ${text.length} chars (rev ${page.revisions[0].revid})`);
}

fs.writeFileSync(path.join(ROOT, 'corpus', 'manifest.json'), JSON.stringify(manifest, null, 2));
console.log('corpus/manifest.json written');
