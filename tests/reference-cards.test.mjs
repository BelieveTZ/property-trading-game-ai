import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL("../app/GameApp.tsx", import.meta.url),
  "utf8",
);
const css = await readFile(
  new URL("../app/globals.css", import.meta.url),
  "utf8",
);

test("opens scrollable Chance, Community Chest, and Income Tax cards", () => {
  assert.match(source, /const isReferenceStyle =[\s\S]*?tile\.type === "chance"/);
  assert.match(source, /tile\.type === "community"/);
  assert.match(source, /tile\.id === 4/);
  assert.match(source, /function renderReferenceModal\(\)/);
  assert.match(source, /CARD_DECKS\[deck\]\.flatMap/);
  assert.match(source, /className="reference-card-list"/);
  assert.match(source, /className="tax-reference-card"/);
  assert.match(source, /\{renderReferenceModal\(\)\}/);
  assert.doesNotMatch(source, /String\(index \+ 1\)\.padStart/);
  assert.doesNotMatch(source, /"税务卡"/);
  assert.doesNotMatch(source, /向下滚动查看/);
  assert.match(
    css,
    /\.reference-card-list\s*\{[\s\S]*?overflow-y:\s*auto;/,
  );
  assert.match(
    css,
    /\.tax-reference-card\s*\{[\s\S]*?margin:\s*0;[\s\S]*?background:\s*#fff;/,
  );
});

test("keeps card faces clean while retaining the board grid", () => {
  assert.match(
    css,
    /\.board\s*\{[\s\S]*?border:\s*3px solid var\(--ink\);/,
  );
  assert.match(
    css,
    /\.tile\s*\{[\s\S]*?border:\s*0\.5px solid rgba\(32, 36, 31, 0\.55\);/,
  );
  assert.match(css, /\.color-band\s*\{[\s\S]*?border-bottom:\s*0;/);
  assert.match(css, /\.deed-card\s*\{[\s\S]*?border:\s*0;/);
  assert.match(css, /\.deed-card > header\s*\{[\s\S]*?border-bottom:\s*0;/);
  assert.match(css, /\.reference-modal\s*\{[\s\S]*?border:\s*0;/);
  assert.match(css, /\.reference-card\s*\{[\s\S]*?border:\s*0;/);
});
