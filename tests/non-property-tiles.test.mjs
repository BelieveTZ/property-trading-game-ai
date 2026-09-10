import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { TILES, UTILITY_COLOR } from "../app/board-catalog.mjs";

test("centers special spaces while keeping railroads and utilities deed-styled", async () => {
  const source = await readFile(
    new URL("../app/GameApp.tsx", import.meta.url),
    "utf8",
  );
  const css = await readFile(
    new URL("../app/globals.css", import.meta.url),
    "utf8",
  );

  assert.match(source, /tile\.type === "property" \|\|[\s\S]*?tile\.type === "station" \|\|[\s\S]*?tile\.type === "utility"/);
  assert.match(source, /isDeedStyle \? "deed-tile" : "non-property"/);
  assert.match(source, /\{isDeedStyle && \(/);
  assert.match(source, /\{!isDeedStyle && \(/);
  assert.match(source, /utility: ""/);
  assert.equal(UTILITY_COLOR, "#ffffff");
  assert.equal(TILES[12].color, UTILITY_COLOR);
  assert.equal(TILES[28].color, UTILITY_COLOR);
  assert.match(source, /utility: UTILITY_COLOR/);
  assert.match(source, /if \(tile\.type === "utility"\) return UTILITY_COLOR;/);
  assert.match(source, /className="asset-band" style=\{\{ background: tileStripColor\(tile\) \}\}/);
  assert.match(source, /community: "▣"/);
  assert.match(source, /tile\.id === 25 \? "long-name" : ""/);
  assert.match(source, /className="rail-name-line">巴尔的摩/);
  assert.match(source, /className="rail-name-line">与俄亥俄/);
  assert.match(source, /className="rail-name-line">铁路/);
  assert.doesNotMatch(source, /⌂/);
  assert.doesNotMatch(source, /\{tile\.price && <small>/);
  assert.doesNotMatch(source, /title=\{`\$\{tile\.name\}/);
  assert.match(css, /\.tile\.non-property \{\s*grid-template-rows: 100%;/);
  assert.match(css, /\.tile\.non-property \.tile-copy \.tile-icon \{[\s\S]*?opacity: 1;/);
  assert.match(css, /\.tile-copy strong \{[\s\S]*?width: 100%;[\s\S]*?text-align: center;/);
  assert.match(css, /\.tile\.long-name \.tile-copy strong \{[\s\S]*?-webkit-line-clamp: 3;/);
  assert.doesNotMatch(css, /\.tile\.long-name \.tile-copy strong \{[^}]*font-size:/);
  assert.match(css, /\.rail-name-line \{[\s\S]*?white-space: nowrap;/);
  assert.match(css, /\.tile\.type-chance \.tile-copy/);
  assert.match(css, /\.tile\.type-community \.tile-copy/);
  assert.doesNotMatch(css, /\.tile\.type-station \.tile-copy/);
  assert.doesNotMatch(css, /\.tile\.edge-(?:left|right)\.non-property/);
  assert.doesNotMatch(css, /\.tile\.type-utility \.tile-copy/);
  assert.match(css, /\.tile\.type-jail \.tile-copy/);
  assert.match(css, /\.tile\.type-parking \.tile-copy/);
  assert.match(css, /\.tile\.type-gotojail \.tile-copy/);
  assert.doesNotMatch(css, /url\(/);
});
