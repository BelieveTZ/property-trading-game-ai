import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("points player-colored ownership flags toward the board center", async () => {
  const source = await readFile(
    new URL("../app/GameApp.tsx", import.meta.url),
    "utf8",
  );
  const css = await readFile(
    new URL("../app/globals.css", import.meta.url),
    "utf8",
  );

  assert.match(source, /style=\{\{ "--owner-color": owner\.color \} as CSSProperties\}/);
  assert.match(source, /\$\{owner \? "owned" : ""\}/);
  assert.match(css, /\.tile\.owned \{[\s\S]*?overflow: visible;[\s\S]*?z-index: 4;/);
  assert.match(css, /\.owner-mark::before \{[\s\S]*?background: var\(--owner-color\);/);
  assert.match(css, /\.owner-mark::after \{[\s\S]*?background: var\(--owner-color\);[\s\S]*?clip-path: polygon\(/);
  assert.match(css, /\.tile\.edge-left \.owner-mark \{[\s\S]*?left: calc\(100% - 5px\);[\s\S]*?rotate\(90deg\)/);
  assert.match(css, /\.tile\.edge-right \.owner-mark \{[\s\S]*?right: calc\(100% - 5px\);[\s\S]*?rotate\(-90deg\)/);
  assert.match(css, /\.tile\.edge-top \.owner-mark \{[\s\S]*?top: calc\(100% - 5px\);[\s\S]*?rotate\(180deg\)/);
  assert.match(css, /\.tile\.edge-bottom \.owner-mark \{[\s\S]*?bottom: calc\(100% - 5px\);/);
});
