import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("draws directional corners and splits jail occupants by status", async () => {
  const source = await readFile(
    new URL("../app/GameApp.tsx", import.meta.url),
    "utf8",
  );
  const css = await readFile(
    new URL("../app/globals.css", import.meta.url),
    "utf8",
  );

  assert.match(source, /start: "←"/);
  assert.match(source, /gotojail: "↙"/);
  assert.match(source, /jail: ""/);
  assert.match(source, /occupants\.filter\(\(player\) => !player\.inJail\)/);
  assert.match(source, /occupants\.filter\(\(player\) => player\.inJail\)/);
  assert.match(source, /className="jail-label jail-label-visiting">探监/);
  assert.match(source, /className="jail-label jail-label-jailed">坐牢/);
  assert.match(source, /className="tokens jail-visiting-tokens"/);
  assert.match(source, /className="tokens jail-jailed-tokens"/);

  assert.match(css, /\.tile\.type-start \.tile-copy \{[\s\S]*?background: #dcebd7;/);
  assert.match(css, /\.tile\.type-gotojail \.tile-copy \{[\s\S]*?background: #edd3c9;/);
  assert.match(css, /\.tile\.type-jail \.tile-copy \{[\s\S]*?background: #e8d6bd;/);
  assert.match(css, /\.tile\.type-jail \{[\s\S]*?border: 0;/);
  assert.match(css, /\.tile\.type-jail \.tile-copy::before \{[\s\S]*?content: none;/);
  assert.match(
    css,
    /\.tile\.type-jail \.tile-copy::after \{[\s\S]*?top: 0;[\s\S]*?right: 0;[\s\S]*?width: 66%;[\s\S]*?height: 68%;[\s\S]*?border-left: 2px solid[\s\S]*?border-bottom: 2px solid/,
  );
  assert.doesNotMatch(css, /clip-path: polygon\(0 0, 100% 0, 100% 50%, 0 50%\)/);
  assert.match(css, /\.tokens\.jail-visiting-tokens \{[\s\S]*?left: 3px;[\s\S]*?bottom: 3px;/);
  assert.match(css, /\.tokens\.jail-jailed-tokens \{[\s\S]*?top: 3px;[\s\S]*?right: 3px;/);
  assert.match(css, /\.tokens i \{[\s\S]*?width: 16px;[\s\S]*?height: 16px;/);
});
