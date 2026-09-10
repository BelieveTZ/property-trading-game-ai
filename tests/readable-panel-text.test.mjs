import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const css = await readFile(
  new URL("../app/globals.css", import.meta.url),
  "utf8",
);
const source = await readFile(
  new URL("../app/GameApp.tsx", import.meta.url),
  "utf8",
);

test("keeps player, model, trade, and asset text readable", () => {
  assert.match(
    css,
    /\.player-copy small\s*\{[\s\S]*?font-size:\s*10px;/,
  );
  assert.match(
    css,
    /\.model-status > div:last-child > span\s*\{[\s\S]*?font-size:\s*10px;/,
  );
  assert.match(
    css,
    /\.model-status p\s*\{[\s\S]*?font-size:\s*10px;/,
  );
  assert.match(
    css,
    /\.edit-grid span\s*\{[\s\S]*?font-size:\s*10px;/,
  );
  assert.match(
    css,
    /\.trade-value-grid span\s*\{[\s\S]*?font-size:\s*10px;/,
  );
  assert.match(
    css,
    /\.trade-value-grid input\s*\{[\s\S]*?font-size:\s*11px;/,
  );
  assert.match(
    css,
    /\.asset-name strong\s*\{[\s\S]*?font-size:\s*11px;/,
  );
  assert.match(
    css,
    /\.asset-name small\s*\{[\s\S]*?font-size:\s*9px;/,
  );
});

test("uses a prohibited cursor for every disabled button", () => {
  assert.match(
    css,
    /button:disabled\s*\{[\s\S]*?cursor:\s*not-allowed;/,
  );
  assert.doesNotMatch(css, /\.primary-button:disabled\s*\{[^}]*cursor:\s*wait/);
});

test("stretches the control panel to the bottom of the workspace", () => {
  assert.match(css, /\.workspace\s*\{[\s\S]*?align-items:\s*stretch;/);
  assert.match(
    css,
    /\.control-panel\s*\{[\s\S]*?min-height:\s*100%;[\s\S]*?display:\s*flex;/,
  );
  assert.match(
    css,
    /\.panel-scroll\s*\{[\s\S]*?min-height:\s*0;[\s\S]*?flex:\s*1;/,
  );
});

test("keeps calibration scrolling inside the fixed desktop sidebar", () => {
  assert.match(
    css,
    /\.app-shell\s*\{[\s\S]*?height:\s*100vh;[\s\S]*?overflow:\s*hidden;/,
  );
  assert.match(
    css,
    /\.workspace\s*\{[\s\S]*?height:\s*100%;[\s\S]*?overflow:\s*hidden;/,
  );
  assert.match(
    css,
    /\.control-panel\s*\{[\s\S]*?height:\s*100%;[\s\S]*?overflow:\s*hidden;/,
  );
  assert.match(
    css,
    /\.panel-scroll\s*\{[\s\S]*?overflow-y:\s*auto;[\s\S]*?overscroll-behavior-y:\s*contain;/,
  );
  assert.match(
    css,
    /@media \(max-width:\s*940px\)[\s\S]*?\.app-shell\s*\{[\s\S]*?height:\s*auto;[\s\S]*?overflow:\s*visible;/,
  );
});

test("keeps control section headings free of redundant helper copy", () => {
  assert.doesNotMatch(source, /读取实体骰子结果/);
  assert.doesNotMatch(source, /录入双方实际报价/);
  assert.doesNotMatch(source, /录入桌面实际成交内容/);
  assert.doesNotMatch(source, /项资产/);
});

test("labels model generations in Chinese", () => {
  assert.match(
    source,
    /model-orb"><strong>\{leagueModel\.generations\}<\/strong><span>代<\/span>/,
  );
  assert.doesNotMatch(source, /model-orb"><span>G<\/span>/);
});
