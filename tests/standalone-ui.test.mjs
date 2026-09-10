import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("the root route exposes the standalone play and spectator flows", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const source = await readFile(
    new URL("../app/StandaloneGameApp.tsx", import.meta.url),
    "utf8",
  );

  assert.match(page, /import StandaloneGameApp/);
  assert.match(page, /<StandaloneGameApp \/>/);
  assert.match(source, />开始游玩</);
  assert.match(source, />观看对局</);
  assert.match(source, />掷骰子</);
  assert.match(source, /实时建议/);
  assert.match(source, /applyStandaloneAction/);
  assert.match(source, />资产管理</);
  assert.match(source, />发起交易</);
  assert.match(source, /接受报价/);
  assert.match(source, /拒绝报价/);
  assert.match(source, /提出反报价/);
  assert.match(source, /当前出价/);
  assert.match(source, /导入回放/);
  assert.match(source, /暂停观战/);
  assert.match(source, /session\.game\.gameOver/);
  assert.match(source, /牌局结束/);
  assert.match(source, /winnerId/);
  assert.match(source, /start_property_trading_game/);
  assert.match(source, /document\.modelContext/);
  assert.doesNotMatch(source, /录入骰子|全局校准|桌面上实际抽到/);
});

test("the standalone game is installable and keeps its application shell offline", async () => {
  const manifest = await readFile(new URL("../app/manifest.ts", import.meta.url), "utf8");
  const worker = await readFile(new URL("../public/sw.js", import.meta.url), "utf8");
  assert.match(manifest, /地产交易游戏AI/);
  assert.match(manifest, /display: "standalone"/);
  assert.match(worker, /property-trading-game-ai-shell/);
  assert.match(worker, /caches\.open/);
  assert.match(worker, /matchAll/);
  assert.match(worker, /request\.mode === "navigate"/);
  assert.match(worker, /Promise\.all\(assets\.map/);
});
