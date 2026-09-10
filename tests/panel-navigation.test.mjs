import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../app/GameApp.tsx", import.meta.url), "utf8");

test("uses concise panel tabs without a duplicate calibration introduction", () => {
  assert.match(source, />当前回合<\/button>/);
  assert.match(source, />全局校准<\/button>/);
  assert.doesNotMatch(source, /<div className="panel-intro">/);
  assert.doesNotMatch(
    source,
    /让数字与桌面保持一致|误录、交易或桌面临时规则/,
  );
});
