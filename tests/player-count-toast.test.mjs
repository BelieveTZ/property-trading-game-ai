import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL("../app/GameApp.tsx", import.meta.url),
  "utf8",
);

test("adds and removes players without a success toast", () => {
  assert.doesNotMatch(source, /setToast\(`已添加/);
  assert.doesNotMatch(source, /setToast\(`已移除/);
  assert.match(source, /setToast\("最多支持 5 名玩家"\)/);
  assert.match(source, /setToast\("牌局至少需要 3 名玩家"\)/);
});
