import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  OFFICIAL_AI_NAMES,
  pickUnusedAiName,
  shuffledAiNames,
} from "../app/player-catalog.mjs";

test("uses the official Simplified Chinese AI name pool without repetition", async () => {
  const source = await readFile(
    new URL("../app/GameApp.tsx", import.meta.url),
    "utf8",
  );

  assert.equal(OFFICIAL_AI_NAMES.length, 21);
  assert.equal(new Set(OFFICIAL_AI_NAMES).size, OFFICIAL_AI_NAMES.length);
  assert.deepEqual(
    shuffledAiNames(() => 0).toSorted(),
    [...OFFICIAL_AI_NAMES].toSorted(),
  );
  assert.notEqual(pickUnusedAiName([{ name: OFFICIAL_AI_NAMES[0] }]), OFFICIAL_AI_NAMES[0]);
  assert.match(source, /createGame\(game\.players\.length, true\)/);
  assert.match(source, /createPlayer\(id, pickUnusedAiName\(game\.players\)\)/);
  assert.match(source, /className="player-name-preset"/);
  assert.match(source, /value=""/);
  assert.match(source, /OFFICIAL_AI_NAMES\.map\(\(name\) =>/);
  assert.match(source, /placeholder="输入名称"/);
  assert.match(source, /updatePlayer\(player\.id, \{ name: e\.target\.value \}\)/);
  assert.doesNotMatch(source, /<datalist id="official-player-names">/);
});

test("shows a green adaptive-stop line for every player-count model", async () => {
  const source = await readFile(
    new URL("../app/GameApp.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /<p className="standard-pass">/);
  assert.doesNotMatch(source, /已达标停止/);
  assert.match(source, /✓ 自适应停止 · 95% 胜率下界/);
  assert.match(source, /✓ 自适应停止 · 独立验证胜率/);
  assert.doesNotMatch(source, /\{" ≥ "\}/);
  assert.doesNotMatch(
    source,
    /stoppingStandard\.targetWinRateLowerBound \* 100/,
  );
  assert.match(source, /leagueModel\.averageRankVsHeuristic\.toFixed\(2\)/);
});
