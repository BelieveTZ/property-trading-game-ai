import assert from "node:assert/strict";
import test from "node:test";
import {
  OFFICIAL_AI_NAMES,
  pickUnusedAiName,
  shuffledAiNames,
} from "../app/player-catalog.mjs";

test("uses the original fictional AI name pool without repetition", () => {
  assert.equal(OFFICIAL_AI_NAMES.length, 21);
  assert.equal(new Set(OFFICIAL_AI_NAMES).size, OFFICIAL_AI_NAMES.length);
  assert.deepEqual(
    shuffledAiNames(() => 0).toSorted(),
    [...OFFICIAL_AI_NAMES].toSorted(),
  );
  assert.notEqual(pickUnusedAiName([{ name: OFFICIAL_AI_NAMES[0] }]), OFFICIAL_AI_NAMES[0]);
});
