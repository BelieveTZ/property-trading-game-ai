import assert from "node:assert/strict";
import test from "node:test";
import { resolveRollRules } from "../app/game-rules.mjs";

test("does not offer an extra turn after rolling doubles out of jail", () => {
  const result = resolveRollRules({
    player: { position: 10, inJail: true, jailTurns: 1 },
    dieOne: 4,
    dieTwo: 4,
    doublesStreak: 0,
  });
  assert.equal(result.player.inJail, false);
  assert.equal(result.shouldMove, true);
  assert.equal(result.extraTurnEligible, false);
});

test("ends the turn after triple doubles sends the player to jail", () => {
  const result = resolveRollRules({
    player: { position: 7, inJail: false, jailTurns: 0 },
    dieOne: 2,
    dieTwo: 2,
    doublesStreak: 2,
  });
  assert.equal(result.player.position, 10);
  assert.equal(result.player.inJail, true);
  assert.equal(result.shouldMove, false);
  assert.equal(result.extraTurnEligible, false);
});
