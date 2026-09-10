import assert from "node:assert/strict";
import test from "node:test";

import { chooseAutomaticAction } from "../app/standalone-policy.mjs";
import {
  applyStandaloneAction,
  createStandaloneSession,
  legalStandaloneActions,
} from "../app/standalone-session.mjs";

test("the automatic policy only returns executable actions throughout a seeded game", () => {
  let session = createStandaloneSession({ mode: "spectate", playerCount: 4, seed: 410 });
  let completedActions = 0;
  for (; completedActions < 10_000 && !session.game.gameOver; completedActions += 1) {
    const action = chooseAutomaticAction(session);
    assert.ok(action, `policy stalled after ${completedActions} actions`);
    assert.ok(legalStandaloneActions(session).includes(action.kind));
    const result = applyStandaloneAction(session, action);
    assert.equal(result.ok, true, `${action.kind}: ${result.reason}`);
    session = result.session;
  }
  assert.ok(completedActions > 200);
  assert.equal(session.game.gameOver, true, "seeded spectator game did not reach a winner");
});
