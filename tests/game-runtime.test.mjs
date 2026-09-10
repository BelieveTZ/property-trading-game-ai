import assert from "node:assert/strict";
import test from "node:test";

import { CARD_DECKS } from "../app/card-catalog.mjs";
import {
  createGameRuntime,
  drawRuntimeCard,
  rollRuntimeDice,
} from "../app/game-runtime.mjs";

test("the runtime reproduces dice and shuffled card draws from one seed", () => {
  let left = createGameRuntime(20260910, CARD_DECKS);
  let right = createGameRuntime(20260910, CARD_DECKS);

  for (let turn = 0; turn < 20; turn += 1) {
    const leftRoll = rollRuntimeDice(left);
    const rightRoll = rollRuntimeDice(right);
    assert.deepEqual(leftRoll.dice, rightRoll.dice);
    assert.ok(leftRoll.dice.every((die) => die >= 1 && die <= 6));
    left = leftRoll.runtime;
    right = rightRoll.runtime;
  }

  const leftDraw = drawRuntimeCard(left, "chance", CARD_DECKS);
  const rightDraw = drawRuntimeCard(right, "chance", CARD_DECKS);
  assert.equal(leftDraw.card.id, rightDraw.card.id);
  assert.deepEqual(leftDraw.runtime, rightDraw.runtime);
  assert.doesNotThrow(() => JSON.stringify(leftDraw.runtime));
});

test("a deck deals every physical copy before reshuffling", () => {
  let runtime = createGameRuntime(73, CARD_DECKS);
  const drawn = [];
  for (let index = 0; index < 16; index += 1) {
    const result = drawRuntimeCard(runtime, "chance", CARD_DECKS);
    drawn.push(result.card.id);
    runtime = result.runtime;
  }

  const expected = CARD_DECKS.chance.flatMap((card) =>
    Array.from({ length: card.copies ?? 1 }, () => card.id),
  );
  assert.deepEqual(drawn.toSorted(), expected.toSorted());

  const next = drawRuntimeCard(runtime, "chance", CARD_DECKS);
  assert.ok(CARD_DECKS.chance.some((card) => card.id === next.card.id));
  assert.equal(next.runtime.decks.chance.cursor, 1);
});
