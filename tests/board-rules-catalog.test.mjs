import assert from "node:assert/strict";
import test from "node:test";

import {
  BOARD_SIZE,
  DEED_TILE_IDS,
  boardRuleFor,
  indexedBoardRules,
} from "../shared/board-rules.mjs";

test("the board-rules catalog exposes the classic deed economics", () => {
  assert.equal(BOARD_SIZE, 40);
  assert.equal(DEED_TILE_IDS.length, 28);
  assert.deepEqual(boardRuleFor(1), {
    type: "property",
    group: "brown",
    price: 60,
    baseRent: 2,
    mortgage: 30,
    buildCost: 50,
    rents: [2, 10, 30, 90, 160, 250],
  });
  assert.deepEqual(boardRuleFor(39).rents, [50, 200, 600, 1400, 1700, 2000]);
  assert.deepEqual(boardRuleFor(5).rents, [25, 50, 100, 200]);
  assert.equal(boardRuleFor(12).mortgage, 75);
});

test("the indexed training view is derived from the same catalog", () => {
  const indexed = indexedBoardRules();

  assert.equal(indexed.prices.length, BOARD_SIZE);
  assert.equal(indexed.prices[9], 120);
  assert.equal(indexed.baseRents[28], 20);
  assert.equal(indexed.buildCosts[34], 200);
  assert.equal(indexed.groupIndexes[5], 8);
  assert.equal(indexed.groupIndexes[12], 9);
  assert.deepEqual(indexed.groupSizes, [2, 3, 3, 3, 3, 3, 3, 2, 4, 2]);
});
