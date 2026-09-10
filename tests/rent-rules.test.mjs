import assert from "node:assert/strict";
import test from "node:test";

import { TILES } from "../app/board-catalog.mjs";
import { propertyRent } from "../app/rent-rules.mjs";

const deed = (tileId, houses = 0, mortgaged = false) => ({
  tileId,
  ownerId: 0,
  houses,
  mortgaged,
});

test("property rent follows monopoly, building, railroad, utility, and mortgage rules", () => {
  assert.equal(propertyRent(TILES[1], deed(1), [deed(1)]), 2);
  assert.equal(propertyRent(TILES[1], deed(1), [deed(1), deed(3)]), 4);
  assert.equal(propertyRent(TILES[1], deed(1, 1), [deed(1, 1), deed(3)]), 10);
  assert.equal(propertyRent(TILES[1], deed(1, 5), [deed(1, 5), deed(3)]), 250);
  assert.equal(propertyRent(TILES[1], deed(1, 0, true), [deed(1, 0, true), deed(3)]), 0);

  assert.equal(propertyRent(TILES[5], deed(5), [deed(5)]), 25);
  assert.equal(propertyRent(TILES[5], deed(5), [deed(5), deed(15)]), 50);
  assert.equal(propertyRent(TILES[5], deed(5), [deed(5), deed(15), deed(25), deed(35)]), 200);

  assert.equal(propertyRent(TILES[12], deed(12), [deed(12)], 7), 28);
  assert.equal(propertyRent(TILES[12], deed(12), [deed(12), deed(28)], 7), 70);
});
