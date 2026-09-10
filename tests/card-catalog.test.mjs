import assert from "node:assert/strict";
import test from "node:test";

import { CARD_DECKS } from "../app/card-catalog.mjs";

const compact = (card) => ({
  ...card.effect,
  ...(card.copies ? { copies: card.copies } : {}),
});

test("the original card catalog preserves every fixed-rule effect", () => {
  assert.equal(CARD_DECKS.chance.reduce((total, card) => total + (card.copies ?? 1), 0), 16);
  assert.equal(CARD_DECKS.community.reduce((total, card) => total + (card.copies ?? 1), 0), 16);
  assert.deepEqual(CARD_DECKS.chance.map(compact), [
    { kind: "move", destination: 39, collectGo: false },
    { kind: "move", destination: 0, collectGo: true },
    { kind: "move", destination: 24, collectGo: true },
    { kind: "move", destination: 11, collectGo: true },
    { kind: "nearest", target: "station", copies: 2 },
    { kind: "nearest", target: "utility" },
    { kind: "cash", amount: 50 },
    { kind: "jailFree" },
    { kind: "back", spaces: 3 },
    { kind: "jail" },
    { kind: "repairs", perHouse: 25, perHotel: 100 },
    { kind: "cash", amount: -15 },
    { kind: "move", destination: 5, collectGo: true },
    { kind: "payEach", amount: 50 },
    { kind: "cash", amount: 150 },
  ]);
  assert.deepEqual(CARD_DECKS.community.map(compact), [
    { kind: "move", destination: 0, collectGo: true },
    { kind: "cash", amount: 200 },
    { kind: "cash", amount: -50 },
    { kind: "cash", amount: 50 },
    { kind: "jailFree" },
    { kind: "jail" },
    { kind: "cash", amount: 100 },
    { kind: "cash", amount: 20 },
    { kind: "collectEach", amount: 10 },
    { kind: "cash", amount: 100 },
    { kind: "cash", amount: -100 },
    { kind: "cash", amount: -50 },
    { kind: "cash", amount: 25 },
    { kind: "repairs", perHouse: 40, perHotel: 115 },
    { kind: "cash", amount: 10 },
    { kind: "cash", amount: 100 },
  ]);

  const ids = [...CARD_DECKS.chance, ...CARD_DECKS.community].map((card) => card.id);
  assert.equal(new Set(ids).size, ids.length);
  assert.ok(CARD_DECKS.chance.every((card) => card.id.startsWith("turning-point-")));
  assert.ok(CARD_DECKS.community.every((card) => card.id.startsWith("city-fund-")));
});
