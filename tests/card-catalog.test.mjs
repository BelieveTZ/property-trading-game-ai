import assert from "node:assert/strict";
import test from "node:test";

import { CARD_DECKS } from "../app/card-catalog.mjs";

const compact = (card) => ({ ...card.effect, ...(card.copies ? { copies: card.copies } : {}) });

test("the physical card catalog binds every card to its official effect", () => {
  assert.equal(CARD_DECKS.chance.reduce((total, card) => total + (card.copies ?? 1), 0), 16);
  assert.equal(CARD_DECKS.community.reduce((total, card) => total + (card.copies ?? 1), 0), 16);
  assert.deepEqual(
    Object.fromEntries([...CARD_DECKS.chance, ...CARD_DECKS.community].map((card) => [card.id, compact(card)])),
    {
      "chance-boardwalk": { kind: "move", destination: 39, collectGo: false },
      "chance-go": { kind: "move", destination: 0, collectGo: true },
      "chance-illinois": { kind: "move", destination: 24, collectGo: true },
      "chance-st-charles": { kind: "move", destination: 11, collectGo: true },
      "chance-nearest-station": { kind: "nearest", target: "station", copies: 2 },
      "chance-nearest-utility": { kind: "nearest", target: "utility" },
      "chance-dividend": { kind: "cash", amount: 50 },
      "chance-jail-free": { kind: "jailFree" },
      "chance-back-three": { kind: "back", spaces: 3 },
      "chance-jail": { kind: "jail" },
      "chance-repairs": { kind: "repairs", perHouse: 25, perHotel: 100 },
      "chance-speeding": { kind: "cash", amount: -15 },
      "chance-reading": { kind: "move", destination: 5, collectGo: true },
      "chance-chairman": { kind: "payEach", amount: 50 },
      "chance-loan": { kind: "cash", amount: 150 },
      "community-go": { kind: "move", destination: 0, collectGo: true },
      "community-bank-error": { kind: "cash", amount: 200 },
      "community-doctor": { kind: "cash", amount: -50 },
      "community-stock": { kind: "cash", amount: 50 },
      "community-jail-free": { kind: "jailFree" },
      "community-jail": { kind: "jail" },
      "community-holiday": { kind: "cash", amount: 100 },
      "community-tax-refund": { kind: "cash", amount: 20 },
      "community-birthday": { kind: "collectEach", amount: 10 },
      "community-insurance": { kind: "cash", amount: 100 },
      "community-hospital": { kind: "cash", amount: -100 },
      "community-school": { kind: "cash", amount: -50 },
      "community-consultancy": { kind: "cash", amount: 25 },
      "community-repairs": { kind: "repairs", perHouse: 40, perHotel: 115 },
      "community-beauty": { kind: "cash", amount: 10 },
      "community-inherit": { kind: "cash", amount: 100 },
    },
  );
});
