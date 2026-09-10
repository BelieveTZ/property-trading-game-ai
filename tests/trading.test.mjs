import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { applyTrade } from "../app/game-rules.mjs";
import {
  FastRandom,
  createEmptyGenome,
  tryTrade,
} from "../training/league-trainer.mjs";
import { indexedBoardRules } from "../shared/board-rules.mjs";

const trainer = await readFile(
  new URL("../training/league-trainer.mjs", import.meta.url),
  "utf8",
);
const advisor = await readFile(
  new URL("../app/ai-advisor.mjs", import.meta.url),
  "utf8",
);

test("records bilateral cash, deed and jail-card trades with classic safeguards", () => {
  assert.match(advisor, /export function recommendTradeProposal/);
  assert.match(advisor, /const considerCashForDeed =/);
  assert.match(advisor, /const evaluation = evaluateTrade/);
  assert.match(advisor, /!evaluation\.accepted/);
  assert.match(advisor, /!rejectedKeys\.includes\(candidate\.key\)/);
  const result = applyTrade(
    {
      players: [
        { id: 0, cash: 500, jailFreeCards: 1, bankrupt: false },
        { id: 1, cash: 500, jailFreeCards: 0, bankrupt: false },
      ],
      properties: [
        { tileId: 6, ownerId: 0, houses: 0, mortgaged: false },
        { tileId: 12, ownerId: 1, houses: 0, mortgaged: true },
      ],
      activePlayerId: 0,
      tradeLockPlayerId: null,
      tradeLockedTileIds: [],
      debt: null,
      paymentQueue: [],
    },
    {
      fromId: 0,
      toId: 1,
      fromCash: 20,
      toCash: 100,
      fromCards: 1,
      toCards: 0,
      fromPropertyIds: [6],
      toPropertyIds: [12],
    },
    { mortgageValues: { 12: 75 } },
  );
  assert.equal(result.ok, true);
  assert.equal(result.game.players[0].cash, 572);
  assert.equal(result.game.players[1].cash, 420);
  assert.equal(result.game.players[1].jailFreeCards, 1);
  assert.deepEqual(
    result.game.properties.map((property) => property.ownerId),
    [1, 0],
  );
});

test("locks a traded deed against same-turn round trips", () => {
  const game = {
    players: [
      { id: 0, cash: 500, jailFreeCards: 0, bankrupt: false },
      { id: 1, cash: 500, jailFreeCards: 0, bankrupt: false },
    ],
    properties: [
      { tileId: 6, ownerId: 0, houses: 0, mortgaged: false },
    ],
    activePlayerId: 0,
    tradeLockPlayerId: null,
    tradeLockedTileIds: [],
    debt: null,
    paymentQueue: [],
  };
  const first = applyTrade(game, {
    fromId: 0,
    toId: 1,
    fromCash: 0,
    toCash: 10,
    fromCards: 0,
    toCards: 0,
    fromPropertyIds: [6],
    toPropertyIds: [],
  });
  const reverse = applyTrade(first.game, {
    fromId: 1,
    toId: 0,
    fromCash: 0,
    toCash: 10,
    fromCards: 0,
    toCards: 0,
    fromPropertyIds: [6],
    toPropertyIds: [],
  });
  assert.equal(reverse.ok, false);
  assert.equal(reverse.reason, "same-turn-round-trip");

  const brain = createEmptyGenome();
  brain.outputBias[4] = -1;
  brain.outputBias[5] = 1;
  const state = {
    cash: new Int32Array([1500, 1500]),
    alive: new Uint8Array([1, 1]),
    owners: new Int8Array(40).fill(-1),
    houses: new Uint8Array(40),
    mortgaged: new Uint8Array(40),
    groupCounts: new Uint8Array(20),
    features: new Float64Array(9),
  };
  const group = indexedBoardRules().groupIndexes[9];
  state.owners[9] = 1;
  state.groupCounts[group] = 2;
  state.groupCounts[10 + group] = 1;
  const lockedTiles = new Uint8Array(40);
  const tradedTile = tryTrade(
    state,
    0,
    [brain, brain],
    new FastRandom(20260724),
    1,
    100,
    0,
    lockedTiles,
  );
  assert.equal(tradedTile, 9);
  lockedTiles[tradedTile] = 1;
  assert.equal(
    tryTrade(
      state,
      1,
      [brain, brain],
      new FastRandom(20260724),
      1,
      100,
      0,
      lockedTiles,
    ),
    -1,
  );
});

test("values buying and selling from symmetric portfolio deltas", () => {
  for (const implementation of [advisor, trainer]) {
    assert.match(implementation, /const beforeBonus = groupBonus/);
    assert.match(implementation, /const afterBonus = groupBonus/);
    assert.match(
      implementation,
      /marginalBonus = [\s\S]*?afterBonus - beforeBonus[\s\S]*?beforeBonus - afterBonus/,
    );
  }
  assert.match(advisor, /\(tile\.rent \?\? 0\) \* 4/);
  assert.match(trainer, /BASE_RENTS\[tile\] \* 4/);
  assert.match(
    trainer,
    /symmetric marginal portfolio trade valuation/,
  );
});

test("trains and deploys a sixth neural action for bilateral trading", async () => {
  assert.match(trainer, /const INPUTS = LEAGUE_INPUT_COUNT/);
  assert.match(trainer, /const OUTPUTS = LEAGUE_ACTION_NAMES\.length/);
  assert.match(trainer, /function tryTrade\(/);
  assert.match(trainer, /function tryTradesAtWindow\(/);
  assert.match(trainer, /const maximumTrades = brains\.length/);
  assert.match(trainer, /tradedThisTurn/);
  assert.match(trainer, /tradeWindow/);
  assert.match(trainer, /tryTradesAtWindow\([\s\S]*?-1,[\s\S]*?tradedThisTurn/);
  assert.match(trainer, /tryTradesAtWindow\([\s\S]*?0\.5,[\s\S]*?tradedThisTurn/);
  assert.match(trainer, /tryTradesAtWindow\([\s\S]*?1,[\s\S]*?tradedThisTurn/);
  assert.match(trainer, /evaluate\(buyerBrain, state\.features, LEAGUE_OUTPUT\.trade\)/);
  assert.match(trainer, /evaluate\(sellerBrain, state\.features, LEAGUE_OUTPUT\.trade\)/);
  assert.match(trainer, /LEAGUE_ACTION_NAMES/);

  for (const filename of [
    "pretrained-model-3p.json",
    "pretrained-model.json",
    "pretrained-model-5p.json",
  ]) {
    const model = JSON.parse(
      await readFile(new URL(`../app/${filename}`, import.meta.url), "utf8"),
    );
    assert.equal(model.outputCount, 6);
    assert.equal(model.actionNames.at(-1), "trade");
    assert.equal(model.achievedStandard.passed, true);
  }
});
