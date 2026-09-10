import assert from "node:assert/strict";
import test from "node:test";
import {
  evaluateModel,
  evaluateTrade,
  recommendAuction,
  recommendBuild,
  recommendJail,
  recommendPayment,
  recommendPurchase,
  recommendTradeProposal,
  tradeAssetValue,
  tradeProposalKey,
} from "../app/ai-advisor.mjs";
import {
  LEAGUE_OUTPUT,
  encodeLeagueFeatures,
} from "../shared/league-model.mjs";

const neutralModel = {
  hiddenCount: 1,
  inputCount: 9,
  outputCount: 6,
  hiddenBias: [0],
  inputHidden: Array(9).fill(0),
  hiddenOutput: Array(36).fill(0),
  outputBias: Array(6).fill(0),
  actionNames: ["buy", "auctionBid", "build", "leaveJail", "cashReserve", "trade"],
};

test("league features and output heads have a named checkpoint-compatible schema", () => {
  assert.deepEqual(LEAGUE_OUTPUT, {
    buy: 0,
    auctionBid: 1,
    build: 2,
    leaveJail: 3,
    cashReserve: 4,
    trade: 5,
  });
  assert.deepEqual(
    encodeLeagueFeatures({
      cash: 1500,
      postCash: 750,
      price: 200,
      outcomeValue: -100,
      groupProgress: 0.5,
      completesSet: true,
      turnProgress: 0.25,
      advantage: 1000,
      tradeWindow: -0.75,
    }),
    [1, 0.5, 0.5, -0.5, 0.5, 1, 0.25, 0.5, -0.75],
  );
});

test("model evaluation treats missing trailing features as zero", () => {
  const model = {
    ...neutralModel,
    inputHidden: [1, 2, 3, ...Array(6).fill(0)],
    hiddenOutput: [1, ...Array(35).fill(0)],
  };
  assert.equal(evaluateModel(model, [0.5], 0), Math.tanh(Math.tanh(0.5)));
});

test("trade valuation uses symmetric marginal portfolio value", () => {
  const properties = [
    { tileId: 6, ownerId: 0, houses: 0, mortgaged: false },
    { tileId: 8, ownerId: 0, houses: 0, mortgaged: false },
    { tileId: 9, ownerId: 1, houses: 0, mortgaged: false },
  ];
  const property = properties[2];
  const buyerValue = tradeAssetValue(property, 0, properties, true);
  const sellerValue = tradeAssetValue(property, 1, properties, false);
  assert.ok(buyerValue > sellerValue);
});

test("trade advice rejects an unaffordable offer and canonicalizes proposal keys", () => {
  const game = {
    round: 2,
    lastDice: [3, 4],
    pending: null,
    players: [
      { id: 0, cash: 50, bankrupt: false },
      { id: 1, cash: 1500, bankrupt: false },
    ],
    properties: [{ tileId: 6, ownerId: 1, houses: 0, mortgaged: false }],
  };
  const result = evaluateTrade(game, neutralModel, game.players[0], {
    propertyIdsGiven: [],
    propertyIdsReceived: [6],
    cashGiven: 100,
    cashReceived: 0,
  });
  assert.equal(result.accepted, false);
  assert.equal(result.postCash, -50);
  assert.equal(
    tradeProposalKey({
      fromId: 0,
      toId: 1,
      fromCash: 20.9,
      toCash: 0,
      fromCards: 0,
      toCards: 0,
      fromPropertyIds: [9, 6],
      toPropertyIds: [],
    }),
    "0|1|20|0|0|0|6.9|",
  );
});

test("purchase, auction, and jail recommendations return executable decisions", () => {
  const player = { id: 0, cash: 1500, inJail: false, bankrupt: false };
  const game = {
    round: 1,
    players: [player, { id: 1, cash: 1500, bankrupt: false }],
    properties: [
      { tileId: 1, ownerId: null, houses: 0, mortgaged: false },
      { tileId: 3, ownerId: null, houses: 0, mortgaged: false },
    ],
  };
  const policy = { liquidity: 0.95, roi: 1.25, monopoly: 1.5, reserve: 240 };
  const tile = { id: 1, name: "地中海大道", group: "brown", price: 60, rent: 2 };

  assert.equal(recommendPurchase({ game, model: neutralModel, policy, player, tile }).buy, true);
  assert.equal(
    recommendAuction({ game, model: neutralModel, policy, player, tile, currentBid: 10 }).participate,
    true,
  );
  assert.equal(recommendJail({ game, model: neutralModel, player: { ...player, inJail: true } }).payToLeave, false);
  const buildGame = {
    ...game,
    properties: [
      { tileId: 1, ownerId: 0, houses: 0, mortgaged: false },
      { tileId: 3, ownerId: 0, houses: 0, mortgaged: false },
    ],
  };
  assert.deepEqual(
    recommendBuild({ game: buildGame, model: neutralModel, player }).tileIds,
    [1],
  );
});

test("payment advice only recommends fundraising when a legal asset action exists", () => {
  const player = { id: 0, cash: 0, bankrupt: false };
  const game = {
    players: [player, { id: 1, cash: 1500, bankrupt: false }],
    properties: [],
  };

  assert.equal(
    recommendPayment({ game, player, amount: 50 }).action,
    "declare-bankruptcy",
  );

  const withMortgageableProperty = {
    ...game,
    properties: [
      { tileId: 1, ownerId: 0, houses: 0, mortgaged: false },
      { tileId: 3, ownerId: 1, houses: 0, mortgaged: false },
    ],
  };
  assert.equal(
    recommendPayment({ game: withMortgageableProperty, player, amount: 50 }).action,
    "fundraise",
  );
});

test("build advice never consumes more buildings than the bank has available", () => {
  const player = { id: 0, cash: 5000, bankrupt: false };
  const game = {
    round: 1,
    players: [player, { id: 1, cash: 1500, bankrupt: false }],
    properties: [
      { tileId: 1, ownerId: 0, houses: 0, mortgaged: false },
      { tileId: 3, ownerId: 0, houses: 0, mortgaged: false },
      { tileId: 6, ownerId: 0, houses: 0, mortgaged: false },
      { tileId: 8, ownerId: 0, houses: 0, mortgaged: false },
      { tileId: 9, ownerId: 0, houses: 0, mortgaged: false },
      ...Array.from({ length: 7 }, (_, index) => ({
        tileId: 100 + index,
        ownerId: 1,
        houses: 4,
        mortgaged: false,
      })),
      { tileId: 107, ownerId: 1, houses: 3, mortgaged: false },
    ],
  };

  assert.deepEqual(
    recommendBuild({ game, model: neutralModel, player }).tileIds,
    [1],
  );
});

test("proactive trade recommendation returns a new executable offer after rejection", () => {
  const player = { id: 0, name: "你", cash: 1500, bankrupt: false };
  const game = {
    round: 2,
    activePlayerId: 0,
    lastDice: [3, 4],
    pending: null,
    tradeLockPlayerId: null,
    tradeLockedTileIds: [],
    players: [player, { id: 1, name: "AI", cash: 1500, bankrupt: false }],
    properties: [
      { tileId: 6, ownerId: 0, houses: 0, mortgaged: false },
      { tileId: 8, ownerId: 0, houses: 0, mortgaged: false },
      { tileId: 9, ownerId: 1, houses: 0, mortgaged: false },
    ],
  };

  const first = recommendTradeProposal({ game, model: neutralModel, player });
  assert.ok(first);
  assert.deepEqual(first.draft.toPropertyIds, [9]);
  assert.ok(first.draft.fromCash > 0);

  const next = recommendTradeProposal({
    game,
    model: neutralModel,
    player,
    rejectedKeys: [first.key],
  });
  assert.ok(next);
  assert.notEqual(next.key, first.key);
});
