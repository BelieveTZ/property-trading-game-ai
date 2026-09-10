import assert from "node:assert/strict";
import test from "node:test";

import {
  applyManagementPlan,
  applyRecordedCard,
  applyTrade,
  addGamePlayer,
  advanceTurn,
  canBuildOn,
  canDeclareBankruptcy,
  canMortgage,
  canEndTurn,
  canPayJailFee,
  canSettleAuction,
  canSellFrom,
  clearPaidDebt,
  declareBankruptcy,
  queuePayments,
  recordPayment,
  removeGamePlayer,
  resolvePendingDecision,
  resolveRollRules,
  recordDiceRoll,
  settlePropertyAuction,
  settleBankruptcy,
  useJailFreeCard,
  payJailFee,
} from "../app/game-rules.mjs";
import { TILES as boardTiles } from "../app/board-catalog.mjs";

const tiles = {
  6: { id: 6, type: "property", group: "sky", houseCost: 50 },
  8: { id: 8, type: "property", group: "sky", houseCost: 50 },
  9: { id: 9, type: "property", group: "sky", houseCost: 50 },
};

test("an asset-management plan applies legal actions through one interface", () => {
  const game = {
    players: [{ id: 0, cash: 200, bankrupt: false }],
    properties: [
      { tileId: 6, ownerId: 0, houses: 0, mortgaged: false },
      { tileId: 8, ownerId: 0, houses: 0, mortgaged: false },
      { tileId: 9, ownerId: 0, houses: 0, mortgaged: false },
    ],
    debt: null,
    paymentQueue: [],
  };

  const result = applyManagementPlan(
    game,
    [
      { kind: "build", tileId: 6 },
      { kind: "build", tileId: 8 },
    ],
    { tiles, groupSizes: { sky: 3 }, mortgageValues: { 6: 50, 8: 50, 9: 60 } },
  );

  assert.deepEqual(result.outcomes.map((outcome) => outcome.ok), [true, true]);
  assert.equal(result.game.players[0].cash, 100);
  assert.deepEqual(result.game.properties.map((property) => property.houses), [1, 1, 0]);
});

test("asset management rejects illegal mortgage and reports one stable reason", () => {
  const game = {
    players: [{ id: 0, cash: 200, bankrupt: false }],
    properties: [
      { tileId: 6, ownerId: 0, houses: 0, mortgaged: false },
      { tileId: 8, ownerId: 0, houses: 1, mortgaged: false },
      { tileId: 9, ownerId: 0, houses: 0, mortgaged: false },
    ],
    debt: null,
    paymentQueue: [],
  };

  const result = applyManagementPlan(
    game,
    [{ kind: "mortgage", tileId: 6 }],
    { tiles, groupSizes: { sky: 3 }, mortgageValues: { 6: 50 } },
  );

  assert.equal(result.outcomes[0].ok, false);
  assert.equal(result.outcomes[0].reason, "group-has-buildings");
  assert.deepEqual(result.game, game);
});

test("a player cannot build while any deed in the color group is mortgaged", () => {
  const properties = [
    { tileId: 6, ownerId: 0, houses: 0, mortgaged: false },
    { tileId: 8, ownerId: 0, houses: 0, mortgaged: true },
    { tileId: 9, ownerId: 0, houses: 0, mortgaged: false },
  ];

  assert.equal(
    canBuildOn({
      tileId: 6,
      playerId: 0,
      cash: 500,
      properties,
      tiles,
      groupSizes: { sky: 3 },
    }),
    false,
  );
});

test("buildings must be sold evenly from the most developed deed", () => {
  const properties = [
    { tileId: 6, ownerId: 0, houses: 1, mortgaged: false },
    { tileId: 8, ownerId: 0, houses: 2, mortgaged: false },
    { tileId: 9, ownerId: 0, houses: 2, mortgaged: false },
  ];

  assert.equal(canSellFrom({ tileId: 6, playerId: 0, properties, tiles }), false);
  assert.equal(canSellFrom({ tileId: 8, playerId: 0, properties, tiles }), true);
});

test("building actions respect the bank's 32 houses and 12 hotels", () => {
  const fullHouseSupply = [
    { tileId: 6, ownerId: 0, houses: 0, mortgaged: false },
    { tileId: 8, ownerId: 0, houses: 0, mortgaged: false },
    { tileId: 9, ownerId: 0, houses: 0, mortgaged: false },
    ...Array.from({ length: 8 }, (_, index) => ({
      tileId: 100 + index,
      ownerId: 1,
      houses: 4,
      mortgaged: false,
    })),
  ];
  assert.equal(
    canBuildOn({
      tileId: 6,
      playerId: 0,
      cash: 500,
      properties: fullHouseSupply,
      tiles,
      groupSizes: { sky: 3 },
    }),
    false,
  );

  const fullHotelSupply = [
    { tileId: 6, ownerId: 0, houses: 4, mortgaged: false },
    { tileId: 8, ownerId: 0, houses: 4, mortgaged: false },
    { tileId: 9, ownerId: 0, houses: 4, mortgaged: false },
    ...Array.from({ length: 12 }, (_, index) => ({
      tileId: 200 + index,
      ownerId: 1,
      houses: 5,
      mortgaged: false,
    })),
  ];
  assert.equal(
    canBuildOn({
      tileId: 6,
      playerId: 0,
      cash: 500,
      properties: fullHotelSupply,
      tiles,
      groupSizes: { sky: 3 },
    }),
    false,
  );

  const noHousesForHotelSale = [
    { tileId: 6, ownerId: 0, houses: 5, mortgaged: false },
    { tileId: 8, ownerId: 0, houses: 5, mortgaged: false },
    { tileId: 9, ownerId: 0, houses: 5, mortgaged: false },
    ...Array.from({ length: 8 }, (_, index) => ({
      tileId: 300 + index,
      ownerId: 1,
      houses: 4,
      mortgaged: false,
    })),
  ];
  assert.equal(
    canSellFrom({
      tileId: 6,
      playerId: 0,
      properties: noHousesForHotelSale,
      tiles,
    }),
    false,
  );
});

test("a deed cannot be mortgaged while its color group has buildings", () => {
  const properties = [
    { tileId: 6, ownerId: 0, houses: 0, mortgaged: false },
    { tileId: 8, ownerId: 0, houses: 1, mortgaged: false },
    { tileId: 9, ownerId: 0, houses: 0, mortgaged: false },
  ];

  assert.equal(canMortgage({ tileId: 6, playerId: 0, properties, tiles }), false);
});

test("paying to leave jail is unavailable below the fifty-dollar fee", () => {
  assert.equal(canPayJailFee(49), false);
  assert.equal(canPayJailFee(50), true);
});

test("jail doubles release the player without granting an extra turn", () => {
  const outcome = resolveRollRules({
    player: { position: 10, inJail: true, jailTurns: 1 },
    dieOne: 3,
    dieTwo: 3,
    doublesStreak: 2,
  });

  assert.equal(outcome.player.inJail, false);
  assert.equal(outcome.player.jailTurns, 0);
  assert.equal(outcome.shouldMove, true);
  assert.equal(outcome.extraTurnEligible, false);
  assert.equal(outcome.doublesStreak, 0);
});

test("three consecutive doubles send the player to jail without moving", () => {
  const outcome = resolveRollRules({
    player: { position: 7, inJail: false, jailTurns: 0 },
    dieOne: 4,
    dieTwo: 4,
    doublesStreak: 2,
  });

  assert.equal(outcome.player.position, 10);
  assert.equal(outcome.player.inJail, true);
  assert.equal(outcome.shouldMove, false);
  assert.equal(outcome.extraTurnEligible, false);
  assert.equal(outcome.doublesStreak, 0);
});

test("an auction only settles for a live winner at a positive whole-dollar price", () => {
  const winner = { cash: 100, bankrupt: false };
  assert.equal(canSettleAuction(winner, -10), false);
  assert.equal(canSettleAuction(winner, 10.5), false);
  assert.equal(canSettleAuction({ ...winner, bankrupt: true }, 10), false);
  assert.equal(canSettleAuction(winner, 101), false);
  assert.equal(canSettleAuction(winner, 100), true);
});

test("dice recording and landing form one deterministic state transition", () => {
  const game = {
    players: [
      { id: 0, name: "你", cash: 1500, position: 0, inJail: false, jailTurns: 0, bankrupt: false },
      { id: 1, name: "AI", cash: 1500, position: 0, inJail: false, jailTurns: 0, bankrupt: false },
    ],
    properties: [{ tileId: 6, ownerId: null, houses: 0, mortgaged: false }],
    activePlayerId: 0,
    doublesStreak: 0,
    extraTurnEligible: false,
    lastDice: null,
    pending: null,
    log: [],
  };
  const tiles = Array.from({ length: 40 }, (_, id) => ({ id, name: `格${id}`, type: "parking" }));
  tiles[6] = { id: 6, name: "东方大道", type: "property" };

  const result = recordDiceRoll(game, { dieOne: 2, dieTwo: 4 }, { tiles });
  assert.equal(result.ok, true);
  assert.equal(result.game.players[0].position, 6);
  assert.deepEqual(result.game.pending, { kind: "property", tileId: 6 });
  assert.deepEqual(game.pending, null);
});

function cardGame(overrides = {}) {
  return {
    players: [
      { id: 0, name: "你", cash: 200, position: 39, inJail: false, jailTurns: 0, jailFreeCards: 0, bankrupt: false },
      { id: 1, name: "AI 1", cash: 100, position: 0, inJail: false, jailTurns: 0, jailFreeCards: 0, bankrupt: false },
      { id: 2, name: "AI 2", cash: 100, position: 0, inJail: false, jailTurns: 0, jailFreeCards: 0, bankrupt: false },
    ],
    properties: [
      { tileId: 1, ownerId: null, houses: 0, mortgaged: false },
      { tileId: 5, ownerId: 1, houses: 0, mortgaged: false },
      { tileId: 6, ownerId: 0, houses: 2, mortgaged: false },
      { tileId: 8, ownerId: 0, houses: 5, mortgaged: false },
      { tileId: 12, ownerId: 1, houses: 0, mortgaged: false },
      { tileId: 35, ownerId: 1, houses: 0, mortgaged: false },
    ],
    activePlayerId: 0,
    doublesStreak: 0,
    extraTurnEligible: false,
    lastDice: [3, 4],
    pending: { kind: "card", deck: "chance" },
    debt: null,
    paymentQueue: [],
    log: [],
    ...overrides,
  };
}

const cardContext = {
  tiles: boardTiles,
  rentFor: () => 25,
};

test("recorded cash and movement cards settle through deterministic state transitions", () => {
  const debit = applyRecordedCard(
    cardGame({ players: cardGame().players.map((player, index) => index === 0 ? { ...player, cash: 40 } : player) }),
    { deck: "chance", card: { label: "罚款", effect: { kind: "cash", amount: -60 } } },
    cardContext,
  );
  assert.equal(debit.game.players[0].cash, -20);
  assert.equal(debit.game.debt.amount, 20);

  const move = applyRecordedCard(
    cardGame(),
    { deck: "chance", card: { label: "前往起点", effect: { kind: "move", destination: 0, collectGo: true } } },
    cardContext,
  );
  assert.equal(move.game.players[0].position, 0);
  assert.equal(move.game.players[0].cash, 400);
  assert.equal(move.game.pending.kind, "notice");

  const back = applyRecordedCard(
    cardGame({ players: cardGame().players.map((player, index) => index === 0 ? { ...player, position: 39 } : player) }),
    { deck: "chance", card: { label: "后退", effect: { kind: "back", spaces: 3 } } },
    cardContext,
  );
  assert.equal(back.game.players[0].position, 36);
  assert.deepEqual(back.game.pending, { kind: "card", deck: "chance" });
});

test("recorded nearest-space cards create the correct special rent", () => {
  const game = cardGame();
  game.players[0].position = 7;
  const result = applyRecordedCard(
    game,
    { deck: "chance", diceTotal: 7, card: { label: "最近公用事业", effect: { kind: "nearest", target: "utility" } } },
    cardContext,
  );
  assert.equal(result.game.players[0].position, 12);
  assert.deepEqual(result.game.pending, { kind: "rent", tileId: 12, ownerId: 1, amount: 70 });

  const stationGame = cardGame();
  stationGame.players[0].position = 34;
  const station = applyRecordedCard(
    stationGame,
    { deck: "chance", card: { label: "最近铁路", effect: { kind: "nearest", target: "station" } } },
    cardContext,
  );
  assert.deepEqual(station.game.pending, { kind: "rent", tileId: 35, ownerId: 1, amount: 50 });
});

test("recorded repair and per-player cards preserve debts and payment order", () => {
  const repairs = applyRecordedCard(
    cardGame(),
    { deck: "chance", card: { label: "维修", effect: { kind: "repairs", perHouse: 25, perHotel: 100 } } },
    cardContext,
  );
  assert.equal(repairs.game.players[0].cash, 50);
  assert.equal(repairs.game.debt, null);

  const payEach = applyRecordedCard(
    cardGame({ players: cardGame().players.map((player, index) => index === 0 ? { ...player, cash: 60 } : player) }),
    { deck: "chance", card: { label: "付给每人", effect: { kind: "payEach", amount: 50 } } },
    cardContext,
  );
  assert.equal(payEach.game.players[1].cash, 150);
  assert.equal(payEach.game.players[2].cash, 150);
  assert.equal(payEach.game.debt.amount, 40);
  assert.equal(payEach.game.debt.creditorId, 2);

  const collectGame = cardGame();
  collectGame.players[1].cash = 5;
  const collectEach = applyRecordedCard(
    collectGame,
    { deck: "chance", card: { label: "向每人收取", effect: { kind: "collectEach", amount: 10 } } },
    cardContext,
  );
  assert.equal(collectEach.game.players[0].cash, 210);
  assert.equal(collectEach.game.debt.debtorId, 1);
  assert.equal(collectEach.game.debt.amount, 5);
  assert.equal(collectEach.game.paymentQueue.length, 1);
});

test("recorded jail and jail-free cards update only the intended turn state", () => {
  const jailed = applyRecordedCard(
    cardGame(),
    { deck: "chance", card: { label: "入狱", effect: { kind: "jail" } } },
    cardContext,
  );
  assert.equal(jailed.game.players[0].position, 10);
  assert.equal(jailed.game.players[0].inJail, true);
  assert.equal(jailed.game.doublesStreak, 0);

  const free = applyRecordedCard(
    cardGame(),
    { deck: "chance", card: { label: "监狱通行证", effect: { kind: "jailFree" } } },
    cardContext,
  );
  assert.equal(free.game.players[0].jailFreeCards, 1);
  assert.equal(free.game.pending, null);
});

test("purchase and payment decisions settle without React-owned mutations", () => {
  const game = {
    players: [
      { id: 0, name: "你", cash: 100, bankrupt: false },
      { id: 1, name: "AI", cash: 100, bankrupt: false },
    ],
    properties: [{ tileId: 1, ownerId: null, houses: 0, mortgaged: false }],
    activePlayerId: 0,
    pending: { kind: "property", tileId: 1 },
    debt: null,
    log: [],
  };
  const bought = resolvePendingDecision(game, "buy", { tiles: boardTiles });
  assert.equal(bought.ok, true);
  assert.equal(bought.game.players[0].cash, 40);
  assert.equal(bought.game.properties[0].ownerId, 0);
  assert.equal(bought.game.pending, null);

  const rentGame = {
    ...bought.game,
    pending: { kind: "rent", tileId: 1, ownerId: 1, amount: 50 },
  };
  const rent = resolvePendingDecision(rentGame, "confirm", { tiles: boardTiles });
  assert.equal(rent.game.players[0].cash, -10);
  assert.equal(rent.game.players[1].cash, 150);
  assert.equal(rent.game.debt.amount, 10);
});

test("jail exits and player roster changes are validated state transitions", () => {
  const game = {
    players: [
      { id: 0, name: "你", cash: 50, inJail: true, jailTurns: 2, jailFreeCards: 1, bankrupt: false },
      { id: 1, name: "AI 1", cash: 100, bankrupt: false },
      { id: 2, name: "AI 2", cash: 100, bankrupt: false },
    ],
    properties: [{ tileId: 1, ownerId: 2, houses: 0, mortgaged: false }],
    activePlayerId: 0,
    myPlayerId: 0,
    doublesStreak: 1,
    pending: null,
    debt: null,
    paymentQueue: [],
    log: [],
  };
  const used = useJailFreeCard(game, 0);
  assert.equal(used.ok, true);
  assert.equal(used.game.players[0].jailFreeCards, 0);
  assert.equal(used.game.players[0].inJail, false);

  const paid = payJailFee(game, 0);
  assert.equal(paid.ok, true);
  assert.equal(paid.game.players[0].cash, 0);
  assert.equal(paid.game.players[0].inJail, false);

  const added = addGamePlayer(game, { id: 3, name: "AI 3", cash: 1500, bankrupt: false }, 5);
  assert.equal(added.game.players.length, 4);
  const removed = removeGamePlayer(added.game, 2, 3);
  assert.equal(removed.ok, true);
  assert.equal(removed.game.properties[0].ownerId, null);

  const auctionGame = {
    ...added.game,
    activePlayerId: 0,
    pending: { kind: "property", tileId: 1 },
  };
  const duringAuction = removeGamePlayer(auctionGame, 0, 3);
  assert.equal(duringAuction.ok, true);
  assert.deepEqual(duringAuction.game.pending, { kind: "property", tileId: 1 });
  assert.equal(duringAuction.game.lastDice, null);
  assert.equal(duringAuction.game.extraTurnEligible, false);
  assert.equal(duringAuction.game.tradeLockPlayerId, null);
  assert.deepEqual(duringAuction.game.tradeLockedTileIds, []);
});

test("removing the active player preserves seat order and round boundaries", () => {
  const players = Array.from({ length: 4 }, (_, id) => ({
    id,
    name: `玩家 ${id}`,
    cash: 1500,
    bankrupt: false,
  }));
  const base = {
    players,
    properties: [{ tileId: 1, ownerId: null, houses: 0, mortgaged: false }],
    myPlayerId: 0,
    round: 2,
    lastDice: [3, 4],
    doublesStreak: 0,
    extraTurnEligible: false,
    tradeLockPlayerId: null,
    tradeLockedTileIds: [],
    pending: null,
    debt: null,
    paymentQueue: [],
    log: [],
  };

  const middle = removeGamePlayer({ ...base, activePlayerId: 2 }, 2, 3);
  assert.equal(middle.game.activePlayerId, 3);
  assert.equal(middle.game.round, 2);

  const wrapped = removeGamePlayer({ ...base, activePlayerId: 3 }, 3, 3);
  assert.equal(wrapped.game.activePlayerId, 0);
  assert.equal(wrapped.game.round, 3);

  const auction = removeGamePlayer(
    {
      ...base,
      activePlayerId: 2,
      pending: { kind: "property", tileId: 1 },
    },
    2,
    3,
  );
  assert.equal(auction.game.activePlayerId, 3);
  assert.deepEqual(auction.game.pending, { kind: "property", tileId: 1 });
});

test("the last surviving player ends the game and cannot take another turn", () => {
  const game = {
    players: [
      { id: 0, name: "你", cash: 500, bankrupt: false, inJail: false },
      { id: 1, name: "AI 1", cash: 0, bankrupt: true, inJail: false },
      { id: 2, name: "AI 2", cash: 0, bankrupt: true, inJail: false },
    ],
    properties: [],
    activePlayerId: 0,
    round: 3,
    pending: null,
    debt: null,
    paymentQueue: [],
    auctionQueue: [],
    lastDice: null,
    doublesStreak: 0,
    extraTurnEligible: false,
    tradeLockPlayerId: null,
    tradeLockedTileIds: [],
    log: [],
  };
  const finished = advanceTurn(game, 0);
  assert.equal(finished.ok, true);
  assert.equal(finished.game.gameOver, true);
  assert.equal(finished.game.winnerId, 0);
  assert.equal(finished.game.round, 3);

  const repeated = advanceTurn(finished.game, 0);
  assert.equal(repeated.ok, false);
  assert.equal(repeated.reason, "game-over");
  const roll = recordDiceRoll(
    finished.game,
    { dieOne: 1, dieTwo: 2 },
    { tiles: boardTiles },
  );
  assert.equal(roll.ok, false);
  assert.equal(roll.reason, "game-over");
  assert.equal(
    addGamePlayer(
      finished.game,
      { id: 3, name: "AI 3", cash: 1500, bankrupt: false },
      5,
    ).reason,
    "game-over",
  );
  assert.equal(removeGamePlayer(finished.game, 1, 2).reason, "game-over");
  assert.equal(declareBankruptcy(finished.game, 0).reason, "game-over");
});

test("bankruptcy immediately records the sole survivor as winner", () => {
  const game = {
    players: [
      { id: 0, name: "你", cash: -50, jailFreeCards: 0, bankrupt: false },
      { id: 1, name: "AI 1", cash: 0, jailFreeCards: 0, bankrupt: true },
      { id: 2, name: "AI 2", cash: 500, jailFreeCards: 0, bankrupt: false },
    ],
    properties: [],
    activePlayerId: 0,
    round: 3,
    pending: null,
    debt: { debtorId: 0, creditorId: 2, amount: 50, reason: "rent" },
    paymentQueue: [],
    auctionQueue: [],
    lastDice: [2, 3],
    doublesStreak: 0,
    extraTurnEligible: false,
    tradeLockPlayerId: null,
    tradeLockedTileIds: [],
    log: [],
  };

  const result = declareBankruptcy(game, 0);
  assert.equal(result.ok, true);
  assert.equal(result.game.gameOver, true);
  assert.equal(result.game.winnerId, 2);
  assert.equal(result.game.activePlayerId, 2);
  assert.equal(result.game.pending, null);
});

test("auction settlement and turn advancement are atomic transitions", () => {
  const game = {
    players: [
      { id: 0, name: "你", cash: 500, bankrupt: false, inJail: false },
      { id: 1, name: "AI", cash: 500, bankrupt: false, inJail: false },
    ],
    properties: [{ tileId: 6, ownerId: null, houses: 0, mortgaged: false }],
    activePlayerId: 0,
    round: 1,
    pending: { kind: "property", tileId: 6 },
    auctionQueue: [],
    lastDice: [2, 4],
    doublesStreak: 0,
    extraTurnEligible: false,
    tradeLockPlayerId: 0,
    tradeLockedTileIds: [6],
    debt: null,
    log: [],
  };
  const auction = settlePropertyAuction(game, { winnerId: 1, price: 120 }, { tiles: { 6: { name: "东方大道" } } });
  assert.equal(auction.ok, true);
  assert.equal(auction.game.players[1].cash, 380);
  assert.equal(auction.game.properties[0].ownerId, 1);

  const turn = advanceTurn(auction.game, 0);
  assert.equal(turn.ok, true);
  assert.equal(turn.game.activePlayerId, 1);
  assert.equal(turn.game.lastDice, null);
  assert.deepEqual(turn.game.tradeLockedTileIds, []);
});

test("a solvent player cannot be declared bankrupt", () => {
  const game = {
    players: [{ id: 0, cash: 1, bankrupt: false }],
    debt: null,
  };
  assert.equal(canDeclareBankruptcy(game, 0), false);
  game.players[0].cash = -1;
  assert.equal(canDeclareBankruptcy(game, 0), true);
});

test("bankruptcy is a single validated transition and cannot be undone by repeating it", () => {
  const game = {
    players: [
      { id: 0, cash: -50, jailFreeCards: 0, bankrupt: false },
      { id: 1, cash: 500, jailFreeCards: 0, bankrupt: false },
    ],
    properties: [{ tileId: 6, ownerId: 0, houses: 0, mortgaged: false }],
    activePlayerId: 0,
    round: 1,
    lastDice: [2, 3],
    doublesStreak: 0,
    extraTurnEligible: false,
    tradeLockPlayerId: null,
    tradeLockedTileIds: [],
    pending: null,
    debt: { debtorId: 0, creditorId: 1, amount: 50, reason: "rent" },
    paymentQueue: [],
    auctionQueue: [],
  };
  const declared = declareBankruptcy(game, 0);
  assert.equal(declared.ok, true);
  assert.equal(declared.game.players[0].bankrupt, true);
  assert.equal(declared.game.properties[0].ownerId, 1);

  const repeated = declareBankruptcy(declared.game, 0);
  assert.equal(repeated.ok, false);
  assert.equal(repeated.reason, "game-over");
  assert.equal(repeated.game.players[0].bankrupt, true);
});

test("an unpaid player debt blocks the turn until it is resolved", () => {
  const game = {
    players: [
      { id: 0, cash: 100, bankrupt: false },
      { id: 1, cash: 1500, bankrupt: false },
    ],
    debt: null,
  };

  const next = recordPayment(game, {
    debtorId: 0,
    creditorId: 1,
    amount: 250,
    reason: "rent",
  });

  assert.equal(next.players[0].cash, -150);
  assert.equal(next.players[1].cash, 1750);
  assert.deepEqual(next.debt, {
    debtorId: 0,
    creditorId: 1,
    amount: 150,
    reason: "rent",
  });
  assert.equal(canEndTurn(next, 0), false);
});

test("partial fundraising updates the remaining debt amount", () => {
  const game = {
    players: [{ id: 0, cash: -20, bankrupt: false }],
    debt: { debtorId: 0, creditorId: null, amount: 150, reason: "tax" },
  };

  assert.equal(clearPaidDebt(game).debt.amount, 20);
  game.players[0].cash = 0;
  assert.equal(clearPaidDebt(game).debt, null);
});

test("collect-from-each pauses on an insolvent payer and resumes the queue", () => {
  const game = {
    players: [
      { id: 0, cash: 100, bankrupt: false },
      { id: 1, cash: 5, bankrupt: false },
      { id: 2, cash: 100, bankrupt: false },
    ],
    debt: null,
    paymentQueue: [],
  };

  const waiting = queuePayments(game, [
    { debtorId: 1, creditorId: 0, amount: 10, reason: "birthday" },
    { debtorId: 2, creditorId: 0, amount: 10, reason: "birthday" },
  ]);

  assert.equal(waiting.players[0].cash, 110);
  assert.equal(waiting.players[1].cash, -5);
  assert.equal(waiting.players[2].cash, 100);
  assert.deepEqual(waiting.debt, {
    debtorId: 1,
    creditorId: 0,
    amount: 5,
    reason: "birthday",
  });
  assert.equal(waiting.paymentQueue.length, 1);

  waiting.players[1].cash = 0;
  const complete = clearPaidDebt(waiting);
  assert.equal(complete.players[0].cash, 120);
  assert.equal(complete.players[2].cash, 90);
  assert.equal(complete.debt, null);
  assert.deepEqual(complete.paymentQueue, []);
});

test("bankruptcy during collect-from-each claws back the shortfall then continues", () => {
  const waiting = queuePayments(
    {
      players: [
        { id: 0, cash: 100, jailFreeCards: 0, bankrupt: false },
        { id: 1, cash: 5, jailFreeCards: 0, bankrupt: false },
        { id: 2, cash: 100, jailFreeCards: 0, bankrupt: false },
      ],
      properties: [],
      activePlayerId: 0,
      debt: null,
      paymentQueue: [],
    },
    [
      { debtorId: 1, creditorId: 0, amount: 10, reason: "birthday" },
      { debtorId: 2, creditorId: 0, amount: 10, reason: "birthday" },
    ],
  );

  const result = settleBankruptcy(waiting, 1).game;

  assert.equal(result.players[0].cash, 115);
  assert.equal(result.players[1].bankrupt, true);
  assert.equal(result.players[2].cash, 90);
  assert.equal(result.debt, null);
  assert.deepEqual(result.paymentQueue, []);
});

test("a bilateral trade atomically moves cash, deeds, cards, interest and locks", () => {
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
  assert.equal(result.interestTotal, 8);
  assert.equal(result.game.players[0].cash, 572);
  assert.equal(result.game.players[1].cash, 420);
  assert.equal(result.game.players[0].jailFreeCards, 0);
  assert.equal(result.game.players[1].jailFreeCards, 1);
  assert.equal(result.game.properties[0].ownerId, 1);
  assert.equal(result.game.properties[1].ownerId, 0);
  assert.deepEqual(result.game.tradeLockedTileIds, [6, 12]);
});

test("a trade rejects ownership changes and developed color groups", () => {
  const game = {
    players: [
      { id: 0, cash: 500, jailFreeCards: 0, bankrupt: false },
      { id: 1, cash: 500, jailFreeCards: 0, bankrupt: false },
    ],
    properties: [
      { tileId: 6, ownerId: 0, houses: 0, mortgaged: false },
      { tileId: 8, ownerId: 0, houses: 1, mortgaged: false },
    ],
    activePlayerId: 0,
    tradeLockPlayerId: null,
    tradeLockedTileIds: [],
    debt: null,
    paymentQueue: [],
  };
  const draft = {
    fromId: 0,
    toId: 1,
    fromCash: 0,
    toCash: 0,
    fromCards: 0,
    toCards: 0,
    fromPropertyIds: [6],
    toPropertyIds: [],
  };

  assert.equal(
    applyTrade(game, draft, { colorGroups: { 6: [6, 8] } }).reason,
    "group-has-buildings",
  );
  assert.equal(
    applyTrade(
      {
        ...game,
        properties: game.properties.map((property) => ({ ...property, houses: 0 })),
      },
      { ...draft, fromPropertyIds: [99] },
    ).reason,
    "ownership-changed",
  );
});

test("a turn cannot end while any active player has negative cash", () => {
  const game = {
    players: [
      { id: 0, cash: 100, bankrupt: false },
      { id: 1, cash: -1, bankrupt: false },
    ],
    activePlayerId: 0,
    pending: null,
    debt: null,
  };

  assert.equal(canEndTurn(game, 0), false);
});

test("bankruptcy to a player transfers assets without creating cash", () => {
  const game = {
    players: [
      { id: 0, cash: -150, jailFreeCards: 1, bankrupt: false },
      { id: 1, cash: 1750, jailFreeCards: 1, bankrupt: false },
    ],
    properties: [
      { tileId: 6, ownerId: 0, houses: 0, mortgaged: false },
    ],
    debt: { debtorId: 0, creditorId: 1, amount: 150, reason: "rent" },
  };

  const result = settleBankruptcy(game, 0, {
    mortgageValues: { 6: 50 },
    buildingCosts: { 6: 50 },
  });

  assert.equal(result.game.players[0].cash, 0);
  assert.equal(result.game.players[0].jailFreeCards, 0);
  assert.equal(result.game.players[0].bankrupt, true);
  assert.equal(result.game.players[1].cash, 1600);
  assert.equal(result.game.players[1].jailFreeCards, 2);
  assert.equal(result.game.properties[0].ownerId, 1);
  assert.deepEqual(result.auctionTileIds, []);
});

test("bankruptcy to the bank returns and auctions every deed", () => {
  const game = {
    players: [{ id: 0, cash: -10, jailFreeCards: 0, bankrupt: false }],
    properties: [
      { tileId: 6, ownerId: 0, houses: 0, mortgaged: true },
      { tileId: 8, ownerId: 0, houses: 2, mortgaged: false },
    ],
    debt: { debtorId: 0, creditorId: null, amount: 10, reason: "tax" },
  };

  const result = settleBankruptcy(game, 0);

  assert.deepEqual(result.auctionTileIds, [6, 8]);
  assert.deepEqual(result.game.pending, { kind: "property", tileId: 6 });
  assert.deepEqual(result.game.auctionQueue, [8]);
  assert.deepEqual(
    result.game.properties.map(({ ownerId, houses, mortgaged }) => ({
      ownerId,
      houses,
      mortgaged,
    })),
    [
      { ownerId: null, houses: 0, mortgaged: false },
      { ownerId: null, houses: 0, mortgaged: false },
    ],
  );
});

test("an active player's bankruptcy hands a fresh turn to the next survivor", () => {
  const game = {
    players: [
      { id: 0, cash: -1, jailFreeCards: 0, bankrupt: false },
      { id: 1, cash: 1500, jailFreeCards: 0, bankrupt: false },
      { id: 2, cash: 1500, jailFreeCards: 0, bankrupt: false },
    ],
    properties: [],
    activePlayerId: 0,
    round: 1,
    lastDice: [3, 4],
    doublesStreak: 1,
    extraTurnEligible: true,
    debt: { debtorId: 0, creditorId: null, amount: 1, reason: "tax" },
  };

  const result = settleBankruptcy(game, 0);

  assert.equal(result.game.activePlayerId, 1);
  assert.equal(result.game.lastDice, null);
  assert.equal(result.game.doublesStreak, 0);
  assert.equal(result.game.extraTurnEligible, false);
});
