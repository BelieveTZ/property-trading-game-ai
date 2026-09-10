import assert from "node:assert/strict";
import test from "node:test";

import {
  applyStandaloneAction,
  createStandaloneSession,
  legalManagementActions,
  legalStandaloneActions,
  planDebtLiquidation,
  readStandaloneSession,
  replayStandaloneGame,
  recommendStandaloneAction,
  standaloneActorId,
  writeStandaloneSession,
} from "../app/standalone-session.mjs";

const policy = {
  liquidity: 0.95,
  roi: 1.25,
  monopoly: 1.5,
  pressure: 0.8,
  reserve: 240,
  games: 0,
  generation: 0,
  fitness: 0,
};

test("a standalone session owns random dice, automatic cards and replay", () => {
  const initial = createStandaloneSession({
    mode: "play",
    playerCount: 3,
    seed: 20260910,
    policy,
  });
  const sameSeed = createStandaloneSession({
    mode: "play",
    playerCount: 3,
    seed: 20260910,
    policy,
  });
  assert.deepEqual(sameSeed.initialGame, initial.initialGame);
  assert.deepEqual(legalStandaloneActions(initial), ["roll"]);

  initial.game.players[0].position = 5;
  const rolled = applyStandaloneAction(initial, { kind: "roll", dice: [1, 1] });
  assert.equal(rolled.ok, true);
  assert.deepEqual(rolled.session.game.lastDice, [1, 1]);
  assert.equal(rolled.session.runtime.decks.chance.cursor, 1);
  assert.notEqual(rolled.session.game.pending?.kind, "card");
  assert.deepEqual(rolled.session.actions, [{ kind: "roll", dice: [1, 1] }]);

  const replayed = replayStandaloneGame({
    mode: "play",
    playerCount: 3,
    seed: 20260910,
    policy,
    initialGame: initial.game,
    actions: rolled.session.actions,
  });
  assert.deepEqual(replayed.game, rolled.session.game);
  assert.deepEqual(replayed.runtime, rolled.session.runtime);
});

test("the standalone interface rejects an action outside the current legal set", () => {
  const session = createStandaloneSession({
    mode: "spectate",
    playerCount: 4,
    seed: 19,
    policy,
  });
  const result = applyStandaloneAction(session, { kind: "end-turn" });
  assert.equal(result.ok, false);
  assert.equal(result.reason, "illegal-action");
  assert.deepEqual(result.session, session);
});

test("an open auction advances bidder by bidder and settles only after rivals pass", () => {
  const session = createStandaloneSession({ playerCount: 3, seed: 31, policy });
  session.game.pending = { kind: "property", tileId: 1 };

  const opened = applyStandaloneAction(session, { kind: "decline" });
  assert.equal(opened.ok, true);
  assert.deepEqual(legalStandaloneActions(opened.session), ["auction-bid", "auction-pass"]);
  assert.equal(opened.session.auction.activeBidderId, 0);

  const bid = applyStandaloneAction(opened.session, { kind: "auction-bid", amount: 1 });
  assert.equal(bid.ok, true);
  assert.equal(bid.session.auction.highBidderId, 0);
  assert.equal(bid.session.auction.currentBid, 1);
  assert.equal(bid.session.auction.activeBidderId, 1);

  const firstPass = applyStandaloneAction(bid.session, { kind: "auction-pass" });
  assert.equal(firstPass.ok, true);
  assert.equal(firstPass.session.auction.activeBidderId, 2);
  const settled = applyStandaloneAction(firstPass.session, { kind: "auction-pass" });
  assert.equal(settled.ok, true);
  assert.equal(settled.session.auction.open, false);
  assert.equal(settled.session.game.properties.find((item) => item.tileId === 1).ownerId, 0);
  assert.equal(settled.session.game.players[0].cash, 1499);
});

test("debt liquidation sells buildings evenly before mortgaging deeds", () => {
  const session = createStandaloneSession({ playerCount: 3, seed: 32, policy });
  session.game.players[0].cash = -100;
  session.game.debt = { debtorId: 0, creditorId: null, amount: 100, reason: "test" };
  for (const tileId of [1, 3]) {
    const deed = session.game.properties.find((item) => item.tileId === tileId);
    deed.ownerId = 0;
    deed.houses = 1;
  }
  const actions = planDebtLiquidation(session);
  assert.deepEqual(actions.slice(0, 2), [
    { kind: "sell-building", tileId: 1 },
    { kind: "sell-building", tileId: 3 },
  ]);
  const raised = applyStandaloneAction(session, { kind: "manage", actions });
  assert.equal(raised.ok, true);
  assert.equal(raised.session.game.debt, null);
  assert.ok(raised.session.game.players[0].cash >= 0);
});

test("bilateral negotiation supports three counteroffers and an atomic acceptance", () => {
  let session = createStandaloneSession({ playerCount: 3, seed: 33, policy });
  session.game.lastDice = [2, 3];
  session.game.properties.find((item) => item.tileId === 1).ownerId = 0;
  let draft = {
    fromId: 0,
    toId: 1,
    fromCash: 0,
    toCash: 60,
    fromCards: 0,
    toCards: 0,
    fromPropertyIds: [1],
    toPropertyIds: [],
  };
  let result = applyStandaloneAction(session, { kind: "trade-propose", draft });
  assert.equal(result.ok, true);
  assert.deepEqual(legalStandaloneActions(result.session), [
    "trade-accept",
    "trade-reject",
    "trade-counter",
  ]);

  for (const price of [70, 80, 90]) {
    draft = { ...draft, toCash: price };
    result = applyStandaloneAction(result.session, { kind: "trade-counter", draft });
    assert.equal(result.ok, true);
  }
  assert.deepEqual(legalStandaloneActions(result.session), ["trade-accept", "trade-reject"]);
  const accepted = applyStandaloneAction(result.session, { kind: "trade-accept" });
  assert.equal(accepted.ok, true);
  assert.equal(accepted.session.game.properties.find((item) => item.tileId === 1).ownerId, 1);
  assert.equal(accepted.session.game.players[0].cash, 1590);
  assert.equal(accepted.session.negotiation, null);
  assert.match(accepted.session.game.log[0], /接受报价并完成交易/);
});

test("standalone saves validate by deterministic replay and reject tampering", () => {
  const initial = createStandaloneSession({ playerCount: 3, seed: 34, policy });
  const rolled = applyStandaloneAction(initial, { kind: "roll" });
  assert.equal(rolled.ok, true);
  const text = writeStandaloneSession(rolled.session);
  assert.deepEqual(readStandaloneSession(text), rolled.session);

  const tampered = JSON.parse(text);
  tampered.game.players[0].cash += 1;
  assert.throws(() => readStandaloneSession(tampered), /does not match replay/);

  const tamperedAuction = JSON.parse(text);
  tamperedAuction.auction = {
    open: true,
    tileId: 1,
    activeBidderId: 0,
    highBidderId: 0,
    currentBid: 1,
    passedIds: [],
  };
  assert.throws(() => readStandaloneSession(tamperedAuction), /does not match replay/);

  const tamperedNegotiation = JSON.parse(text);
  tamperedNegotiation.negotiation = {
    open: true,
    proposerId: 0,
    responderId: 1,
    counterCount: 0,
    draft: tamperedNegotiation.tradeDraft,
  };
  assert.throws(() => readStandaloneSession(tamperedNegotiation), /does not match replay/);
});

test("management choices and advice are always members of the current legal set", () => {
  const session = createStandaloneSession({ playerCount: 3, seed: 35, policy });
  session.game.lastDice = [1, 2];
  session.game.properties.find((item) => item.tileId === 1).ownerId = 0;
  const management = legalManagementActions(session);
  assert.ok(management.some((action) => action.kind === "mortgage" && action.tileId === 1));
  const advice = recommendStandaloneAction(session);
  assert.ok(legalStandaloneActions(session).includes(advice.kind));
});

test("queued debt is controlled by its debtor and bank-held deeds enter auction directly", () => {
  let session = createStandaloneSession({ playerCount: 3, seed: 36, policy });
  session.game.activePlayerId = 0;
  session.game.players[1].cash = -1;
  session.game.debt = { debtorId: 1, creditorId: null, amount: 1, reason: "queue" };
  session.game.properties.find((item) => item.tileId === 1).ownerId = 1;
  assert.equal(standaloneActorId(session), 1);

  const bankrupt = applyStandaloneAction(session, { kind: "bankrupt" });
  assert.equal(bankrupt.ok, true);
  assert.equal(bankrupt.session.auction.open, true);
  assert.equal(bankrupt.session.auction.tileId, 1);
  assert.deepEqual(legalStandaloneActions(bankrupt.session), ["auction-bid", "auction-pass"]);
});
