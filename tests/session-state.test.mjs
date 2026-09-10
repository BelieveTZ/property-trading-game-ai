import assert from "node:assert/strict";
import test from "node:test";

import {
  createFreshSession,
  createSessionSnapshot,
  LEGACY_SESSION_STORAGE_KEYS,
  readSessionSnapshot,
  parseSessionSnapshot,
  SESSION_STORAGE_KEY,
  writeSessionSnapshot,
} from "../app/session-state.mjs";

test("uses the MonopolyAI storage namespace while preserving the previous key for migration", () => {
  assert.equal(SESSION_STORAGE_KEY, "monopoly-ai-session-v3");
  assert.deepEqual(LEGACY_SESSION_STORAGE_KEYS, ["deed-advisor-session-v3"]);
});

test("starting a new game resets every persisted decision flow", () => {
  const game = {
    players: [
      { id: 0, name: "你", cash: 1500, bankrupt: false },
      { id: 1, name: "AI 1", cash: 1500, bankrupt: false },
      { id: 2, name: "AI 2", cash: 1500, bankrupt: false },
    ],
    properties: [
      { tileId: 1, ownerId: null, houses: 0, mortgaged: false },
    ],
    activePlayerId: 0,
    myPlayerId: 0,
    pending: null,
  };
  const policy = { reserve: 240 };
  const fresh = createFreshSession(game, policy);
  assert.deepEqual(fresh.auction, { open: false, winnerId: 0, price: 60 });
  assert.deepEqual(fresh.tradeDraft, {
    fromId: 0,
    toId: 1,
    fromCash: 0,
    toCash: 0,
    fromCards: 0,
    toCards: 0,
    fromPropertyIds: [],
    toPropertyIds: [],
  });
  assert.deepEqual(fresh.rejectedTradeProposalKeys, []);
  assert.equal(fresh.buildDecisionCompleted, false);
});

test("a finished game remains a valid round-trippable session", () => {
  const game = {
    players: [
      { id: 0, name: "你", cash: 1500, bankrupt: false },
      { id: 1, name: "AI 1", cash: 0, bankrupt: true },
      { id: 2, name: "AI 2", cash: 0, bankrupt: true },
    ],
    properties: [
      { tileId: 1, ownerId: null, houses: 0, mortgaged: false },
    ],
    activePlayerId: 0,
    myPlayerId: 0,
    gameOver: true,
    winnerId: 0,
    pending: null,
    debt: null,
    paymentQueue: [],
    auctionQueue: [],
    lastDice: null,
    doublesStreak: 0,
    extraTurnEligible: false,
    tradeLockPlayerId: null,
    tradeLockedTileIds: [],
  };
  const session = createFreshSession(game, { reserve: 240 });
  const restored = readSessionSnapshot(writeSessionSnapshot(session));
  assert.equal(restored.game.gameOver, true);
  assert.equal(restored.game.winnerId, 0);

  const contradictory = structuredClone(session);
  contradictory.game.pending = { kind: "property", tileId: 1 };
  contradictory.auction = { open: true, winnerId: 0, price: 60 };
  assert.throws(
    () => readSessionSnapshot(writeSessionSnapshot(contradictory)),
    /Invalid session data/,
  );

  const wrongActivePlayer = structuredClone(session);
  wrongActivePlayer.game.activePlayerId = 1;
  assert.throws(
    () => readSessionSnapshot(writeSessionSnapshot(wrongActivePlayer)),
    /Invalid session data/,
  );
});

test("an in-progress decision round-trips through a versioned session", () => {
  const session = {
    game: {
      players: [{ id: 0 }, { id: 1 }, { id: 2 }],
      properties: [
        { tileId: 6, ownerId: null, houses: 0, mortgaged: false },
      ],
      pending: { kind: "property", tileId: 6 },
    },
    policy: { generation: 120 },
    auction: { open: true, winnerId: 1, price: 121 },
    tradeDraft: {
      fromId: 0,
      toId: 1,
      fromCash: 0,
      toCash: 130,
      fromCards: 0,
      toCards: 0,
      fromPropertyIds: [6],
      toPropertyIds: [],
    },
    rejectedTradeProposalKeys: ["0|1|0|130|0|0|6|"],
    buildDecisionCompleted: true,
  };

  const saved = createSessionSnapshot(session);
  const restored = parseSessionSnapshot(JSON.parse(JSON.stringify(saved)));

  assert.equal(saved.version, 3);
  assert.deepEqual(restored, session);
});

test("the application owns the exported session version", () => {
  const saved = createSessionSnapshot({ version: 999, marker: "kept" });

  assert.equal(saved.version, 3);
  assert.equal(saved.marker, "kept");
});

test("legacy game and policy data migrate through the session document interface", () => {
  const restored = readSessionSnapshot({
    game: {
      players: [{ id: 0 }, { id: 1 }, { id: 2 }],
      properties: [],
      pending: null,
    },
    policy: { generation: 12 },
  });

  assert.equal(restored.policy.generation, 12);
  assert.deepEqual(restored.auction, { open: false, winnerId: 0, price: 60 });
  assert.deepEqual(restored.tradeDraft.fromId, 0);
  assert.deepEqual(restored.tradeDraft.toId, 1);
});

test("session reading normalizes legacy game fields before returning the document", () => {
  const restored = readSessionSnapshot({
    game: {
      players: [
        { id: 0, name: "你", cash: 1500, position: 0, bankrupt: false },
        { id: 1, name: "林舟", cash: 1500, position: 0, bankrupt: false },
      ],
      properties: [
        { tileId: 6, ownerId: 99, houses: 0, mortgaged: false },
      ],
      activePlayerId: 99,
      myPlayerId: 0,
      pending: null,
      log: [],
    },
    policy: { generation: 12 },
  });

  assert.equal(restored.game.players.length, 3);
  assert.notEqual(restored.game.players[1].name, "林舟");
  assert.equal(restored.game.properties[0].ownerId, null);
  assert.equal(restored.game.activePlayerId, 0);
  assert.deepEqual(restored.game.paymentQueue, []);
  assert.deepEqual(restored.game.tradeLockedTileIds, []);
});

test("session text round-trips through the same document interface", () => {
  const session = {
    game: {
      players: [{ id: 0 }, { id: 1 }, { id: 2 }],
      properties: [],
      pending: null,
    },
    policy: { generation: 12 },
    auction: { open: false, winnerId: 0, price: 60 },
    tradeDraft: {
      fromId: 0,
      toId: 1,
      fromCash: 0,
      toCash: 0,
      fromCards: 0,
      toCards: 0,
      fromPropertyIds: [],
      toPropertyIds: [],
    },
    rejectedTradeProposalKeys: [],
    buildDecisionCompleted: false,
  };

  const restored = readSessionSnapshot(writeSessionSnapshot(session));
  assert.deepEqual(restored.policy, session.policy);
  assert.deepEqual(restored.auction, session.auction);
  assert.equal(restored.game.players.length, 3);
  assert.deepEqual(restored.game.paymentQueue, []);
  assert.deepEqual(restored.game.tradeLockedTileIds, []);
});

test("an imported session rejects duplicate players and impossible ownership", () => {
  const invalid = createSessionSnapshot({
    game: {
      players: [
        { id: 0, cash: 1500, position: 0 },
        { id: 0, cash: 1500, position: 0 },
        { id: 2, cash: Number.NaN, position: 0 },
      ],
      properties: [
        { tileId: 6, ownerId: 99, houses: 0, mortgaged: false },
        { tileId: 8, ownerId: null, houses: 0, mortgaged: false },
      ],
    },
    policy: {},
    auction: { open: false, winnerId: 0, price: 60 },
    tradeDraft: {},
    rejectedTradeProposalKeys: [],
    buildDecisionCompleted: false,
  });

  assert.throws(
    () => parseSessionSnapshot(invalid, { deedTileIds: [6, 8] }),
    /invalid session/i,
  );
});

test("reading a current session rejects an invalid owner instead of repairing it", () => {
  const invalid = createSessionSnapshot({
    game: {
      players: [{ id: 0 }, { id: 1 }, { id: 2 }],
      properties: [
        { tileId: 6, ownerId: 99, houses: 0, mortgaged: false },
      ],
      pending: null,
    },
    policy: {},
    auction: { open: false, winnerId: 0, price: 60 },
    tradeDraft: {
      fromId: 0,
      toId: 1,
      fromCash: 0,
      toCash: 0,
      fromCards: 0,
      toCards: 0,
      fromPropertyIds: [],
      toPropertyIds: [],
    },
    rejectedTradeProposalKeys: [],
    buildDecisionCompleted: false,
  });

  assert.throws(
    () => readSessionSnapshot(invalid, { deedTileIds: [6] }),
    /invalid session/i,
  );
});

test("reading a current session rejects invalid payment references", () => {
  const invalid = createSessionSnapshot({
    game: {
      players: [{ id: 0 }, { id: 1 }, { id: 2 }],
      properties: [],
      pending: null,
      paymentQueue: [
        { debtorId: 99, creditorId: 0, amount: 10, reason: "birthday" },
      ],
    },
    policy: {},
    auction: { open: false, winnerId: 0, price: 60 },
    tradeDraft: {
      fromId: 0,
      toId: 1,
      fromCash: 0,
      toCash: 0,
      fromCards: 0,
      toCards: 0,
      fromPropertyIds: [],
      toPropertyIds: [],
    },
    rejectedTradeProposalKeys: [],
    buildDecisionCompleted: false,
  });

  assert.throws(() => readSessionSnapshot(invalid), /invalid session/i);
});

test("an imported open auction must reference its pending unsold deed", () => {
  const inconsistent = createSessionSnapshot({
    game: {
      players: [{ id: 0 }, { id: 1 }, { id: 2 }],
      properties: [],
      pending: null,
    },
    policy: {},
    auction: { open: true, winnerId: 1, price: 100 },
    tradeDraft: {
      fromId: 0,
      toId: 1,
      fromCash: 0,
      toCash: 0,
      fromCards: 0,
      toCards: 0,
      fromPropertyIds: [],
      toPropertyIds: [],
    },
    rejectedTradeProposalKeys: [],
    buildDecisionCompleted: false,
  });

  assert.throws(() => parseSessionSnapshot(inconsistent), /invalid session/i);
});

test("an imported trade draft must reference two valid players", () => {
  const invalid = createSessionSnapshot({
    game: {
      players: [{ id: 0 }, { id: 1 }, { id: 2 }],
      properties: [],
      pending: null,
    },
    policy: {},
    auction: { open: false, winnerId: 0, price: 60 },
    tradeDraft: {
      fromId: 99,
      toId: 99,
      fromCash: 0,
      toCash: 0,
      fromCards: 0,
      toCards: 0,
      fromPropertyIds: [],
      toPropertyIds: [],
    },
    rejectedTradeProposalKeys: [],
    buildDecisionCompleted: false,
  });

  assert.throws(() => parseSessionSnapshot(invalid), /invalid session/i);
});

test("an imported bankruptcy auction queue only contains known deeds", () => {
  const invalid = createSessionSnapshot({
    game: {
      players: [{ id: 0 }, { id: 1 }, { id: 2 }],
      properties: [
        { tileId: 6, ownerId: null, houses: 0, mortgaged: false },
      ],
      pending: null,
      auctionQueue: [999],
    },
    policy: {},
    auction: { open: false, winnerId: 0, price: 60 },
    tradeDraft: {
      fromId: 0,
      toId: 1,
      fromCash: 0,
      toCash: 0,
      fromCards: 0,
      toCards: 0,
      fromPropertyIds: [],
      toPropertyIds: [],
    },
    rejectedTradeProposalKeys: [],
    buildDecisionCompleted: false,
  });

  assert.throws(
    () => parseSessionSnapshot(invalid, { deedTileIds: [6] }),
    /invalid session/i,
  );
});

test("an imported card payment queue only references live players", () => {
  const invalid = createSessionSnapshot({
    game: {
      players: [{ id: 0 }, { id: 1 }, { id: 2 }],
      properties: [],
      pending: null,
      paymentQueue: [
        { debtorId: 99, creditorId: 0, amount: 10, reason: "birthday" },
      ],
    },
    policy: {},
    auction: { open: false, winnerId: 0, price: 60 },
    tradeDraft: {
      fromId: 0,
      toId: 1,
      fromCash: 0,
      toCash: 0,
      fromCards: 0,
      toCards: 0,
      fromPropertyIds: [],
      toPropertyIds: [],
    },
    rejectedTradeProposalKeys: [],
    buildDecisionCompleted: false,
  });

  assert.throws(() => parseSessionSnapshot(invalid), /invalid session/i);
});

test("an imported card payment queue rejects bankrupt debtors and creditors", () => {
  for (const payment of [
    { debtorId: 2, creditorId: 0, amount: 10, reason: "birthday" },
    { debtorId: 0, creditorId: 2, amount: 10, reason: "chairman" },
  ]) {
    const invalid = createSessionSnapshot({
      game: {
        players: [
          { id: 0, bankrupt: false },
          { id: 1, bankrupt: false },
          { id: 2, bankrupt: true },
        ],
        properties: [],
        pending: null,
        paymentQueue: [payment],
      },
      policy: {},
      auction: { open: false, winnerId: 0, price: 60 },
      tradeDraft: {
        fromId: 0,
        toId: 1,
        fromCash: 0,
        toCash: 0,
        fromCards: 0,
        toCards: 0,
        fromPropertyIds: [],
        toPropertyIds: [],
      },
      rejectedTradeProposalKeys: [],
      buildDecisionCompleted: false,
    });

    assert.throws(() => parseSessionSnapshot(invalid), /invalid session/i);
  }
});
