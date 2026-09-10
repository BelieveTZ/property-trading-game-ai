import { TILES } from "./board-catalog.mjs";
import { clearPaidDebt } from "./game-rules.mjs";
import {
  MAX_PLAYERS,
  MIN_PLAYERS,
  createPlayer,
  migrateLegacyAiNames,
  pickUnusedAiName,
} from "./player-catalog.mjs";

export const SESSION_STORAGE_KEY = "monopoly-ai-session-v3";
export const LEGACY_SESSION_STORAGE_KEYS = Object.freeze([
  "deed-advisor-session-v3",
]);

export function createFreshSession(game, policy) {
  const activePlayers = game.players.filter((player) => !player.bankrupt);
  const fromId = activePlayers[0]?.id ?? game.players[0]?.id ?? 0;
  const toId =
    activePlayers.find((player) => player.id !== fromId)?.id ?? fromId;
  return {
    game: structuredClone(game),
    policy: structuredClone(policy),
    auction: { open: false, winnerId: fromId, price: 60 },
    tradeDraft: {
      fromId,
      toId,
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
}

export function createSessionSnapshot(session) {
  return {
    ...structuredClone(session),
    version: 3,
  };
}

function migrateLegacySession(value) {
  if (!value || typeof value !== "object" || value.version !== undefined) {
    throw new Error("Unsupported session version");
  }
  const players = Array.isArray(value.game?.players)
    ? value.game.players.filter((player) => player.bankrupt !== true)
    : [];
  const fromId = players[0]?.id ?? 0;
  const toId = players[1]?.id ?? fromId;
  return createSessionSnapshot({
    game: value.game,
    policy: value.policy,
    auction: { open: false, winnerId: fromId, price: 60 },
    tradeDraft: {
      fromId,
      toId,
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
}

export function readSessionSnapshot(value, options = {}) {
  const parsed = typeof value === "string" ? JSON.parse(value) : value;
  const isCurrent = parsed?.version === 3;
  if (isCurrent) parseSessionSnapshot(parsed, options);
  const current = isCurrent ? parsed : migrateLegacySession(parsed);
  const game = normalizeGameDocument(current.game);
  const normalized = {
    ...current,
    game,
    auction: game.gameOver
      ? { ...current.auction, open: false }
      : current.auction,
  };
  return parseSessionSnapshot(normalized, options);
}

export function writeSessionSnapshot(session, { space = 0 } = {}) {
  return JSON.stringify(createSessionSnapshot(session), null, space);
}

export function normalizeGameDocument(game) {
  const next = structuredClone(game);
  next.players = Array.isArray(next.players)
    ? next.players.slice(0, MAX_PLAYERS)
    : [];
  next.players = next.players.map((player) => ({
    ...player,
    jailFreeCards: Number.isFinite(Number(player.jailFreeCards))
      ? Math.max(0, Number(player.jailFreeCards))
      : 0,
  }));
  next.extraTurnEligible = next.extraTurnEligible === true;
  next.debt = next.debt ?? null;
  next.paymentQueue = Array.isArray(next.paymentQueue) ? next.paymentQueue : [];
  next.auctionQueue = Array.isArray(next.auctionQueue)
    ? next.auctionQueue.filter((tileId) => Number.isInteger(tileId))
    : [];

  const usedIds = new Set(next.players.map((player) => player.id));
  while (next.players.length < MIN_PLAYERS) {
    const id =
      Array.from({ length: MAX_PLAYERS }, (_, candidate) => candidate).find(
        (candidate) => !usedIds.has(candidate),
      ) ?? next.players.length;
    next.players.push(createPlayer(id, pickUnusedAiName(next.players)));
    usedIds.add(id);
  }
  next.players = migrateLegacyAiNames(next.players, next.myPlayerId);
  next.players.sort((a, b) => a.id - b.id);
  const survivors = next.players.filter((player) => !player.bankrupt);
  next.gameOver = survivors.length === 1;
  next.winnerId = next.gameOver ? survivors[0].id : null;
  if (next.gameOver) {
    next.activePlayerId = next.winnerId;
    next.pending = null;
    next.debt = null;
    next.paymentQueue = [];
    next.auctionQueue = [];
    next.lastDice = null;
    next.doublesStreak = 0;
    next.extraTurnEligible = false;
    next.tradeLockPlayerId = null;
    next.tradeLockedTileIds = [];
  }

  const validIds = new Set(next.players.map((player) => player.id));
  const validPropertyIds = new Set(
    next.properties.map((property) => property.tileId),
  );
  next.paymentQueue = next.paymentQueue.filter(
    (payment) =>
      validIds.has(payment.debtorId) &&
      validIds.has(payment.creditorId) &&
      payment.debtorId !== payment.creditorId &&
      Number.isFinite(payment.amount) &&
      payment.amount > 0,
  );
  if (Array.isArray(next.tradeLockedTileIds)) {
    next.tradeLockPlayerId =
      next.tradeLockPlayerId !== null && validIds.has(next.tradeLockPlayerId)
        ? next.tradeLockPlayerId
        : null;
    next.tradeLockedTileIds = Array.from(
      new Set(
        next.tradeLockedTileIds.filter((tileId) => validPropertyIds.has(tileId)),
      ),
    );
  } else {
    const log = Array.isArray(next.log) ? next.log : [];
    const turnBoundaryIndex = log.findIndex((entry) => entry.startsWith("轮到 "));
    const currentTurnLog =
      turnBoundaryIndex >= 0 ? log.slice(0, turnBoundaryIndex) : log;
    const completedTradeLog = currentTurnLog.filter((entry) =>
      entry.includes("完成交易："),
    );
    next.tradeLockedTileIds = next.properties
      .filter((property) => {
        const tileName = TILES[property.tileId]?.name;
        return (
          Boolean(tileName) &&
          completedTradeLog.some((entry) => entry.includes(tileName))
        );
      })
      .map((property) => property.tileId);
    next.tradeLockPlayerId = next.tradeLockedTileIds.length
      ? next.activePlayerId
      : null;
  }
  next.properties = next.properties.map((property) =>
    property.ownerId !== null && !validIds.has(property.ownerId)
      ? { ...property, ownerId: null, houses: 0, mortgaged: false }
      : property,
  );
  const fallback =
    next.players.find((player) => !player.bankrupt) ?? next.players[0];
  if (!validIds.has(next.activePlayerId)) {
    next.activePlayerId = fallback.id;
    next.doublesStreak = 0;
    next.extraTurnEligible = false;
    next.lastDice = null;
    next.pending = null;
  }
  if (!validIds.has(next.myPlayerId)) next.myPlayerId = fallback.id;
  if (next.pending?.kind === "rent" && !validIds.has(next.pending.ownerId)) {
    next.pending = null;
  }
  if (next.pending?.kind === "card" && !("deck" in next.pending)) {
    next.pending = null;
  }
  if (
    next.debt &&
    (!validIds.has(next.debt.debtorId) ||
      (next.debt.creditorId !== null && !validIds.has(next.debt.creditorId)))
  ) {
    next.debt = null;
  }
  return clearPaidDebt(next);
}

/**
 * @param {unknown} value
 * @param {{ deedTileIds?: number[] }} [options]
 */
export function parseSessionSnapshot(value, { deedTileIds = [] } = {}) {
  if (!value || value.version !== 3) {
    throw new Error("Unsupported session version");
  }
  const {
    game,
    policy,
    auction,
    tradeDraft,
    rejectedTradeProposalKeys,
    buildDecisionCompleted,
  } = value;
  const players = game?.players;
  const properties = game?.properties;
  const validPlayers =
    Array.isArray(players) &&
    players.length >= 3 &&
    players.length <= 5 &&
    players.every(
      (player) =>
        Number.isInteger(player?.id) &&
        (player.cash === undefined || Number.isFinite(player.cash)) &&
        (player.position === undefined ||
          (Number.isInteger(player.position) &&
            player.position >= 0 &&
            player.position < 40)),
    ) &&
    new Set(players.map((player) => player.id)).size === players.length;
  const playerIds = new Set(
    Array.isArray(players) ? players.map((player) => player.id) : [],
  );
  const activePlayerIds = Array.isArray(players)
    ? players.filter((player) => player.bankrupt !== true).map((player) => player.id)
    : [];
  const propertyIds = Array.isArray(properties)
    ? properties.map((property) => property.tileId)
    : [];
  const validProperties =
    Array.isArray(properties) &&
    new Set(propertyIds).size === propertyIds.length &&
    properties.every(
      (property) =>
        Number.isInteger(property?.tileId) &&
        (property.ownerId === null || playerIds.has(property.ownerId)) &&
        Number.isInteger(property.houses) &&
        property.houses >= 0 &&
        property.houses <= 5 &&
        typeof property.mortgaged === "boolean" &&
        (!property.mortgaged || property.houses === 0),
    ) &&
    (deedTileIds.length === 0 ||
      (propertyIds.length === deedTileIds.length &&
        deedTileIds.every((tileId) => propertyIds.includes(tileId))));
  const validReferences =
    (game?.activePlayerId === undefined || playerIds.has(game.activePlayerId)) &&
    (game?.myPlayerId === undefined || playerIds.has(game.myPlayerId)) &&
    (game?.auctionQueue === undefined ||
      (Array.isArray(game.auctionQueue) &&
        game.auctionQueue.every((tileId) => propertyIds.includes(tileId)))) &&
    (game?.paymentQueue === undefined ||
      (Array.isArray(game.paymentQueue) &&
        game.paymentQueue.every(
          (payment) =>
            activePlayerIds.includes(payment?.debtorId) &&
            activePlayerIds.includes(payment?.creditorId) &&
            payment.debtorId !== payment.creditorId &&
            Number.isFinite(payment.amount) &&
            payment.amount > 0 &&
            typeof payment.reason === "string",
        )));
  const survivorIds = activePlayerIds;
  const validOutcome =
    (game?.gameOver === undefined && game?.winnerId === undefined) ||
    (typeof game?.gameOver === "boolean" &&
      (game.gameOver
        ? survivorIds.length === 1 &&
          game.winnerId === survivorIds[0] &&
          game.activePlayerId === game.winnerId &&
          game.pending === null &&
          game.debt === null &&
          (game.paymentQueue === undefined || game.paymentQueue.length === 0) &&
          (game.auctionQueue === undefined || game.auctionQueue.length === 0) &&
          (game.lastDice === undefined || game.lastDice === null) &&
          game.extraTurnEligible !== true &&
          (game.tradeLockPlayerId === undefined ||
            game.tradeLockPlayerId === null) &&
          (game.tradeLockedTileIds === undefined ||
            game.tradeLockedTileIds.length === 0) &&
          auction?.open === false
        : game.winnerId === null));
  const validPolicy =
    policy &&
    typeof policy === "object" &&
    Object.values(policy).every(
      (item) => typeof item !== "number" || Number.isFinite(item),
    );
  const validFlow =
    auction &&
    typeof auction.open === "boolean" &&
    Number.isInteger(auction.winnerId) &&
    playerIds.has(auction.winnerId) &&
    Number.isFinite(auction.price) &&
    auction.price >= 0 &&
    (!auction.open ||
      (game?.pending?.kind === "property" &&
        properties.some(
          (property) =>
            property.tileId === game.pending.tileId &&
            property.ownerId === null,
        ))) &&
    tradeDraft &&
    playerIds.has(tradeDraft.fromId) &&
    playerIds.has(tradeDraft.toId) &&
    (tradeDraft.fromId !== tradeDraft.toId || activePlayerIds.length < 2) &&
    [
      tradeDraft.fromCash,
      tradeDraft.toCash,
      tradeDraft.fromCards,
      tradeDraft.toCards,
    ].every((amount) => Number.isFinite(amount) && amount >= 0) &&
    Array.isArray(tradeDraft.fromPropertyIds ?? []) &&
    Array.isArray(tradeDraft.toPropertyIds ?? []) &&
    [...(tradeDraft.fromPropertyIds ?? []), ...(tradeDraft.toPropertyIds ?? [])].every(
      (tileId) => propertyIds.includes(tileId),
    ) &&
    Array.isArray(rejectedTradeProposalKeys) &&
    rejectedTradeProposalKeys.every((key) => typeof key === "string") &&
    typeof buildDecisionCompleted === "boolean";
  if (
    !validPlayers ||
    !validProperties ||
    !validReferences ||
    !validOutcome ||
    !validPolicy ||
    !validFlow
  ) {
    throw new Error("Invalid session data");
  }
  return {
    game,
    policy,
    auction,
    tradeDraft,
    rejectedTradeProposalKeys,
    buildDecisionCompleted,
  };
}
