import { CARD_DECKS } from "./card-catalog.mjs";
import { DEED_DETAILS, GROUP_SIZES, TILES } from "./board-catalog.mjs";
import {
  advanceTurn,
  applyManagementPlan,
  applyRecordedCard,
  applyTrade,
  canBuildOn,
  canMortgage,
  canSellFrom,
  declareBankruptcy,
  payJailFee,
  recordDiceRoll,
  resolvePendingDecision,
  settlePropertyAuction,
  skipPropertyAuction,
  spendJailFreeCard,
} from "./game-rules.mjs";
import { createGameRuntime, drawRuntimeCard, rollRuntimeDice } from "./game-runtime.mjs";
import { createPlayer, OFFICIAL_AI_NAMES } from "./player-catalog.mjs";
import { propertyRent } from "./rent-rules.mjs";

const MANAGEMENT_RULES = {
  tiles: TILES,
  groupSizes: GROUP_SIZES,
  mortgageValues: Object.fromEntries(
    Object.entries(DEED_DETAILS).map(([tileId, deed]) => [Number(tileId), deed.mortgage]),
  ),
};

export const STANDALONE_SESSION_VERSION = 2;

function closedAuction() {
  return {
    open: false,
    tileId: null,
    activeBidderId: null,
    highBidderId: null,
    currentBid: 0,
    passedIds: [],
  };
}

function openAuction(game) {
  const bidderIds = game.players
    .filter((player) => !player.bankrupt)
    .map((player) => player.id);
  const firstIndex = Math.max(0, bidderIds.indexOf(game.activePlayerId));
  return {
    open: true,
    tileId: game.pending?.kind === "property" ? game.pending.tileId : null,
    activeBidderId: bidderIds[firstIndex] ?? bidderIds[0] ?? null,
    highBidderId: null,
    currentBid: 0,
    passedIds: [],
  };
}

function createGame(playerCount) {
  return {
    players: Array.from({ length: playerCount }, (_, index) =>
      createPlayer(index, index === 0 ? "你" : OFFICIAL_AI_NAMES[index - 1]),
    ),
    properties: TILES.filter((tile) =>
      ["property", "station", "utility"].includes(tile.type),
    ).map((tile) => ({
      tileId: tile.id,
      ownerId: null,
      houses: 0,
      mortgaged: false,
    })),
    activePlayerId: 0,
    myPlayerId: 0,
    tradeLockPlayerId: null,
    tradeLockedTileIds: [],
    round: 1,
    doublesStreak: 0,
    extraTurnEligible: false,
    lastDice: null,
    pending: null,
    debt: null,
    paymentQueue: [],
    auctionQueue: [],
    gameOver: false,
    winnerId: null,
    log: ["第 1 轮开始 · 所有人从启程格出发"],
  };
}

function rulesContext() {
  return {
    tiles: TILES,
    rentFor: ({ tile, property, ownerProperties, diceTotal }) =>
      propertyRent(tile, property, ownerProperties, diceTotal),
  };
}

export function createTradeDraft(fromId, toId) {
  return {
    fromId,
    toId,
    fromCash: 0,
    toCash: 0,
    fromCards: 0,
    toCards: 0,
    fromPropertyIds: [],
    toPropertyIds: [],
  };
}

function defaultTradeDraft(game) {
  const active = game.players.filter((player) => !player.bankrupt);
  return createTradeDraft(
    active[0]?.id ?? 0,
    active[1]?.id ?? active[0]?.id ?? 0,
  );
}

function prependLog(game, message) {
  game.log = [message, ...(Array.isArray(game.log) ? game.log : [])].slice(0, 60);
}

/**
 * @param {{ mode?: "play" | "spectate", playerCount?: number, seed?: number, policy?: any, initialGame?: any }} [options]
 */
export function createStandaloneSession({
  mode = "play",
  playerCount = 4,
  seed = Date.now(),
  policy,
  initialGame,
} = {}) {
  if (!['play', 'spectate'].includes(mode)) throw new Error("Unsupported game mode");
  if (!Number.isInteger(playerCount) || playerCount < 3 || playerCount > 5) {
    throw new Error("Player count must be between 3 and 5");
  }
  const game = structuredClone(initialGame ?? createGame(playerCount));
  return {
    version: STANDALONE_SESSION_VERSION,
    mode,
    humanPlayerId: mode === "play" ? game.players[0].id : null,
    initialGame: structuredClone(game),
    game,
    runtime: createGameRuntime(seed, CARD_DECKS),
    policy: structuredClone(policy ?? {}),
    auction: closedAuction(),
    tradeDraft: defaultTradeDraft(game),
    negotiation: null,
    rejectedTradeProposalKeys: [],
    tradeAttemptsThisTurn: [],
    historyComplete: true,
    actions: [],
  };
}

export function standaloneActorId(session) {
  return session.auction.open
    ? session.auction.activeBidderId
    : session.negotiation?.open
      ? session.negotiation.responderId
      : session.game.debt
        ? session.game.debt.debtorId
        : session.game.activePlayerId;
}

export function legalStandaloneActions(session) {
  const { game, auction } = session;
  if (game.gameOver) return [];
  if (auction.open) return ["auction-bid", "auction-pass"];
  if (session.negotiation?.open) {
    return [
      "trade-accept",
      "trade-reject",
      ...(session.negotiation.counterCount < 3 ? ["trade-counter"] : []),
    ];
  }
  if (game.debt) return ["manage", "trade-propose", "bankrupt"];
  if (game.pending?.kind === "property") {
    const tile = TILES[game.pending.tileId];
    const player = game.players.find((candidate) => candidate.id === game.activePlayerId);
    return [
      ...(player && player.cash >= (tile?.price ?? Infinity) ? ["buy"] : []),
      "decline",
    ];
  }
  if (["rent", "tax", "notice"].includes(game.pending?.kind)) return ["confirm"];
  if (game.pending?.kind === "card") return ["draw-card"];
  if (game.lastDice === null) {
    const player = game.players.find((candidate) => candidate.id === game.activePlayerId);
    return [
      "roll",
      ...(player?.inJail && player.jailFreeCards > 0 ? ["use-pass"] : []),
      ...(player?.inJail && player.cash >= 50 ? ["pay-fee"] : []),
    ];
  }
  return [
    "manage",
    ...(!session.tradeAttemptsThisTurn.includes(game.activePlayerId)
      ? ["trade-propose"]
      : []),
    "end-turn",
  ];
}

export function legalManagementActions(session, playerId = standaloneActorId(session)) {
  const player = session.game.players.find(
    (candidate) => candidate.id === playerId && !candidate.bankrupt,
  );
  if (!player) return [];
  const actions = [];
  for (const property of session.game.properties) {
    if (property.ownerId !== playerId) continue;
    if (
      canSellFrom({
        tileId: property.tileId,
        playerId,
        properties: session.game.properties,
        tiles: TILES,
      })
    ) {
      actions.push({ kind: "sell-building", tileId: property.tileId });
    }
    if (
      canMortgage({
        tileId: property.tileId,
        playerId,
        properties: session.game.properties,
        tiles: TILES,
      })
    ) {
      actions.push({ kind: "mortgage", tileId: property.tileId });
    }
    if (!session.game.debt && property.mortgaged) {
      const cost = Math.ceil((MANAGEMENT_RULES.mortgageValues[property.tileId] ?? 0) * 1.1);
      if (player.cash >= cost) actions.push({ kind: "unmortgage", tileId: property.tileId });
    }
    if (
      !session.game.debt &&
      canBuildOn({
        tileId: property.tileId,
        playerId,
        cash: player.cash,
        properties: session.game.properties,
        tiles: TILES,
        groupSizes: GROUP_SIZES,
      })
    ) {
      actions.push({ kind: "build", tileId: property.tileId });
    }
  }
  return actions;
}

export function recommendStandaloneAction(session) {
  const legal = legalStandaloneActions(session);
  if (!legal.length) return null;
  const priority = [
    "trade-accept",
    "confirm",
    "use-pass",
    "pay-fee",
    "buy",
    "auction-bid",
    "roll",
    "manage",
    "end-turn",
    "trade-reject",
    "decline",
    "auction-pass",
    "bankrupt",
  ];
  const kind = priority.find((candidate) => legal.includes(candidate)) ?? legal[0];
  return { kind };
}

function auctionPlayers(session) {
  return session.game.players.filter((player) => !player.bankrupt);
}

function nextAuctionBidder(session, afterId) {
  const players = auctionPlayers(session);
  const start = Math.max(0, players.findIndex((player) => player.id === afterId));
  for (let offset = 1; offset <= players.length; offset += 1) {
    const candidate = players[(start + offset) % players.length];
    if (
      !session.auction.passedIds.includes(candidate.id) &&
      candidate.id !== session.auction.highBidderId
    ) {
      return candidate.id;
    }
  }
  return null;
}

function settleOrAdvanceAuction(session, lastActorId) {
  const nextBidderId = nextAuctionBidder(session, lastActorId);
  if (nextBidderId !== null) {
    session.auction.activeBidderId = nextBidderId;
    return { ok: true, reason: null, game: session.game };
  }
  const result = session.auction.highBidderId === null
    ? skipPropertyAuction(session.game)
    : settlePropertyAuction(
        session.game,
        {
          winnerId: session.auction.highBidderId,
          price: session.auction.currentBid,
        },
        { tiles: TILES },
      );
  if (!result.ok) return result;
  session.game = result.game;
  session.auction = result.nextTileId === undefined
    ? closedAuction()
    : openAuction(result.game);
  return { ok: true, reason: null, game: session.game };
}

function validateTradeDraft(game, draft) {
  if (!draft || typeof draft !== "object") {
    return { ok: false, reason: "invalid-contents", game };
  }
  return applyTrade(game, draft, tradeContext(game, draft));
}

export function planDebtLiquidation(session) {
  const debtorId = session.game.debt?.debtorId;
  if (debtorId === undefined) return [];
  let game = structuredClone(session.game);
  const actions = [];
  const cash = () => game.players.find((player) => player.id === debtorId)?.cash ?? -Infinity;
  for (let guard = 0; guard < 64 && cash() < 0; guard += 1) {
    const sale = game.properties
      .filter((property) =>
        canSellFrom({
          tileId: property.tileId,
          playerId: debtorId,
          properties: game.properties,
          tiles: TILES,
        }),
      )
      .sort((left, right) => right.houses - left.houses || left.tileId - right.tileId)[0];
    const mortgage = sale
      ? null
      : game.properties
          .filter((property) =>
            canMortgage({
              tileId: property.tileId,
              playerId: debtorId,
              properties: game.properties,
              tiles: TILES,
            }),
          )
          .sort((left, right) => left.tileId - right.tileId)[0];
    const action = sale
      ? { kind: "sell-building", tileId: sale.tileId }
      : mortgage
        ? { kind: "mortgage", tileId: mortgage.tileId }
        : null;
    if (!action) break;
    const result = applyManagementPlan(game, [action], MANAGEMENT_RULES);
    if (!result.outcomes[0]?.ok) break;
    actions.push(action);
    game = result.game;
  }
  return actions;
}

function drawPendingCards(session) {
  let next = session;
  for (let guard = 0; guard < 4 && next.game.pending?.kind === "card"; guard += 1) {
    const deck = next.game.pending.deck;
    const draw = drawRuntimeCard(next.runtime, deck, CARD_DECKS);
    let runtime = draw.runtime;
    let diceTotal = 7;
    if (draw.card.effect.kind === "nearest" && draw.card.effect.target === "utility") {
      const roll = rollRuntimeDice(runtime);
      runtime = roll.runtime;
      diceTotal = roll.dice[0] + roll.dice[1];
    }
    const result = applyRecordedCard(
      next.game,
      { deck, card: draw.card, diceTotal },
      rulesContext(),
    );
    if (!result.ok) return { ...next, runtime };
    next = { ...next, game: result.game, runtime };
  }
  return next;
}

function tradeContext(game, draft) {
  const selected = [...draft.fromPropertyIds, ...draft.toPropertyIds];
  return {
    mortgageValues: MANAGEMENT_RULES.mortgageValues,
    colorGroups: Object.fromEntries(
      selected.map((tileId) => {
        const group = TILES[tileId]?.group;
        return [
          tileId,
          group
            ? game.properties
                .filter((property) => TILES[property.tileId]?.group === group)
                .map((property) => property.tileId)
            : [tileId],
        ];
      }),
    ),
  };
}

export function applyStandaloneAction(session, action, { record = true } = {}) {
  const legal = legalStandaloneActions(session);
  if (!legal.includes(action?.kind)) {
    return { ok: false, reason: "illegal-action", session: structuredClone(session) };
  }
  let next = structuredClone(session);
  let result;

  if (action.kind === "roll") {
    let dice = action.dice;
    if (!dice) {
      const roll = rollRuntimeDice(next.runtime);
      next.runtime = roll.runtime;
      dice = roll.dice;
    }
    if (
      !Array.isArray(dice) ||
      dice.length !== 2 ||
      dice.some((die) => !Number.isInteger(die) || die < 1 || die > 6)
    ) {
      return { ok: false, reason: "invalid-dice", session: structuredClone(session) };
    }
    result = recordDiceRoll(
      next.game,
      { dieOne: dice[0], dieTwo: dice[1] },
      rulesContext(),
    );
  } else if (action.kind === "buy") {
    result = resolvePendingDecision(next.game, "buy", { tiles: TILES });
  } else if (action.kind === "decline") {
    result = resolvePendingDecision(next.game, "skip", { tiles: TILES });
    if (result.ok && result.auction) next.auction = openAuction(result.game);
  } else if (action.kind === "confirm") {
    result = resolvePendingDecision(next.game, "confirm", { tiles: TILES });
  } else if (action.kind === "end-turn") {
    result = advanceTurn(next.game, next.game.activePlayerId);
    if (result.ok) next.tradeAttemptsThisTurn = [];
  } else if (action.kind === "auction-bid") {
    const bidderId = next.auction.activeBidderId;
    const bidder = next.game.players.find((player) => player.id === bidderId);
    if (
      !bidder ||
      !Number.isInteger(action.amount) ||
      action.amount <= next.auction.currentBid ||
      action.amount > bidder.cash
    ) {
      result = { ok: false, reason: "invalid-bid", game: next.game };
    } else {
      next.auction.currentBid = action.amount;
      next.auction.highBidderId = bidderId;
      result = settleOrAdvanceAuction(next, bidderId);
    }
  } else if (action.kind === "auction-pass") {
    const bidderId = next.auction.activeBidderId;
    next.auction.passedIds = [...new Set([...next.auction.passedIds, bidderId])];
    result = settleOrAdvanceAuction(next, bidderId);
  } else if (action.kind === "use-pass") {
    result = spendJailFreeCard(next.game, next.game.activePlayerId);
  } else if (action.kind === "pay-fee") {
    result = payJailFee(next.game, next.game.activePlayerId);
  } else if (action.kind === "manage") {
    const managed = applyManagementPlan(next.game, action.actions ?? [], MANAGEMENT_RULES);
    const failed = managed.outcomes.find((outcome) => !outcome.ok);
    result = { ok: !failed, reason: failed?.reason ?? null, game: managed.game };
    if (result.ok && managed.outcomes.length) {
      const player = managed.game.players.find(
        (candidate) => candidate.id === managed.outcomes[0].playerId,
      );
      prependLog(managed.game, `${player?.name ?? "玩家"} 完成 ${managed.outcomes.length} 项资产操作`);
    }
  } else if (action.kind === "trade-propose") {
    const validation = validateTradeDraft(next.game, action.draft);
    if (!validation.ok) {
      result = validation;
    } else {
      next.negotiation = {
        open: true,
        proposerId: action.draft.fromId,
        responderId: action.draft.toId,
        counterCount: 0,
        draft: structuredClone(action.draft),
      };
      next.tradeAttemptsThisTurn = [
        ...new Set([...next.tradeAttemptsThisTurn, action.draft.fromId]),
      ];
      const proposer = next.game.players.find((player) => player.id === action.draft.fromId);
      const responder = next.game.players.find((player) => player.id === action.draft.toId);
      prependLog(next.game, `${proposer?.name ?? "玩家"} 向 ${responder?.name ?? "玩家"} 提出交易`);
      result = { ok: true, reason: null, game: next.game };
    }
  } else if (action.kind === "trade-counter") {
    const negotiation = next.negotiation;
    const participants = new Set([action.draft?.fromId, action.draft?.toId]);
    const expected = new Set([negotiation.proposerId, negotiation.responderId]);
    const validation = validateTradeDraft(next.game, action.draft);
    if (
      negotiation.counterCount >= 3 ||
      participants.size !== 2 ||
      [...participants].some((id) => !expected.has(id)) ||
      !validation.ok
    ) {
      result = { ok: false, reason: validation.reason ?? "invalid-counter", game: next.game };
    } else {
      next.negotiation = {
        ...negotiation,
        proposerId: negotiation.responderId,
        responderId: negotiation.proposerId,
        counterCount: negotiation.counterCount + 1,
        draft: structuredClone(action.draft),
      };
      const proposer = next.game.players.find(
        (player) => player.id === next.negotiation.proposerId,
      );
      prependLog(next.game, `${proposer?.name ?? "玩家"} 提出第 ${next.negotiation.counterCount} 次反报价`);
      result = { ok: true, reason: null, game: next.game };
    }
  } else if (action.kind === "trade-accept") {
    result = applyTrade(
      next.game,
      next.negotiation.draft,
      tradeContext(next.game, next.negotiation.draft),
    );
    if (result.ok) {
      const responder = result.game.players.find(
        (player) => player.id === next.negotiation.responderId,
      );
      prependLog(result.game, `${responder?.name ?? "玩家"} 接受报价并完成交易`);
      next.negotiation = null;
    }
  } else if (action.kind === "trade-reject") {
    next.rejectedTradeProposalKeys = [
      ...next.rejectedTradeProposalKeys,
      JSON.stringify(next.negotiation.draft),
    ];
    const responder = next.game.players.find(
      (player) => player.id === next.negotiation.responderId,
    );
    prependLog(next.game, `${responder?.name ?? "玩家"} 拒绝报价`);
    next.negotiation = null;
    result = { ok: true, reason: null, game: next.game };
  } else if (action.kind === "bankrupt") {
    result = declareBankruptcy(next.game, next.game.debt?.debtorId ?? next.game.activePlayerId, {
      mortgageValues: MANAGEMENT_RULES.mortgageValues,
      buildingCosts: Object.fromEntries(
        TILES.filter((tile) => tile.houseCost).map((tile) => [tile.id, tile.houseCost]),
      ),
    });
    if (result.ok && result.auctionTileIds?.length && !result.game.gameOver) {
      next.auction = openAuction(result.game);
    }
  } else if (action.kind === "draw-card") {
    next = drawPendingCards(next);
    result = { ok: true, reason: null, game: next.game };
  }

  if (!result?.ok) {
    return {
      ok: false,
      reason: result?.reason ?? "action-failed",
      session: structuredClone(session),
    };
  }
  next.game = result.game;
  next = drawPendingCards(next);
  if (record) next.actions = [...next.actions, structuredClone(action)];
  return { ok: true, reason: null, session: next };
}

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value).sort().map((key) => [key, canonical(value[key])]),
    );
  }
  return value;
}

function sameValue(left, right) {
  return JSON.stringify(canonical(left)) === JSON.stringify(canonical(right));
}

export function writeStandaloneSession(session, { space = 0 } = {}) {
  return JSON.stringify(session, null, space);
}

export function readStandaloneSession(value) {
  const parsed = typeof value === "string" ? JSON.parse(value) : structuredClone(value);
  if (!parsed || typeof parsed !== "object") throw new Error("Invalid standalone session");
  if (parsed.version === 1 || parsed.version === 3) {
    const legacyGame = parsed.game;
    if (!legacyGame?.players?.length) throw new Error("Invalid legacy session");
    const migrated = createStandaloneSession({
      mode: parsed.mode === "spectate" ? "spectate" : "play",
      playerCount: legacyGame.players.length,
      seed: parsed.runtime?.seed ?? 1,
      policy: parsed.policy ?? {},
      initialGame: legacyGame,
    });
    migrated.historyComplete = false;
    return migrated;
  }
  if (
    parsed.version !== STANDALONE_SESSION_VERSION ||
    !["play", "spectate"].includes(parsed.mode) ||
    !Array.isArray(parsed.actions) ||
    !parsed.initialGame?.players?.length ||
    !Number.isInteger(parsed.runtime?.seed)
  ) {
    throw new Error("Invalid standalone session");
  }
  const replayed = replayStandaloneGame({
    mode: parsed.mode,
    playerCount: parsed.initialGame.players.length,
    seed: parsed.runtime.seed,
    policy: parsed.policy,
    initialGame: parsed.initialGame,
    actions: parsed.actions,
  });
  if (!sameValue(replayed, parsed)) {
    throw new Error("Standalone session does not match replay");
  }
  return replayed;
}

export function replayStandaloneGame({
  mode,
  playerCount,
  seed,
  policy,
  initialGame,
  actions,
}) {
  let session = createStandaloneSession({ mode, playerCount, seed, policy, initialGame });
  for (const action of actions) {
    const result = applyStandaloneAction(session, action, { record: true });
    if (!result.ok) throw new Error(`Replay failed at action ${session.actions.length}: ${result.reason}`);
    session = result.session;
  }
  return session;
}
