import { TILES } from "./board-catalog.mjs";
import {
  createTradeDraft,
  legalManagementActions,
  legalStandaloneActions,
  planDebtLiquidation,
  recommendStandaloneAction,
  standaloneActorId,
} from "./standalone-session.mjs";

function automaticAuctionAction(session) {
  const bidder = session.game.players.find(
    (player) => player.id === session.auction.activeBidderId,
  );
  const tile = TILES[session.auction.tileId ?? -1];
  const ceiling = Math.min(
    bidder?.cash ?? 0,
    Math.round((tile?.price ?? 0) * 1.08),
  );
  const minimum = session.auction.currentBid + 1;
  const amount = Math.min(ceiling, Math.max(minimum, Math.ceil(minimum / 10) * 10));
  return amount >= minimum
    ? { kind: "auction-bid", amount }
    : { kind: "auction-pass" };
}

function tradeValue(session, draft, playerId) {
  const fromSide = draft.fromId === playerId;
  const givenCash = fromSide ? draft.fromCash : draft.toCash;
  const receivedCash = fromSide ? draft.toCash : draft.fromCash;
  const givenCards = fromSide ? draft.fromCards : draft.toCards;
  const receivedCards = fromSide ? draft.toCards : draft.fromCards;
  const givenIds = fromSide ? draft.fromPropertyIds : draft.toPropertyIds;
  const receivedIds = fromSide ? draft.toPropertyIds : draft.fromPropertyIds;
  const value = (ids) => ids.reduce(
    (total, tileId) => total + (TILES[tileId]?.price ?? 0),
    0,
  );
  return receivedCash + receivedCards * 50 + value(receivedIds)
    - givenCash - givenCards * 50 - value(givenIds);
}

function automaticNegotiationAction(session) {
  const negotiation = session.negotiation;
  const responder = session.game.players.find(
    (player) => player.id === negotiation?.responderId,
  );
  if (!negotiation || !responder) return { kind: "trade-reject" };
  const draft = negotiation.draft;
  const value = tradeValue(session, draft, responder.id);
  if (value >= -10) return { kind: "trade-accept" };
  if (negotiation.counterCount >= 3) return { kind: "trade-reject" };
  const counter = structuredClone(draft);
  if (responder.id === draft.fromId) {
    counter.fromCash = Math.max(0, draft.fromCash + value);
  } else {
    counter.toCash = Math.max(0, draft.toCash + value);
  }
  const hasContents = [
    counter.fromCash,
    counter.toCash,
    counter.fromCards,
    counter.toCards,
    counter.fromPropertyIds.length,
    counter.toPropertyIds.length,
  ].some((amount) => amount > 0);
  return hasContents
    ? { kind: "trade-counter", draft: counter }
    : { kind: "trade-reject" };
}

function automaticTradeDraft(session) {
  const player = session.game.players.find(
    (candidate) => candidate.id === session.game.activePlayerId,
  );
  if (
    !player ||
    session.tradeAttemptsThisTurn.includes(player.id) ||
    session.game.round % 6 !== player.id % 6
  ) {
    return null;
  }
  const deed = session.game.properties.find(
    (property) =>
      property.ownerId !== null &&
      property.ownerId !== player.id &&
      !property.mortgaged &&
      property.houses === 0 &&
      !session.game.tradeLockedTileIds.includes(property.tileId),
  );
  const seller = session.game.players.find(
    (candidate) => candidate.id === deed?.ownerId && !candidate.bankrupt,
  );
  if (!deed || !seller) return null;
  const price = Math.round((TILES[deed.tileId]?.price ?? 0) * 1.05 / 10) * 10;
  if (price <= 0 || player.cash - price < 220) return null;
  return {
    ...createTradeDraft(player.id, seller.id),
    fromCash: price,
    toPropertyIds: [deed.tileId],
  };
}

export function chooseAutomaticAction(session) {
  const legal = legalStandaloneActions(session);
  const actor = session.game.players.find(
    (candidate) => candidate.id === standaloneActorId(session),
  );
  if (legal.includes("trade-accept")) return automaticNegotiationAction(session);
  if (legal.includes("confirm")) return { kind: "confirm" };
  if (legal.includes("auction-bid")) return automaticAuctionAction(session);
  if (session.game.debt && legal.includes("manage")) {
    const actions = planDebtLiquidation(session);
    if (actions.length) return { kind: "manage", actions };
    return { kind: "bankrupt" };
  }
  if (legal.includes("buy")) {
    const tile = TILES[session.game.pending?.tileId ?? -1];
    return actor && actor.cash - (tile?.price ?? 0) >= 180
      ? { kind: "buy" }
      : { kind: "decline" };
  }
  if (legal.includes("decline")) return { kind: "decline" };
  if (legal.includes("use-pass")) return { kind: "use-pass" };
  if (legal.includes("pay-fee") && actor && actor.cash > 300) return { kind: "pay-fee" };
  if (legal.includes("roll")) return { kind: "roll" };
  if (legal.includes("manage")) {
    const management = legalManagementActions(session, session.game.activePlayerId);
    const build = management.find(
      (action) => action.kind === "build" && actor.cash - (TILES[action.tileId]?.houseCost ?? 0) >= 240,
    );
    if (build) return { kind: "manage", actions: [build] };
    const unmortgage = management.find(
      (action) => action.kind === "unmortgage" && actor.cash >= 420,
    );
    if (unmortgage) return { kind: "manage", actions: [unmortgage] };
  }
  if (legal.includes("trade-propose")) {
    const draft = automaticTradeDraft(session);
    if (draft) return { kind: "trade-propose", draft };
  }
  if (legal.includes("end-turn")) return { kind: "end-turn" };
  if (legal.includes("bankrupt")) return { kind: "bankrupt" };
  return recommendStandaloneAction(session);
}
