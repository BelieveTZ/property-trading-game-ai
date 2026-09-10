/**
 * Rules shared by the simulator UI, AI recommendations, and behavior tests.
 * Inputs are plain game-state records so this module remains independent of React.
 */

export function groupPropertiesFor(tileId, properties, tiles) {
  const tile = tiles[tileId];
  if (!tile?.group) return [];
  return properties.filter(
    (property) => tiles[property.tileId]?.group === tile.group,
  );
}

export function buildingSupply(properties) {
  const housesUsed = properties.reduce(
    (total, property) =>
      total + (property.houses > 0 && property.houses < 5 ? property.houses : 0),
    0,
  );
  const hotelsUsed = properties.filter((property) => property.houses === 5).length;
  return {
    houses: Math.max(0, 32 - housesUsed),
    hotels: Math.max(0, 12 - hotelsUsed),
  };
}

export function canBuildOn({
  tileId,
  playerId,
  cash,
  properties,
  tiles,
  groupSizes,
}) {
  const tile = tiles[tileId];
  const property = properties.find((item) => item.tileId === tileId);
  if (!tile || tile.type !== "property" || !tile.group || !property) {
    return false;
  }
  const group = groupPropertiesFor(tileId, properties, tiles);
  if (
    group.length !== groupSizes[tile.group] ||
    group.some((item) => item.ownerId !== playerId || item.mortgaged)
  ) {
    return false;
  }
  const evenlyBuildable =
    property.houses < 5 &&
    property.houses === Math.min(...group.map((item) => item.houses)) &&
    cash >= (tile.houseCost ?? 0);
  if (!evenlyBuildable) return false;
  const supply = buildingSupply(properties);
  return property.houses === 4 ? supply.hotels > 0 : supply.houses > 0;
}

export function canSellFrom({ tileId, playerId, properties, tiles }) {
  const property = properties.find((item) => item.tileId === tileId);
  if (!property || property.ownerId !== playerId || property.houses <= 0) {
    return false;
  }
  const group = groupPropertiesFor(tileId, properties, tiles);
  if (property.houses !== Math.max(...group.map((item) => item.houses))) {
    return false;
  }
  return property.houses !== 5 || buildingSupply(properties).houses >= 4;
}

export function gameOutcome(game) {
  const survivors = game.players.filter((player) => !player.bankrupt);
  return survivors.length === 1
    ? { gameOver: true, winnerId: survivors[0].id }
    : { gameOver: false, winnerId: null };
}

function finalizeGameIfOver(game) {
  const outcome = gameOutcome(game);
  game.gameOver = outcome.gameOver;
  game.winnerId = outcome.winnerId;
  if (!outcome.gameOver) return game;
  const winner = game.players.find((player) => player.id === outcome.winnerId);
  game.activePlayerId = outcome.winnerId;
  game.pending = null;
  game.debt = null;
  game.paymentQueue = [];
  game.auctionQueue = [];
  game.lastDice = null;
  game.doublesStreak = 0;
  game.extraTurnEligible = false;
  game.tradeLockPlayerId = null;
  game.tradeLockedTileIds = [];
  if (!game.log?.some((entry) => entry === `${winner?.name ?? "玩家"} 赢得牌局`)) {
    appendTransitionLog(game, `${winner?.name ?? "玩家"} 赢得牌局`);
  }
  return game;
}

export function canMortgage({ tileId, playerId, properties, tiles }) {
  const property = properties.find((item) => item.tileId === tileId);
  if (
    !property ||
    property.ownerId !== playerId ||
    property.mortgaged ||
    property.houses > 0
  ) {
    return false;
  }
  return !groupPropertiesFor(tileId, properties, tiles).some(
    (item) => item.houses > 0,
  );
}

/**
 * Applies an ordered set of mortgage and building decisions to one game snapshot.
 * Each outcome is observable through this interface; callers do not reproduce the
 * ordering, cash, even-building, or debt rules.
 */
export function applyManagementPlan(
  game,
  actions,
  { tiles = {}, groupSizes = {}, mortgageValues = {} } = {},
) {
  let next = structuredClone(game);
  const outcomes = [];

  for (const action of actions) {
    const property = next.properties.find(
      (item) => item.tileId === action.tileId,
    );
    const owner = next.players.find(
      (player) => player.id === property?.ownerId && !player.bankrupt,
    );
    const tile = tiles[action.tileId];
    if (!property || !owner || !tile) {
      outcomes.push({ ...action, ok: false, reason: "invalid-deed" });
      continue;
    }

    if (action.kind === "mortgage") {
      if (
        groupPropertiesFor(action.tileId, next.properties, tiles).some(
          (item) => item.houses > 0,
        )
      ) {
        outcomes.push({ ...action, ok: false, reason: "group-has-buildings" });
        continue;
      }
      if (
        !canMortgage({
          tileId: action.tileId,
          playerId: owner.id,
          properties: next.properties,
          tiles,
        })
      ) {
        outcomes.push({ ...action, ok: false, reason: "illegal-mortgage" });
        continue;
      }
      const amount = mortgageValues[action.tileId] ?? 0;
      owner.cash += amount;
      property.mortgaged = true;
      outcomes.push({ ...action, ok: true, reason: null, playerId: owner.id, amount });
      continue;
    }

    if (action.kind === "unmortgage") {
      const amount = Math.ceil((mortgageValues[action.tileId] ?? 0) * 1.1);
      if (!property.mortgaged) {
        outcomes.push({ ...action, ok: false, reason: "not-mortgaged" });
      } else if (owner.cash < amount) {
        outcomes.push({ ...action, ok: false, reason: "insufficient-cash" });
      } else {
        owner.cash -= amount;
        property.mortgaged = false;
        outcomes.push({ ...action, ok: true, reason: null, playerId: owner.id, amount });
      }
      continue;
    }

    if (action.kind === "build") {
      if (
        !canBuildOn({
          tileId: action.tileId,
          playerId: owner.id,
          cash: owner.cash,
          properties: next.properties,
          tiles,
          groupSizes,
        })
      ) {
        outcomes.push({ ...action, ok: false, reason: "illegal-build" });
        continue;
      }
      const amount = tile.houseCost ?? 0;
      owner.cash -= amount;
      property.houses += 1;
      outcomes.push({
        ...action,
        ok: true,
        reason: null,
        playerId: owner.id,
        amount,
        houses: property.houses,
      });
      continue;
    }

    if (action.kind === "sell-building") {
      if (
        !canSellFrom({
          tileId: action.tileId,
          playerId: owner.id,
          properties: next.properties,
          tiles,
        })
      ) {
        outcomes.push({ ...action, ok: false, reason: "illegal-sale" });
        continue;
      }
      const amount = Math.round((tile.houseCost ?? 0) / 2);
      owner.cash += amount;
      property.houses -= 1;
      outcomes.push({
        ...action,
        ok: true,
        reason: null,
        playerId: owner.id,
        amount,
        houses: property.houses,
      });
      continue;
    }

    outcomes.push({ ...action, ok: false, reason: "unknown-action" });
  }

  next = clearPaidDebt(next);
  return { game: next, outcomes };
}

export function canPayJailFee(cash) {
  return Number.isFinite(cash) && cash >= 50;
}

export function resolveRollRules({ player, dieOne, dieTwo, doublesStreak }) {
  const nextPlayer = structuredClone(player);
  const isDouble = dieOne === dieTwo;
  const wasInJail = nextPlayer.inJail;

  if (wasInJail) {
    if (isDouble) {
      nextPlayer.inJail = false;
      nextPlayer.jailTurns = 0;
      return {
        player: nextPlayer,
        wasInJail,
        isDouble,
        shouldMove: true,
        jailFee: 0,
        doublesStreak: 0,
        extraTurnEligible: false,
        outcome: "released-by-doubles",
      };
    }
    const failedAttempts = nextPlayer.jailTurns + 1;
    if (failedAttempts < 3) {
      nextPlayer.jailTurns = failedAttempts;
      return {
        player: nextPlayer,
        wasInJail,
        isDouble,
        shouldMove: false,
        jailFee: 0,
        doublesStreak: 0,
        extraTurnEligible: false,
        outcome: "stays-in-jail",
      };
    }
    nextPlayer.inJail = false;
    nextPlayer.jailTurns = 0;
    return {
      player: nextPlayer,
      wasInJail,
      isDouble,
      shouldMove: true,
      jailFee: 50,
      doublesStreak: 0,
      extraTurnEligible: false,
      outcome: "released-by-fee",
    };
  }

  const nextDoublesStreak = isDouble ? doublesStreak + 1 : 0;
  if (nextDoublesStreak >= 3) {
    nextPlayer.position = 10;
    nextPlayer.inJail = true;
    nextPlayer.jailTurns = 0;
    return {
      player: nextPlayer,
      wasInJail,
      isDouble,
      shouldMove: false,
      jailFee: 0,
      doublesStreak: 0,
      extraTurnEligible: false,
      outcome: "triple-doubles-jail",
    };
  }
  return {
    player: nextPlayer,
    wasInJail,
    isDouble,
    shouldMove: true,
    jailFee: 0,
    doublesStreak: nextDoublesStreak,
    extraTurnEligible: isDouble,
    outcome: "move",
  };
}

function appendTransitionLog(game, message) {
  game.log = [message, ...(Array.isArray(game.log) ? game.log : [])].slice(0, 60);
}

/**
 * @param {any} game
 * @param {number} playerId
 * @param {{ tiles?: any[], rentFor?: (details: any) => number }} [options]
 */
export function resolveLanding(
  game,
  playerId,
  { tiles, rentFor = () => Number(0) } = {},
) {
  const next = structuredClone(game);
  const player = next.players.find((candidate) => candidate.id === playerId);
  const tile = player ? tiles?.[player.position] : null;
  if (!player || !tile) {
    return { ok: false, reason: "invalid-landing", game: next };
  }
  const property = next.properties.find((item) => item.tileId === tile.id);
  if (tile.type === "gotojail") {
    player.position = 10;
    player.inJail = true;
    player.jailTurns = 0;
    next.extraTurnEligible = false;
    next.pending = { kind: "notice", label: `${player.name} 被送往监狱` };
    appendTransitionLog(next, `${player.name} 到达「${tile.name}」，移动至监狱`);
  } else if (property && property.ownerId === null) {
    next.pending = { kind: "property", tileId: tile.id };
    appendTransitionLog(next, `${player.name} 到达未售出的「${tile.name}」`);
  } else if (property && property.ownerId !== player.id && !property.mortgaged) {
    const owner = next.players.find((candidate) => candidate.id === property.ownerId);
    if (!owner) return { ok: false, reason: "invalid-owner", game: next };
    const ownerProperties = next.properties.filter((item) => item.ownerId === owner.id);
    const diceTotal = next.lastDice ? next.lastDice[0] + next.lastDice[1] : 0;
    const amount = rentFor({ tile, property, ownerProperties, diceTotal });
    next.pending = { kind: "rent", tileId: tile.id, ownerId: owner.id, amount };
    appendTransitionLog(next, `${player.name} 到达 ${owner.name} 的「${tile.name}」`);
  } else if (tile.type === "tax") {
    const amount = tile.id === 38 ? 100 : 200;
    next.pending = { kind: "tax", amount, label: tile.name };
    appendTransitionLog(next, `${player.name} 需要支付 $${amount} ${tile.name}`);
  } else if (tile.type === "chance" || tile.type === "community") {
    next.pending = { kind: "card", deck: tile.type };
    appendTransitionLog(next, `${player.name} 到达「${tile.name}」，等待录入实体牌面`);
  } else {
    next.pending = { kind: "notice", label: `${player.name} 到达「${tile.name}」` };
    appendTransitionLog(next, `${player.name} 到达「${tile.name}」`);
  }
  return { ok: true, reason: null, game: next };
}

export function recordDiceRoll(game, { dieOne, dieTwo }, context = {}) {
  if (game.gameOver) {
    return { ok: false, reason: "game-over", game: structuredClone(game) };
  }
  if (game.pending) {
    return { ok: false, reason: "pending-decision", game: structuredClone(game) };
  }
  const next = structuredClone(game);
  const player = next.players.find((item) => item.id === next.activePlayerId);
  if (!player) {
    return { ok: false, reason: "unknown-active-player", game: next };
  }
  next.lastDice = [dieOne, dieTwo];
  const roll = resolveRollRules({
    player,
    dieOne,
    dieTwo,
    doublesStreak: next.doublesStreak,
  });
  Object.assign(player, roll.player);
  next.doublesStreak = roll.doublesStreak;
  next.extraTurnEligible = roll.extraTurnEligible;
  if (roll.outcome === "stays-in-jail") {
    next.pending = {
      kind: "notice",
      label: `${player.name} 未掷出双数，仍需停留（${player.jailTurns}/3）`,
    };
    appendTransitionLog(next, `${player.name} 本轮未能离开监狱`);
    return { ok: true, reason: null, game: next };
  }
  if (roll.outcome === "triple-doubles-jail") {
    next.pending = { kind: "notice", label: "连续三次双数，前往监狱" };
    appendTransitionLog(next, `${player.name} 连续三次掷出双数，被送往监狱`);
    return { ok: true, reason: null, game: next };
  }
  if (roll.outcome === "released-by-doubles") {
    appendTransitionLog(next, `${player.name} 掷出双数，离开监狱`);
  } else if (roll.jailFee > 0) {
    player.cash -= roll.jailFee;
    appendTransitionLog(next, `${player.name} 支付 $50 后离开监狱`);
  }
  const oldPosition = player.position;
  player.position = (player.position + dieOne + dieTwo) % 40;
  if (player.position < oldPosition) {
    player.cash += 200;
    appendTransitionLog(next, `${player.name} 经过出发点，领取 $200`);
  }
  appendTransitionLog(next, `${player.name} 掷出 ${dieOne} + ${dieTwo}，前进 ${dieOne + dieTwo} 格`);
  return resolveLanding(next, player.id, context);
}

/**
 * @param {any} game
 * @param {{ deck: string, card: any, diceTotal?: number }} input
 * @param {{ tiles?: any[], rentFor?: (details: any) => number }} [context]
 */
export function applyRecordedCard(
  game,
  { deck, card, diceTotal = 7 },
  { tiles, rentFor = () => Number(0) } = {},
) {
  const next = structuredClone(game);
  const pending = next.pending;
  const player = next.players.find(
    (candidate) => candidate.id === next.activePlayerId,
  );
  if (
    pending?.kind !== "card" ||
    pending.deck !== deck ||
    !player ||
    !card?.effect
  ) {
    return { ok: false, reason: "invalid-card-state", game: next };
  }
  const effect = card.effect;
  if (
    effect.kind === "nearest" &&
    effect.target === "utility" &&
    (!Number.isFinite(diceTotal) || diceTotal < 2 || diceTotal > 12)
  ) {
    return { ok: false, reason: "invalid-utility-roll", game: next };
  }

  next.pending = null;
  appendTransitionLog(next, `${player.name} 录入${deck === "chance" ? "机会" : "社会基金"}牌：${card.label}`);
  const awardGo = () => {
    player.cash += 200;
    appendTransitionLog(next, `${player.name} 经过起点，领取 $200`);
  };
  const moveTo = (destination, collectGo) => {
    const oldPosition = player.position;
    if (collectGo && (destination < oldPosition || destination === 0)) awardGo();
    player.position = destination;
    appendTransitionLog(next, `${player.name} 按牌面移动至「${tiles[destination].name}」`);
    return resolveLanding(next, player.id, { tiles, rentFor }).game;
  };

  if (effect.kind === "cash") {
    if (effect.amount < 0) {
      const paid = recordPayment(next, {
        debtorId: player.id,
        creditorId: null,
        amount: -effect.amount,
        reason: card.label,
      });
      appendTransitionLog(paid, `${player.name} 支付 $${-effect.amount}`);
      return { ok: true, reason: null, game: paid };
    }
    player.cash += effect.amount;
    appendTransitionLog(next, `${player.name}领取 $${effect.amount}`);
  } else if (effect.kind === "move") {
    return {
      ok: true,
      reason: null,
      game: moveTo(effect.destination, effect.collectGo),
    };
  } else if (effect.kind === "nearest") {
    const destinations = effect.target === "station" ? [5, 15, 25, 35] : [12, 28];
    const oldPosition = player.position;
    const destination = destinations.find((tileId) => tileId > oldPosition) ?? destinations[0];
    if (destination < oldPosition) awardGo();
    player.position = destination;
    const tile = tiles[destination];
    const property = next.properties.find((item) => item.tileId === destination);
    if (!tile || !property) {
      return { ok: false, reason: "invalid-card-destination", game: structuredClone(game) };
    }
    appendTransitionLog(next, `${player.name} 按牌面移动至最近的「${tile.name}」`);
    if (property.ownerId === null) {
      next.pending = { kind: "property", tileId: destination };
    } else if (property.ownerId !== player.id && !property.mortgaged) {
      const owner = next.players.find((candidate) => candidate.id === property.ownerId);
      if (!owner) return { ok: false, reason: "invalid-owner", game: structuredClone(game) };
      const ownerProperties = next.properties.filter((item) => item.ownerId === owner.id);
      const amount =
        effect.target === "station"
          ? rentFor({ tile, property, ownerProperties, diceTotal }) * 2
          : diceTotal * 10;
      next.pending = { kind: "rent", tileId: destination, ownerId: owner.id, amount };
    } else {
      next.pending = {
        kind: "notice",
        label:
          property.ownerId === player.id
            ? `「${tile.name}」属于自己，无需支付`
            : `「${tile.name}」已抵押，无需支付`,
      };
    }
  } else if (effect.kind === "back") {
    player.position = (player.position - effect.spaces + 40) % 40;
    appendTransitionLog(next, `${player.name} 后退 ${effect.spaces} 格至「${tiles[player.position].name}」`);
    return {
      ok: true,
      reason: null,
      game: resolveLanding(next, player.id, { tiles, rentFor }).game,
    };
  } else if (effect.kind === "jail") {
    player.position = 10;
    player.inJail = true;
    player.jailTurns = 0;
    next.doublesStreak = 0;
    next.extraTurnEligible = false;
    next.pending = { kind: "notice", label: `${player.name} 按牌面即时入狱` };
  } else if (effect.kind === "repairs") {
    const owned = next.properties.filter((property) => property.ownerId === player.id);
    const houses = owned.reduce(
      (total, property) =>
        total + (property.houses > 0 && property.houses < 5 ? property.houses : 0),
      0,
    );
    const hotels = owned.filter((property) => property.houses === 5).length;
    const amount = houses * effect.perHouse + hotels * effect.perHotel;
    const paid = recordPayment(next, {
      debtorId: player.id,
      creditorId: null,
      amount,
      reason: card.label,
    });
    appendTransitionLog(paid, `${player.name} 为 ${houses} 栋房屋和 ${hotels} 家酒店支付维修费 $${amount}`);
    return { ok: true, reason: null, game: paid };
  } else if (effect.kind === "payEach" || effect.kind === "collectEach") {
    const opponents = next.players.filter(
      (candidate) => candidate.id !== player.id && !candidate.bankrupt,
    );
    const payments = opponents.map((opponent) =>
      effect.kind === "payEach"
        ? { debtorId: player.id, creditorId: opponent.id, amount: effect.amount, reason: card.label }
        : { debtorId: opponent.id, creditorId: player.id, amount: effect.amount, reason: card.label },
    );
    const paid = queuePayments(next, payments);
    appendTransitionLog(
      paid,
      effect.kind === "payEach"
        ? `${player.name} 向 ${opponents.length} 位玩家共支付 $${opponents.length * effect.amount}`
        : `${player.name} 从 ${opponents.length} 位玩家共收取 $${opponents.length * effect.amount}`,
    );
    return { ok: true, reason: null, game: paid };
  } else if (effect.kind === "jailFree") {
    player.jailFreeCards += 1;
    appendTransitionLog(next, `${player.name} 获得一张监狱通行证，当前持有 ${player.jailFreeCards} 张`);
  } else {
    return { ok: false, reason: "unknown-card-effect", game: structuredClone(game) };
  }

  return { ok: true, reason: null, game: next };
}

export function resolvePendingDecision(game, action, { tiles } = {}) {
  const next = structuredClone(game);
  const pending = next.pending;
  const player = next.players.find(
    (candidate) => candidate.id === next.activePlayerId,
  );
  if (!pending || !player) {
    return { ok: false, reason: "no-pending-decision", game: next };
  }

  if (pending.kind === "property") {
    const tile = tiles?.[pending.tileId];
    const property = next.properties.find(
      (candidate) => candidate.tileId === pending.tileId,
    );
    if (!tile || !property || property.ownerId !== null) {
      return { ok: false, reason: "invalid-property", game: next };
    }
    if (action === "skip") {
      return {
        ok: true,
        reason: null,
        game: next,
        auction: {
          winnerId: next.players.find((candidate) => !candidate.bankrupt)?.id ?? 0,
          price: Math.max(10, Math.round((tile.price ?? 60) * 0.6)),
        },
      };
    }
    if (action !== "buy") {
      return { ok: false, reason: "invalid-property-action", game: next };
    }
    if (player.cash < (tile.price ?? 0)) {
      return { ok: false, reason: "insufficient-cash", game: next };
    }
    player.cash -= tile.price ?? 0;
    property.ownerId = player.id;
    appendTransitionLog(next, `${player.name} 以 $${tile.price ?? 0} 买入「${tile.name}」`);
  } else if (pending.kind === "rent") {
    const owner = next.players.find(
      (candidate) => candidate.id === pending.ownerId,
    );
    const tile = tiles?.[pending.tileId];
    if (!owner || !tile) {
      return { ok: false, reason: "invalid-rent", game: next };
    }
    const paid = recordPayment(next, {
      debtorId: player.id,
      creditorId: owner.id,
      amount: pending.amount,
      reason: `租金：${tile.name}`,
    });
    paid.pending = null;
    appendTransitionLog(paid, `${player.name} 向 ${owner.name} 支付租金 $${pending.amount}`);
    return { ok: true, reason: null, game: paid };
  } else if (pending.kind === "tax") {
    const paid = recordPayment(next, {
      debtorId: player.id,
      creditorId: null,
      amount: pending.amount,
      reason: pending.label,
    });
    paid.pending = null;
    appendTransitionLog(paid, `${player.name} 支付 $${pending.amount}「${pending.label}」`);
    return { ok: true, reason: null, game: paid };
  }

  next.pending = null;
  return { ok: true, reason: null, game: next };
}

export function useJailFreeCard(game, playerId) {
  const next = structuredClone(game);
  const player = next.players.find((candidate) => candidate.id === playerId);
  if (!player?.inJail || player.jailFreeCards < 1) {
    return { ok: false, reason: "jail-card-unavailable", game: next };
  }
  player.jailFreeCards -= 1;
  player.inJail = false;
  player.jailTurns = 0;
  appendTransitionLog(next, `${player.name} 使用监狱通行证，剩余 ${player.jailFreeCards} 张`);
  return { ok: true, reason: null, game: next };
}

export function payJailFee(game, playerId) {
  const next = structuredClone(game);
  const player = next.players.find((candidate) => candidate.id === playerId);
  if (!player?.inJail || !canPayJailFee(player.cash)) {
    return { ok: false, reason: "jail-fee-unavailable", game: next };
  }
  player.cash -= 50;
  player.inJail = false;
  player.jailTurns = 0;
  appendTransitionLog(next, `${player.name} 支付 $50 离开监狱`);
  return { ok: true, reason: null, game: next };
}

export function addGamePlayer(game, player, maximumPlayers = 5) {
  const next = structuredClone(game);
  if (next.gameOver) {
    return { ok: false, reason: "game-over", game: next };
  }
  if (
    next.players.length >= maximumPlayers ||
    next.players.some((candidate) => candidate.id === player.id)
  ) {
    return { ok: false, reason: "player-limit-or-duplicate", game: next };
  }
  next.players.push(structuredClone(player));
  next.players.sort((left, right) => left.id - right.id);
  appendTransitionLog(next, `${player.name} 加入牌局`);
  return { ok: true, reason: null, game: next };
}

export function removeGamePlayer(game, playerId, minimumPlayers = 3) {
  const next = structuredClone(game);
  if (next.gameOver) {
    return { ok: false, reason: "game-over", game: next };
  }
  const originalPlayers = next.players;
  const removedIndex = originalPlayers.findIndex(
    (candidate) => candidate.id === playerId,
  );
  const removed = originalPlayers[removedIndex];
  if (!removed || next.players.length <= minimumPlayers) {
    return { ok: false, reason: "player-minimum-or-unknown", game: next };
  }
  next.players = next.players.filter((candidate) => candidate.id !== playerId);
  const removedActivePlayer = next.activePlayerId === playerId;
  let fallback =
    next.players.find((candidate) => !candidate.bankrupt) ?? next.players[0];
  let wrappedTurnOrder = false;
  if (removedActivePlayer) {
    for (let offset = 1; offset < originalPlayers.length; offset += 1) {
      const candidateIndex = (removedIndex + offset) % originalPlayers.length;
      const candidate = next.players.find(
        (player) =>
          player.id === originalPlayers[candidateIndex].id && !player.bankrupt,
      );
      if (!candidate) continue;
      fallback = candidate;
      wrappedTurnOrder = candidateIndex <= removedIndex;
      break;
    }
  }
  next.properties = next.properties.map((property) =>
    property.ownerId === playerId
      ? { ...property, ownerId: null, houses: 0, mortgaged: false }
      : property,
  );
  if (removedActivePlayer) {
    next.activePlayerId = fallback.id;
    if (wrappedTurnOrder && Number.isFinite(next.round)) next.round += 1;
    next.lastDice = null;
    next.doublesStreak = 0;
    next.extraTurnEligible = false;
    next.tradeLockPlayerId = null;
    next.tradeLockedTileIds = [];
  }
  if (next.myPlayerId === playerId) next.myPlayerId = fallback.id;
  if (
    (removedActivePlayer && next.pending?.kind !== "property") ||
    (next.pending?.kind === "rent" && next.pending.ownerId === playerId)
  ) {
    next.pending = null;
  }
  next.debt =
    next.debt?.debtorId === playerId
      ? null
      : next.debt?.creditorId === playerId
        ? { ...next.debt, creditorId: null }
        : next.debt;
  next.paymentQueue = (next.paymentQueue ?? []).filter(
    (payment) =>
      payment.debtorId !== playerId && payment.creditorId !== playerId,
  );
  appendTransitionLog(next, `${removed.name ?? "玩家"} 离开牌局，其资产已归还银行`);
  return {
    ok: true,
    reason: null,
    game: finalizeGameIfOver(next),
    fallbackId: fallback.id,
  };
}

export function canSettleAuction(winner, price) {
  return (
    !!winner &&
    !winner.bankrupt &&
    Number.isInteger(price) &&
    price > 0 &&
    winner.cash >= price
  );
}

export function canDeclareBankruptcy(game, playerId) {
  const player = game.players.find((item) => item.id === playerId);
  return (
    !!player &&
    !player.bankrupt &&
    (player.cash < 0 || game.debt?.debtorId === playerId)
  );
}

export function recordPayment(game, { debtorId, creditorId, amount, reason }) {
  const next = structuredClone(game);
  const debtor = next.players.find((player) => player.id === debtorId);
  const creditor = next.players.find((player) => player.id === creditorId);
  if (!debtor || !Number.isFinite(amount) || amount < 0) return next;
  debtor.cash -= amount;
  if (creditor) creditor.cash += amount;
  next.debt =
    debtor.cash < 0
      ? {
          debtorId,
          creditorId: creditor?.id ?? null,
          amount: -debtor.cash,
          reason,
        }
      : null;
  return next;
}

function drainPaymentQueue(game) {
  const next = structuredClone(game);
  next.paymentQueue = Array.isArray(next.paymentQueue) ? next.paymentQueue : [];
  if (next.debt) return next;

  while (next.paymentQueue.length) {
    const payment = next.paymentQueue.shift();
    const debtor = next.players.find(
      (player) => player.id === payment.debtorId && !player.bankrupt,
    );
    const creditor = next.players.find(
      (player) => player.id === payment.creditorId && !player.bankrupt,
    );
    if (
      !debtor ||
      !creditor ||
      debtor.id === creditor.id ||
      !Number.isFinite(payment.amount) ||
      payment.amount <= 0
    ) {
      continue;
    }
    debtor.cash -= payment.amount;
    creditor.cash += payment.amount;
    if (debtor.cash < 0) {
      next.debt = {
        debtorId: debtor.id,
        creditorId: creditor.id,
        amount: -debtor.cash,
        reason: payment.reason,
      };
      break;
    }
  }
  return next;
}

export function queuePayments(game, payments) {
  const next = structuredClone(game);
  next.paymentQueue = [
    ...(Array.isArray(next.paymentQueue) ? next.paymentQueue : []),
    ...payments,
  ];
  return drainPaymentQueue(next);
}

export function applyTrade(
  game,
  draft,
  { mortgageValues = {}, colorGroups = {} } = {},
) {
  const next = structuredClone(game);
  const from = next.players.find(
    (player) => player.id === draft.fromId && !player.bankrupt,
  );
  const to = next.players.find(
    (player) => player.id === draft.toId && !player.bankrupt,
  );
  if (!from || !to || from.id === to.id) {
    return { ok: false, reason: "invalid-participants", game: next };
  }
  const amounts = [
    draft.fromCash,
    draft.toCash,
    draft.fromCards,
    draft.toCards,
  ];
  const fromPropertyIds = Array.isArray(draft.fromPropertyIds)
    ? draft.fromPropertyIds
    : [];
  const toPropertyIds = Array.isArray(draft.toPropertyIds)
    ? draft.toPropertyIds
    : [];
  const selectedIds = [...fromPropertyIds, ...toPropertyIds];
  if (
    amounts.some((amount) => !Number.isInteger(amount) || amount < 0) ||
    new Set(selectedIds).size !== selectedIds.length ||
    (!amounts.some((amount) => amount > 0) && selectedIds.length === 0)
  ) {
    return { ok: false, reason: "invalid-contents", game: next };
  }
  const fromProperties = fromPropertyIds.map((tileId) =>
    next.properties.find((property) => property.tileId === tileId),
  );
  const toProperties = toPropertyIds.map((tileId) =>
    next.properties.find((property) => property.tileId === tileId),
  );
  if (
    fromProperties.some((property) => !property || property.ownerId !== from.id) ||
    toProperties.some((property) => !property || property.ownerId !== to.id)
  ) {
    return { ok: false, reason: "ownership-changed", game: next };
  }
  if (
    next.tradeLockPlayerId === next.activePlayerId &&
    selectedIds.some((tileId) => next.tradeLockedTileIds?.includes(tileId))
  ) {
    return { ok: false, reason: "same-turn-round-trip", game: next };
  }
  if (
    selectedIds.some((tileId) =>
      (colorGroups[tileId] ?? [tileId]).some((groupTileId) =>
        next.properties.some(
          (property) => property.tileId === groupTileId && property.houses > 0,
        ),
      ),
    )
  ) {
    return { ok: false, reason: "group-has-buildings", game: next };
  }
  if (from.jailFreeCards < draft.fromCards || to.jailFreeCards < draft.toCards) {
    return { ok: false, reason: "insufficient-cards", game: next };
  }
  const interestForFrom = toProperties.reduce(
    (total, property) =>
      total +
      (property.mortgaged
        ? Math.ceil((mortgageValues[property.tileId] ?? 0) * 0.1)
        : 0),
    0,
  );
  const interestForTo = fromProperties.reduce(
    (total, property) =>
      total +
      (property.mortgaged
        ? Math.ceil((mortgageValues[property.tileId] ?? 0) * 0.1)
        : 0),
    0,
  );
  if (from.cash - draft.fromCash + draft.toCash < interestForFrom) {
    return { ok: false, reason: "from-insufficient-cash", game: next };
  }
  if (to.cash - draft.toCash + draft.fromCash < interestForTo) {
    return { ok: false, reason: "to-insufficient-cash", game: next };
  }

  from.cash = from.cash - draft.fromCash + draft.toCash - interestForFrom;
  to.cash = to.cash - draft.toCash + draft.fromCash - interestForTo;
  from.jailFreeCards = from.jailFreeCards - draft.fromCards + draft.toCards;
  to.jailFreeCards = to.jailFreeCards - draft.toCards + draft.fromCards;
  next.properties = next.properties.map((property) => {
    if (fromPropertyIds.includes(property.tileId)) {
      return { ...property, ownerId: to.id };
    }
    if (toPropertyIds.includes(property.tileId)) {
      return { ...property, ownerId: from.id };
    }
    return property;
  });
  const existingLocks =
    next.tradeLockPlayerId === next.activePlayerId
      ? next.tradeLockedTileIds ?? []
      : [];
  next.tradeLockPlayerId = next.activePlayerId;
  next.tradeLockedTileIds = Array.from(new Set([...existingLocks, ...selectedIds]));
  return {
    ok: true,
    reason: null,
    game: clearPaidDebt(next),
    interestTotal: interestForFrom + interestForTo,
  };
}

export function canEndTurn(game, activePlayerId) {
  return (
    !game.gameOver &&
    !game.pending &&
    !game.debt &&
    game.activePlayerId === activePlayerId &&
    !game.players.some((player) => !player.bankrupt && player.cash < 0)
  );
}

export function settlePropertyAuction(game, bid, { tiles = {} } = {}) {
  const next = structuredClone(game);
  if (next.pending?.kind !== "property") {
    return { ok: false, reason: "no-property-auction", game: next };
  }
  const winner = next.players.find((player) => player.id === bid.winnerId);
  if (!canSettleAuction(winner, bid.price)) {
    return { ok: false, reason: "invalid-bid", game: next };
  }
  const tileId = next.pending.tileId;
  const property = next.properties.find((item) => item.tileId === tileId);
  if (!property || property.ownerId !== null) {
    return { ok: false, reason: "property-unavailable", game: next };
  }
  winner.cash -= bid.price;
  property.ownerId = winner.id;
  appendTransitionLog(
    next,
    `${winner.name} 以 $${Math.round(bid.price).toLocaleString("zh-CN")} 竞得「${tiles[tileId]?.name ?? `地块 ${tileId}`}」`,
  );
  const nextTileId = next.auctionQueue.shift();
  next.pending =
    nextTileId === undefined ? null : { kind: "property", tileId: nextTileId };
  return { ok: true, reason: null, game: next, nextTileId };
}

export function skipPropertyAuction(game) {
  const next = structuredClone(game);
  if (next.pending?.kind !== "property") {
    return { ok: false, reason: "no-property-auction", game: next };
  }
  const nextTileId = next.auctionQueue.shift();
  next.pending =
    nextTileId === undefined ? null : { kind: "property", tileId: nextTileId };
  return { ok: true, reason: null, game: next, nextTileId };
}

export function advanceTurn(game, activePlayerId) {
  const next = structuredClone(game);
  if (next.gameOver) {
    return { ok: false, reason: "game-over", game: next };
  }
  if (gameOutcome(next).gameOver) {
    return {
      ok: true,
      reason: null,
      game: finalizeGameIfOver(next),
      gameOver: true,
      winnerId: gameOutcome(next).winnerId,
    };
  }
  if (!canEndTurn(next, activePlayerId)) {
    return {
      ok: false,
      reason: next.debt || next.players.some((player) => !player.bankrupt && player.cash < 0)
        ? "unpaid-debt"
        : "pending-decision",
      game: next,
    };
  }
  const current = next.players.find((player) => player.id === next.activePlayerId);
  if (!current) return { ok: false, reason: "unknown-active-player", game: next };
  const extraTurn = next.extraTurnEligible && !current.inJail;
  if (extraTurn) {
    appendTransitionLog(next, `${current.name} 因掷出双数获得额外回合`);
  } else {
    const alive = next.players.filter((player) => !player.bankrupt);
    const index = alive.findIndex((player) => player.id === next.activePlayerId);
    const nextPlayer = alive[(index + 1) % alive.length];
    if (!nextPlayer) return { ok: false, reason: "no-active-player", game: next };
    if (nextPlayer.id <= next.activePlayerId) next.round += 1;
    next.activePlayerId = nextPlayer.id;
    next.tradeLockPlayerId = null;
    next.tradeLockedTileIds = [];
    next.doublesStreak = 0;
    appendTransitionLog(next, `轮到 ${nextPlayer.name}`);
  }
  next.extraTurnEligible = false;
  next.lastDice = null;
  return { ok: true, reason: null, game: next };
}

export function clearPaidDebt(game) {
  if (!game.debt) return drainPaymentQueue(game);
  const debtor = game.players.find(
    (player) => player.id === game.debt.debtorId,
  );
  if (!debtor) return game;
  if (debtor.cash < 0) {
    return {
      ...game,
      debt: { ...game.debt, amount: -debtor.cash },
    };
  }
  return drainPaymentQueue({ ...game, debt: null });
}

export function settleBankruptcy(
  game,
  playerId,
  { mortgageValues = {}, buildingCosts = {} } = {},
) {
  const next = structuredClone(game);
  if (next.gameOver) return { game: next, auctionTileIds: [] };
  const debtor = next.players.find((player) => player.id === playerId);
  if (!debtor || debtor.bankrupt) return { game: next, auctionTileIds: [] };
  const creditorId =
    next.debt?.debtorId === playerId ? next.debt.creditorId : null;
  const creditor = next.players.find(
    (player) => player.id === creditorId && !player.bankrupt,
  );
  const auctionTileIds = [];

  if (creditor) {
    creditor.cash += Math.min(0, debtor.cash);
    creditor.jailFreeCards =
      (creditor.jailFreeCards ?? 0) + (debtor.jailFreeCards ?? 0);
  }
  debtor.jailFreeCards = 0;

  for (const property of next.properties) {
    if (property.ownerId !== playerId) continue;
    const buildingCredit =
      property.houses * Math.floor((buildingCosts[property.tileId] ?? 0) / 2);
    if (creditor) {
      creditor.cash += buildingCredit;
      property.ownerId = creditor.id;
      if (property.mortgaged) {
        creditor.cash -= Math.ceil((mortgageValues[property.tileId] ?? 0) * 0.1);
      }
    } else {
      property.ownerId = null;
      property.mortgaged = false;
      auctionTileIds.push(property.tileId);
    }
    property.houses = 0;
  }

  debtor.cash = 0;
  debtor.bankrupt = true;
  if (next.activePlayerId === playerId) {
    const previousIndex = next.players.findIndex((player) => player.id === playerId);
    for (let offset = 1; offset < next.players.length; offset += 1) {
      const candidateIndex = (previousIndex + offset) % next.players.length;
      const candidate = next.players[candidateIndex];
      if (candidate.bankrupt) continue;
      next.activePlayerId = candidate.id;
      if (candidateIndex <= previousIndex && Number.isFinite(next.round)) {
        next.round += 1;
      }
      break;
    }
    next.lastDice = null;
    next.doublesStreak = 0;
    next.extraTurnEligible = false;
    next.tradeLockPlayerId = null;
    next.tradeLockedTileIds = [];
  }
  if (auctionTileIds.length) {
    next.pending = { kind: "property", tileId: auctionTileIds[0] };
    next.auctionQueue = auctionTileIds.slice(1);
  }
  next.debt =
    creditor && creditor.cash < 0
      ? {
          debtorId: creditor.id,
          creditorId: null,
          amount: -creditor.cash,
          reason: "mortgage-interest",
        }
      : null;
  const settled = next.debt ? next : drainPaymentQueue(next);
  const finished = finalizeGameIfOver(settled);
  return {
    game: finished,
    auctionTileIds: finished.gameOver ? [] : auctionTileIds,
  };
}

export function declareBankruptcy(game, playerId, options = {}) {
  if (game.gameOver) {
    return {
      ok: false,
      reason: "game-over",
      game: structuredClone(game),
      auctionTileIds: [],
    };
  }
  const player = game.players.find((candidate) => candidate.id === playerId);
  if (!player) {
    return {
      ok: false,
      reason: "unknown-player",
      game: structuredClone(game),
      auctionTileIds: [],
    };
  }
  if (player.bankrupt) {
    return {
      ok: false,
      reason: "already-bankrupt",
      game: structuredClone(game),
      auctionTileIds: [],
    };
  }
  if (!canDeclareBankruptcy(game, playerId)) {
    return {
      ok: false,
      reason: "player-solvent",
      game: structuredClone(game),
      auctionTileIds: [],
    };
  }
  return { ok: true, reason: null, ...settleBankruptcy(game, playerId, options) };
}
