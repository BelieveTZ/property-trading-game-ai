import {
  DEED_DETAILS,
  GROUP_SIZES,
  PROPERTY_GROUP_ORDER,
  TILES,
} from "./board-catalog.mjs";
import { canBuildOn, canMortgage, canSellFrom } from "./game-rules.mjs";
import {
  LEAGUE_OUTPUT,
  encodeLeagueFeatures,
} from "../shared/league-model.mjs";


const MODEL_MAX_HIDDEN = 6;


export function selectPlayerCountModel(models, playerCount) {
  if (playerCount === 3) return models[3];
  if (playerCount === 5) return models[5];
  return models[4];
}


export function trainedPlayerCount(model) {
  return model.playerCount ?? 4;
}


export function evaluateModel(model, features, output) {
  let value = model.outputBias[output];
  const outputOffset = output * MODEL_MAX_HIDDEN;
  for (let hidden = 0; hidden < model.hiddenCount; hidden++) {
    let sum = model.hiddenBias[hidden];
    const inputOffset = hidden * model.inputCount;
    for (let feature = 0; feature < model.inputCount; feature++) {
      sum += model.inputHidden[inputOffset + feature] * (features[feature] ?? 0);
    }
    value += Math.tanh(sum) * model.hiddenOutput[outputOffset + hidden];
  }
  return Math.tanh(value);
}


export function auctionCeiling(
  model,
  features,
  { cash, price, completesSet, reserve },
) {
  const value = (evaluateModel(model, features, LEAGUE_OUTPUT.auctionBid) + 1) * 0.5;
  let multiplier = 0.28 + value * 1.35;
  if (completesSet) multiplier += 0.32;
  return Math.max(0, Math.min(cash - reserve, Math.floor(price * multiplier)));
}


export function tradeProposalKey(draft) {
  const fromProperties = [...draft.fromPropertyIds].sort((a, b) => a - b);
  const toProperties = [...draft.toPropertyIds].sort((a, b) => a - b);
  return [
    draft.fromId,
    draft.toId,
    Math.max(0, Math.floor(draft.fromCash)),
    Math.max(0, Math.floor(draft.toCash)),
    Math.max(0, Math.floor(draft.fromCards)),
    Math.max(0, Math.floor(draft.toCards)),
    fromProperties.join("."),
    toProperties.join("."),
  ].join("|");
}


function mortgageValue(tile) {
  return DEED_DETAILS[tile.id]?.mortgage ?? Math.round((tile.price ?? 0) / 2);
}


export function tradeAssetValue(property, playerId, properties, receiving) {
  const tile = TILES[property.tileId];
  const base = property.mortgaged
    ? mortgageValue(tile)
    : tile.price ?? mortgageValue(tile);
  const incomeValue = property.mortgaged ? 0 : (tile.rent ?? 0) * 4;
  if (!tile.group) return base + incomeValue;
  const ownedCount = properties.filter(
    (item) => item.ownerId === playerId && TILES[item.tileId].group === tile.group,
  ).length;
  const size = GROUP_SIZES[tile.group] ?? 1;
  const afterCount = Math.max(0, Math.min(size, ownedCount + (receiving ? 1 : -1)));
  const groupBonus = (count) =>
    base * 0.35 * (count / size) + (count === size ? base * 0.9 : 0);
  const beforeBonus = groupBonus(ownedCount);
  const afterBonus = groupBonus(afterCount);
  const marginalBonus = receiving
    ? afterBonus - beforeBonus
    : beforeBonus - afterBonus;
  return Math.round(base + incomeValue + marginalBonus);
}


export function playerNetWorth(player, properties) {
  const assets = properties
    .filter((property) => property.ownerId === player.id)
    .reduce((total, property) => {
      const tile = TILES[property.tileId];
      return (
        total +
        (tile.price ?? 0) * (property.mortgaged ? 0.5 : 1) +
        property.houses * (tile.houseCost ?? 0) * 0.5
      );
    }, 0);
  return Math.round(player.cash + assets);
}


function decisionFeatures(game, player, tile, postCash, groupProgress, completesSet) {
  const opponents = game.players.filter(
    (candidate) => candidate.id !== player.id && !candidate.bankrupt,
  );
  const opponentAverage =
    opponents.reduce(
      (total, candidate) => total + playerNetWorth(candidate, game.properties),
      0,
    ) / Math.max(1, opponents.length);
  return encodeLeagueFeatures({
    cash: player.cash,
    postCash,
    price: tile.price ?? 0,
    outcomeValue: tile.rent ?? 0,
    groupProgress,
    completesSet,
    turnProgress: game.round / 30,
    advantage: playerNetWorth(player, game.properties) - opponentAverage,
  });
}


function propertyDecisionContext(game, player, tile) {
  const groupSize = GROUP_SIZES[tile.group ?? ""] ?? 1;
  const ownedInGroup = game.properties.filter(
    (property) =>
      property.ownerId === player.id &&
      TILES[property.tileId].group === tile.group,
  ).length;
  return {
    groupSize,
    ownedInGroup,
    completesSet: ownedInGroup + 1 === groupSize,
  };
}


export function recommendPurchase({ game, model, policy, player, tile }) {
  const postCash = player.cash - (tile.price ?? 0);
  const { groupSize, ownedInGroup, completesSet } = propertyDecisionContext(
    game,
    player,
    tile,
  );
  const features = decisionFeatures(
    game,
    player,
    tile,
    postCash,
    ownedInGroup / groupSize,
    completesSet,
  );
  const localScore =
    policy.liquidity * (postCash / 500) +
    policy.roi * (((tile.rent ?? 0) * 18) / (tile.price ?? 1)) +
    policy.monopoly * Number(completesSet) -
    Math.max(0, policy.reserve - postCash) / 180;
  const neuralScore = evaluateModel(model, features, LEAGUE_OUTPUT.buy);
  const trainedReserve =
    75 +
    ((evaluateModel(model, features, LEAGUE_OUTPUT.cashReserve) + 1) * 0.5) *
      500;
  const effectiveReserve = Math.round(
    trainedReserve * 0.85 + policy.reserve * 0.15,
  );
  const score = neuralScore * 0.85 + Math.tanh(localScore) * 0.15;
  const maximumBid = auctionCeiling(model, features, {
    cash: player.cash,
    price: tile.price ?? 0,
    completesSet,
    reserve: effectiveReserve,
  });
  const buy =
    (score > -0.05 || maximumBid >= (tile.price ?? 0)) &&
    postCash >= effectiveReserve * 0.38 &&
    postCash >= 0;
  return { buy, score, maximumBid, effectiveReserve };
}


export function recommendAuction({
  game,
  model,
  policy,
  player,
  tile,
  currentBid,
}) {
  if (player.bankrupt) return { participate: false, maximumBid: 0 };
  const { groupSize, ownedInGroup, completesSet } = propertyDecisionContext(
    game,
    player,
    tile,
  );
  const features = decisionFeatures(
    game,
    player,
    tile,
    player.cash - (tile.price ?? 0),
    ownedInGroup / groupSize,
    completesSet,
  );
  const trainedReserve =
    75 +
    ((evaluateModel(model, features, LEAGUE_OUTPUT.cashReserve) + 1) * 0.5) *
      500;
  const effectiveReserve = Math.round(
    trainedReserve * 0.85 + policy.reserve * 0.15,
  );
  const valueCeiling = auctionCeiling(model, features, {
    cash: player.cash,
    price: tile.price ?? 0,
    completesSet,
    reserve: effectiveReserve,
  });
  const maximumBid =
    Math.floor(Math.max(0, Math.min(player.cash, valueCeiling)) / 10) * 10;
  return {
    participate: maximumBid >= 10 && currentBid <= maximumBid,
    maximumBid,
  };
}


export function recommendJail({ game, model, player }) {
  const opponents = game.players.filter(
    (candidate) => candidate.id !== player.id && !candidate.bankrupt,
  );
  const opponentAverageCash =
    opponents.reduce((total, candidate) => total + candidate.cash, 0) /
    Math.max(1, opponents.length);
  const features = encodeLeagueFeatures({
    cash: player.cash,
    postCash: player.cash - 50,
    price: 50,
    outcomeValue: 0,
    groupProgress: 0,
    completesSet: false,
    turnProgress: game.round / 30,
    advantage: player.cash - opponentAverageCash,
  });
  return {
    payToLeave:
      Number.isFinite(player.cash) &&
      player.cash >= 50 &&
      evaluateModel(model, features, LEAGUE_OUTPUT.leaveJail) > 0,
  };
}


export function recommendBuild({ game, model, player }) {
  let remainingCash = player.cash;
  const plannedProperties = structuredClone(game.properties);
  const opponents = game.players.filter(
    (candidate) => candidate.id !== player.id && !candidate.bankrupt,
  );
  const opponentAverageCash =
    opponents.reduce((total, candidate) => total + candidate.cash, 0) /
    Math.max(1, opponents.length);
  const tileIds = [];

  for (const group of PROPERTY_GROUP_ORDER) {
    const groupProperties = plannedProperties.filter(
      (property) =>
        property.ownerId === player.id &&
        TILES[property.tileId]?.group === group,
    );
    if (groupProperties.length !== GROUP_SIZES[group]) continue;

    const candidate = groupProperties
      .filter((property) =>
        canBuildOn({
          tileId: property.tileId,
          playerId: player.id,
          cash: remainingCash,
          properties: plannedProperties,
          tiles: TILES,
          groupSizes: GROUP_SIZES,
        }),
      )
      .sort((a, b) => a.tileId - b.tileId)[0];
    if (!candidate) continue;

    const tile = TILES[candidate.tileId];
    const deed = DEED_DETAILS[candidate.tileId];
    if (!deed || deed.kind !== "property") continue;
    const cost = tile.houseCost ?? 0;
    if (remainingCash < cost) continue;
    const rentIncrease =
      deed.rents[candidate.houses + 1] - deed.rents[candidate.houses];
    const features = encodeLeagueFeatures({
      cash: remainingCash,
      postCash: remainingCash - cost,
      price: cost,
      outcomeValue: rentIncrease,
      groupProgress: 1,
      completesSet: true,
      turnProgress: game.round / 30,
      advantage: remainingCash - opponentAverageCash,
    });
    const reserve =
      75 +
      ((evaluateModel(model, features, LEAGUE_OUTPUT.cashReserve) + 1) * 0.5) *
        500;
    if (
      remainingCash - cost < reserve ||
      evaluateModel(model, features, LEAGUE_OUTPUT.build) <= -0.08
    ) {
      continue;
    }
    tileIds.push(candidate.tileId);
    remainingCash -= cost;
    candidate.houses += 1;
  }

  return { tileIds, remainingCash };
}


export function recommendPayment({ game, player, amount }) {
  if (player.cash >= amount) return { action: "pay" };
  const canRaiseFunds = game.properties
    .filter((property) => property.ownerId === player.id)
    .some(
      (property) =>
        canMortgage({
          tileId: property.tileId,
          playerId: player.id,
          properties: game.properties,
          tiles: TILES,
        }) ||
        canSellFrom({
          tileId: property.tileId,
          playerId: player.id,
          properties: game.properties,
          tiles: TILES,
        }),
    );
  return { action: canRaiseFunds ? "fundraise" : "declare-bankruptcy" };
}


export function tradeDecisionWindow(game, player) {
  if (game.lastDice === null) return player.inJail ? -0.75 : -1;
  return game.pending ? 0.5 : 1;
}


export function evaluateTrade(game, model, player, offer) {
  const {
    propertyIdsGiven,
    propertyIdsReceived,
    cashGiven,
    cashReceived,
    cardsGiven = 0,
    cardsReceived = 0,
    decisionWindow = tradeDecisionWindow(game, player),
  } = offer;
  const propertyMap = new Map(
    game.properties.map((property) => [property.tileId, property]),
  );
  const givenProperties = propertyIdsGiven
    .map((tileId) => propertyMap.get(tileId))
    .filter(Boolean);
  const receivedProperties = propertyIdsReceived
    .map((tileId) => propertyMap.get(tileId))
    .filter(Boolean);
  const assetValueGiven = givenProperties.reduce(
    (total, property) => total + tradeAssetValue(property, player.id, game.properties, false),
    0,
  );
  const assetValueReceived = receivedProperties.reduce(
    (total, property) => total + tradeAssetValue(property, player.id, game.properties, true),
    0,
  );
  const mortgageInterest = receivedProperties.reduce(
    (total, property) =>
      property.mortgaged
        ? total + Math.ceil(mortgageValue(TILES[property.tileId]) * 0.1)
        : total,
    0,
  );
  const valueGiven = assetValueGiven + cashGiven + cardsGiven * 50;
  const valueReceived = assetValueReceived + cashReceived + cardsReceived * 50;
  const valueDelta = valueReceived - valueGiven - mortgageInterest;
  const postCash = player.cash - cashGiven + cashReceived - mortgageInterest;
  const afterOwnedIds = new Set(
    game.properties
      .filter(
        (property) =>
          property.ownerId === player.id &&
          !propertyIdsGiven.includes(property.tileId),
      )
      .map((property) => property.tileId),
  );
  for (const tileId of propertyIdsReceived) afterOwnedIds.add(tileId);
  let bestProgress = 0;
  let completesSet = 0;
  for (const tileId of propertyIdsReceived) {
    const tile = TILES[tileId];
    if (!tile.group) continue;
    const size = GROUP_SIZES[tile.group] ?? 1;
    const count = Array.from(afterOwnedIds).filter(
      (ownedId) => TILES[ownedId].group === tile.group,
    ).length;
    bestProgress = Math.max(bestProgress, count / size);
    if (count === size) completesSet = 1;
  }
  const opponents = game.players.filter(
    (candidate) => candidate.id !== player.id && !candidate.bankrupt,
  );
  const opponentAverage =
    opponents.reduce(
      (total, candidate) => total + playerNetWorth(candidate, game.properties),
      0,
    ) / Math.max(1, opponents.length);
  const features = encodeLeagueFeatures({
    cash: player.cash,
    postCash,
    price: (valueGiven + valueReceived) / 2,
    outcomeValue: valueDelta / 2,
    groupProgress: bestProgress,
    completesSet,
    turnProgress: game.round / 30,
    advantage: playerNetWorth(player, game.properties) - opponentAverage,
    tradeWindow: decisionWindow,
  });
  const trainedForTrades =
    model.outputCount > 5 && model.actionNames?.includes("trade");
  const neuralScore = trainedForTrades
    ? evaluateModel(model, features, LEAGUE_OUTPUT.trade)
    : -1;
  const trainedReserve =
    75 +
    ((evaluateModel(model, features, LEAGUE_OUTPUT.cashReserve) + 1) * 0.5) *
      500;
  const tolerance = Math.max(20, valueReceived * 0.08);
  return {
    accepted:
      Boolean(trainedForTrades) &&
      postCash >= 0 &&
      valueDelta >= -tolerance &&
      neuralScore > -0.08,
    neuralScore,
    valueDelta,
    postCash,
    trainedReserve,
  };
}


function groupHasBuildings(tile, properties) {
  if (tile.type !== "property" || !tile.group) return false;
  return properties.some(
    (property) =>
      TILES[property.tileId]?.group === tile.group && property.houses > 0,
  );
}


function tradeDraft(fromId, toId) {
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


function money(value) {
  return `¤${Math.round(value).toLocaleString("zh-CN")}`;
}


/**
 * @param {{ game: any, model: any, player: any, rejectedKeys?: string[] }} input
 */
export function recommendTradeProposal({
  game,
  model,
  player,
  rejectedKeys = [],
}) {
  if (
    player.bankrupt ||
    game.activePlayerId !== player.id ||
    model.outputCount <= LEAGUE_OUTPUT.trade ||
    !model.actionNames?.includes("trade")
  ) {
    return null;
  }

  const candidates = [];
  const opponents = game.players.filter(
    (candidate) => candidate.id !== player.id && !candidate.bankrupt,
  );
  const eligibleProperties = game.properties.filter(
    (property) =>
      property.ownerId !== null &&
      !property.mortgaged &&
      !(
        game.tradeLockPlayerId === game.activePlayerId &&
        game.tradeLockedTileIds.includes(property.tileId)
      ) &&
      !groupHasBuildings(TILES[property.tileId], game.properties),
  );
  const decisionWindow = tradeDecisionWindow(game, player);

  const considerCashForDeed = (property, buyer, seller) => {
    const buyerValue = tradeAssetValue(
      property,
      buyer.id,
      game.properties,
      true,
    );
    const sellerValue = tradeAssetValue(
      property,
      seller.id,
      game.properties,
      false,
    );
    if (buyerValue < sellerValue) return;

    const priceFloor = Math.max(10, Math.ceil(sellerValue / 10) * 10);
    const priceCeiling = Math.floor(buyerValue / 10) * 10;
    if (priceCeiling < priceFloor) return;
    const priceSpan = priceCeiling - priceFloor;
    const prices = Array.from(
      new Set(
        [0, 0.25, 0.5, 0.75, 1].map(
          (ratio) => Math.round((priceFloor + priceSpan * ratio) / 10) * 10,
        ),
      ),
    );

    for (const price of prices) {
      if (buyer.cash < price) continue;
      const playerBuys = player.id === buyer.id;
      const evaluation = evaluateTrade(game, model, player, {
        propertyIdsGiven: playerBuys ? [] : [property.tileId],
        propertyIdsReceived: playerBuys ? [property.tileId] : [],
        cashGiven: playerBuys ? price : 0,
        cashReceived: playerBuys ? 0 : price,
        cardsGiven: 0,
        cardsReceived: 0,
        decisionWindow,
      });
      if (
        !evaluation.accepted ||
        (playerBuys &&
          evaluation.postCash <
            Math.max(50, evaluation.trainedReserve * 0.45))
      ) {
        continue;
      }

      const score =
        evaluation.neuralScore * 2 +
        Math.tanh(evaluation.valueDelta / 180) * 0.75;
      const tile = TILES[property.tileId];
      const counterpart = playerBuys ? seller : buyer;
      const draft = {
        ...tradeDraft(player.id, counterpart.id),
        fromCash: playerBuys ? price : 0,
        toCash: playerBuys ? 0 : price,
        fromPropertyIds: playerBuys ? [] : [property.tileId],
        toPropertyIds: playerBuys ? [property.tileId] : [],
      };
      candidates.push({
        key: tradeProposalKey(draft),
        draft,
        summary: playerBuys
          ? `建议向${counterpart.name}支付 ${money(price)}，换取${tile.short}`
          : `建议将${tile.short}以 ${money(price)} 出售给${counterpart.name}`,
        score,
      });
    }
  };

  for (const property of eligibleProperties) {
    const owner = game.players.find(
      (candidate) => candidate.id === property.ownerId,
    );
    if (!owner || owner.bankrupt) continue;
    if (owner.id === player.id) {
      for (const opponent of opponents) {
        considerCashForDeed(property, opponent, player);
      }
    } else {
      considerCashForDeed(property, player, owner);
    }
  }

  return (
    candidates
      .filter((candidate) => !rejectedKeys.includes(candidate.key))
      .sort((a, b) => b.score - a.score)[0] ?? null
  );
}
