import { availableParallelism } from "node:os";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { appendFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { isMainThread, parentPort, Worker, workerData } from "node:worker_threads";
import { BOARD_SIZE, indexedBoardRules } from "../shared/board-rules.mjs";
import {
  LEAGUE_ACTION_NAMES,
  LEAGUE_INPUT_COUNT,
  LEAGUE_OUTPUT,
  encodeLeagueFeatures,
} from "../shared/league-model.mjs";

const INPUTS = LEAGUE_INPUT_COUNT;
const MAX_HIDDEN = 6;
const OUTPUTS = LEAGUE_ACTION_NAMES.length;
const boardRules = indexedBoardRules();
const PRICES = boardRules.prices;
const BASE_RENTS = boardRules.baseRents;
const HOUSE_COSTS = boardRules.buildCosts;
const GROUPS = boardRules.groupIndexes;
const GROUP_SIZES = boardRules.groupSizes;
const TILE_TYPES = [
  0, 1, 3, 1, 2, 1, 1, 3, 1, 1, 0, 1, 1, 1, 1, 1, 1, 3, 1, 1, 0, 1, 3, 1,
  1, 1, 1, 1, 1, 1, 4, 1, 1, 3, 1, 1, 3, 1, 2, 1,
];
const PROPERTY_RENTS = boardRules.propertyRents;
const CARD_AMOUNTS = [100, -50, 75, -100, 150, 25, -15, 50];

class FastRandom {
  constructor(seed) {
    this.state = seed >>> 0 || 0x9e3779b9;
    this.hasGaussian = false;
    this.gaussian = 0;
  }

  nextUint() {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state;
  }

  nextInt(maxExclusive) {
    return maxExclusive <= 1 ? 0 : this.nextUint() % maxExclusive;
  }

  nextDouble() {
    return this.nextUint() / 4294967296;
  }

  nextGaussian() {
    if (this.hasGaussian) {
      this.hasGaussian = false;
      return this.gaussian;
    }
    const u1 = Math.max(1e-12, this.nextDouble());
    const u2 = this.nextDouble();
    const radius = Math.sqrt(-2 * Math.log(u1));
    const angle = Math.PI * 2 * u2;
    this.gaussian = radius * Math.sin(angle);
    this.hasGaussian = true;
    return radius * Math.cos(angle);
  }
}

function mixSeed(...values) {
  let seed = 0x6d2b79f5;
  for (const value of values) {
    seed ^= (Number(value) >>> 0) + 0x9e3779b9 + ((seed << 6) >>> 0) + (seed >>> 2);
    seed = Math.imul(seed ^ (seed >>> 16), 0x21f0aaad);
    seed = Math.imul(seed ^ (seed >>> 15), 0x735a2d97);
    seed ^= seed >>> 15;
  }
  return seed >>> 0 || 1;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function cloneGenome(genome) {
  return {
    hiddenCount: genome.hiddenCount,
    inputHidden: genome.inputHidden.slice(),
    hiddenBias: genome.hiddenBias.slice(),
    hiddenOutput: genome.hiddenOutput.slice(),
    outputBias: genome.outputBias.slice(),
  };
}

function evaluate(genome, input, output) {
  let value = genome.outputBias[output];
  const outputOffset = output * MAX_HIDDEN;
  for (let hidden = 0; hidden < genome.hiddenCount; hidden++) {
    let sum = genome.hiddenBias[hidden];
    const inputOffset = hidden * INPUTS;
    for (let feature = 0; feature < INPUTS; feature++) {
      sum += genome.inputHidden[inputOffset + feature] * input[feature];
    }
    value += Math.tanh(sum) * genome.hiddenOutput[outputOffset + hidden];
  }
  return Math.tanh(value);
}

function crossover(a, b, random) {
  const child = cloneGenome(a);
  child.hiddenCount = random.nextDouble() < 0.5 ? a.hiddenCount : b.hiddenCount;
  mixArray(child.inputHidden, b.inputHidden, random);
  mixArray(child.hiddenBias, b.hiddenBias, random);
  mixArray(child.hiddenOutput, b.hiddenOutput, random);
  mixArray(child.outputBias, b.outputBias, random);
  return child;
}

function mixArray(target, other, random) {
  for (let i = 0; i < target.length; i++) {
    if (random.nextDouble() < 0.5) target[i] = other[i];
  }
}

function mutate(genome, random, sigma) {
  mutateArray(genome.inputHidden, random, sigma, 0.18);
  mutateArray(genome.hiddenBias, random, sigma, 0.22);
  mutateArray(genome.hiddenOutput, random, sigma, 0.2);
  mutateArray(genome.outputBias, random, sigma, 0.2);
  if (random.nextDouble() < 0.06) {
    genome.hiddenCount = Math.min(MAX_HIDDEN, genome.hiddenCount + 1);
  } else if (random.nextDouble() < 0.025) {
    genome.hiddenCount = Math.max(2, genome.hiddenCount - 1);
  }
}

function mutateArray(values, random, sigma, probability) {
  for (let i = 0; i < values.length; i++) {
    if (random.nextDouble() < probability) {
      values[i] = clamp(values[i] + random.nextGaussian() * sigma, -5, 5);
    }
    if (random.nextDouble() < 0.004) {
      values[i] = random.nextGaussian() * 0.8;
    }
  }
}

function createEmptyGenome() {
  return {
    hiddenCount: 6,
    inputHidden: Array(MAX_HIDDEN * INPUTS).fill(0),
    hiddenBias: Array(MAX_HIDDEN).fill(0),
    hiddenOutput: Array(OUTPUTS * MAX_HIDDEN).fill(0),
    outputBias: Array(OUTPUTS).fill(0),
  };
}

function createHeuristic() {
  const genome = createEmptyGenome();
  setHidden(genome, 0, [-0.15, 1.35, -0.65, 1.05, 0.8, 1.9, 0.05, -0.15], -0.22);
  setHidden(genome, 1, [0.3, 0.8, -0.3, 1.25, 0.95, 1.5, 0.08, -0.15], -0.12);
  setHidden(genome, 2, [0.75, 1.1, -0.7, 1.35, 0.3, 1.8, 0.4, -0.25], -0.18);
  setHidden(genome, 3, [0.45, 0.25, -0.1, 0.05, 0, 0, 1.1, 0.2], -0.65);
  setHidden(genome, 4, [-0.2, -0.55, 0.4, -0.1, -0.15, -0.2, -0.2, 0.55], 0.05);
  setHidden(
    genome,
    5,
    [0.15, 0.85, -0.35, 1.55, 1.15, 1.8, 0.35, -0.35, 0.35],
    -0.12,
  );
  for (let output = 0; output < OUTPUTS; output++) {
    genome.hiddenOutput[output * MAX_HIDDEN + output] = 2.35;
  }
  genome.outputBias[LEAGUE_OUTPUT.buy] = -0.08;
  genome.outputBias[LEAGUE_OUTPUT.auctionBid] = -0.12;
  genome.outputBias[LEAGUE_OUTPUT.build] = -0.18;
  genome.outputBias[LEAGUE_OUTPUT.leaveJail] = -0.1;
  genome.outputBias[LEAGUE_OUTPUT.cashReserve] = 0.05;
  genome.outputBias[LEAGUE_OUTPUT.trade] = -0.12;
  return genome;
}

function setHidden(genome, hidden, weights, bias) {
  for (let i = 0; i < INPUTS; i++) {
    genome.inputHidden[hidden * INPUTS + i] = weights[i] ?? 0;
  }
  genome.hiddenBias[hidden] = bias;
}

function shiftStyle(genome, reserveShift, aggressionShift) {
  genome.outputBias[LEAGUE_OUTPUT.cashReserve] += reserveShift;
  genome.outputBias[LEAGUE_OUTPUT.buy] += aggressionShift;
  genome.outputBias[LEAGUE_OUTPUT.auctionBid] += aggressionShift;
  genome.outputBias[LEAGUE_OUTPUT.build] += aggressionShift * 0.7;
  genome.outputBias[LEAGUE_OUTPUT.trade] += aggressionShift * 0.45;
}

function brainFor(brains, seat) {
  return brains[seat];
}

function fillFeatures(
  features,
  player,
  cash,
  alive,
  postCash,
  price,
  rent,
  progress,
  completes,
  turn,
  maxTurns,
  tradeWindow = 0,
) {
  let opponentTotal = 0;
  let opponentCount = 0;
  for (let other = 0; other < alive.length; other++) {
    if (other === player || !alive[other]) continue;
    opponentTotal += cash[other];
    opponentCount++;
  }
  const opponentAverage = opponentCount ? opponentTotal / opponentCount : cash[player];
  encodeLeagueFeatures(
    {
      cash: cash[player],
      postCash,
      price,
      outcomeValue: rent,
      groupProgress: progress,
      completesSet: completes,
      turnProgress: turn / Math.max(1, maxTurns),
      advantage: cash[player] - opponentAverage,
      tradeWindow,
    },
    features,
  );
}

function cashReserve(brain, features) {
  return 75 + Math.floor(
    ((evaluate(brain, features, LEAGUE_OUTPUT.cashReserve) + 1) * 0.5) * 500,
  );
}

function auctionBidCeiling(brain, features, cash, price, completes, reserve) {
  const value =
    (evaluate(brain, features, LEAGUE_OUTPUT.auctionBid) + 1) * 0.5;
  let multiplier = 0.28 + value * 1.35;
  if (completes) multiplier += 0.32;
  return Math.max(
    0,
    Math.min(cash - reserve, Math.floor(price * multiplier)),
  );
}

function wantsToBuy(state, player, tile, brain, turn, maxTurns) {
  const { cash, alive, groupCounts, features } = state;
  const price = PRICES[tile];
  if (cash[player] < price) return false;
  const group = GROUPS[tile];
  const count = groupCounts[player * 10 + group];
  fillFeatures(
    features,
    player,
    cash,
    alive,
    cash[player] - price,
    price,
    BASE_RENTS[tile],
    count / GROUP_SIZES[group],
    count + 1 === GROUP_SIZES[group] ? 1 : 0,
    turn,
    maxTurns,
  );
  const reserve = cashReserve(brain, features);
  const completes = count + 1 === GROUP_SIZES[group];
  const wouldBidAtLeastFaceValue =
    auctionBidCeiling(
      brain,
      features,
      cash[player],
      price,
      completes,
      reserve,
    ) >= price;
  return (
    (evaluate(brain, features, LEAGUE_OUTPUT.buy) > -0.05 || wouldBidAtLeastFaceValue) &&
    cash[player] - price >= reserve * 0.38
  );
}

function auction(state, tile, brains, turn, maxTurns) {
  const { cash, alive, owners, groupCounts, features } = state;
  const bids = new Int32Array(brains.length);
  let best = -1;
  let bestBid = 0;
  let secondBid = 0;
  const group = GROUPS[tile];
  for (let player = 0; player < brains.length; player++) {
    if (!alive[player]) continue;
    const brain = brainFor(brains, player);
    const count = groupCounts[player * 10 + group];
    fillFeatures(
      features,
      player,
      cash,
      alive,
      cash[player] - PRICES[tile],
      PRICES[tile],
      BASE_RENTS[tile],
      count / GROUP_SIZES[group],
      count + 1 === GROUP_SIZES[group] ? 1 : 0,
      turn,
      maxTurns,
    );
    const reserve = cashReserve(brain, features);
    bids[player] = auctionBidCeiling(
      brain,
      features,
      cash[player],
      PRICES[tile],
      count + 1 === GROUP_SIZES[group],
      reserve,
    );
    if (bids[player] > bestBid) {
      secondBid = bestBid;
      bestBid = bids[player];
      best = player;
    } else if (bids[player] > secondBid) {
      secondBid = bids[player];
    }
  }
  if (best >= 0 && bestBid >= 10) {
    const paid = Math.min(bestBid, Math.max(10, secondBid + 10));
    cash[best] -= paid;
    owners[tile] = best;
    groupCounts[best * 10 + group]++;
  }
}

function tryBuild(state, player, brain, turn, maxTurns) {
  const { cash, alive, owners, houses, mortgaged, groupCounts, features } = state;
  let housesAvailable = 32;
  let hotelsAvailable = 12;
  for (const buildings of houses) {
    if (buildings === 5) hotelsAvailable--;
    else if (buildings > 0) housesAvailable -= buildings;
  }
  for (let group = 0; group < 8; group++) {
    if (groupCounts[player * 10 + group] !== GROUP_SIZES[group]) continue;
    let candidate = -1;
    let minimumHouses = 6;
    for (let tile = 0; tile < BOARD_SIZE; tile++) {
      if (
        GROUPS[tile] !== group ||
        owners[tile] !== player ||
        mortgaged[tile] ||
        houses[tile] >= minimumHouses ||
        houses[tile] >= 5
      ) {
        continue;
      }
      minimumHouses = houses[tile];
      candidate = tile;
    }
    if (candidate < 0) continue;
    if (houses[candidate] === 4 ? hotelsAvailable < 1 : housesAvailable < 1) {
      continue;
    }
    const cost = HOUSE_COSTS[candidate];
    if (cash[player] < cost) continue;
    const rents = PROPERTY_RENTS[candidate];
    const currentRent = rents[houses[candidate]];
    const nextRent = rents[houses[candidate] + 1];
    fillFeatures(
      features,
      player,
      cash,
      alive,
      cash[player] - cost,
      cost,
      nextRent - currentRent,
      1,
      1,
      turn,
      maxTurns,
    );
    const reserve = cashReserve(brain, features);
    if (
      cash[player] - cost >= reserve &&
      evaluate(brain, features, LEAGUE_OUTPUT.build) > -0.08
    ) {
      cash[player] -= cost;
      if (houses[candidate] === 4) {
        hotelsAvailable--;
        housesAvailable += 4;
      } else {
        housesAvailable--;
      }
      houses[candidate]++;
    }
  }
}

function calculateRent(state, tile, owner, roll) {
  const { houses, groupCounts } = state;
  const group = GROUPS[tile];
  if (group === 8) {
    const count = groupCounts[owner * 10 + group];
    return 25 << Math.max(0, count - 1);
  }
  if (group === 9) {
    const count = groupCounts[owner * 10 + group];
    return roll * (count === 2 ? 10 : 4);
  }
  if (houses[tile] > 0) {
    return PROPERTY_RENTS[tile][houses[tile]];
  }
  const ownsSet = groupCounts[owner * 10 + group] === GROUP_SIZES[group];
  return BASE_RENTS[tile] * (ownsSet ? 2 : 1);
}

function tradePropertyValue(state, player, tile, acquiring) {
  const group = GROUPS[tile];
  const count = state.groupCounts[player * 10 + group];
  const size = GROUP_SIZES[group];
  const afterCount = clamp(count + (acquiring ? 1 : -1), 0, size);
  const groupBonus = (groupCount) =>
    PRICES[tile] * 0.35 * (groupCount / size) +
    (groupCount === size ? PRICES[tile] * 0.9 : 0);
  const beforeBonus = groupBonus(count);
  const afterBonus = groupBonus(afterCount);
  const marginalBonus = acquiring
    ? afterBonus - beforeBonus
    : beforeBonus - afterBonus;
  return Math.round(
    PRICES[tile] +
      marginalBonus +
      BASE_RENTS[tile] * 4,
  );
}

function tryTrade(
  state,
  buyer,
  brains,
  random,
  turn,
  maxTurns,
  tradeWindow,
  tradedTiles,
  sellerFilter = -1,
) {
  const buyerBrain = brainFor(brains, buyer);
  const candidates = [];
  const blockedGroups = new Uint8Array(8);
  for (let tile = 0; tile < BOARD_SIZE; tile++) {
    const group = GROUPS[tile];
    if (group >= 0 && group < 8 && state.houses[tile] > 0) {
      blockedGroups[group] = 1;
    }
  }
  for (let tile = 0; tile < BOARD_SIZE; tile++) {
    const seller = state.owners[tile];
    const group = GROUPS[tile];
    if (
      tradedTiles[tile] ||
      seller < 0 ||
      seller === buyer ||
      (sellerFilter >= 0 && seller !== sellerFilter) ||
      !state.alive[seller] ||
      state.mortgaged[tile] ||
      (group < 8 && blockedGroups[group])
    ) {
      continue;
    }
    const buyerValue = tradePropertyValue(state, buyer, tile, true);
    const sellerValue = tradePropertyValue(state, seller, tile, false);
    const completes =
      state.groupCounts[buyer * 10 + group] + 1 === GROUP_SIZES[group] ? 1 : 0;
    const progress =
      (state.groupCounts[buyer * 10 + group] + 1) / GROUP_SIZES[group];
    candidates.push({
      tile,
      seller,
      buyerValue,
      sellerValue,
      completes,
      progress,
      priority: buyerValue - sellerValue + completes * 220 + random.nextInt(41),
    });
  }
  candidates.sort((a, b) => b.priority - a.priority);
  for (const candidate of candidates.slice(0, 5)) {
    const midpoint = (candidate.buyerValue + candidate.sellerValue) / 2;
    const price = Math.max(
      10,
      Math.round((midpoint * (0.94 + random.nextDouble() * 0.12)) / 10) * 10,
    );
    if (state.cash[buyer] < price) continue;
    fillFeatures(
      state.features,
      buyer,
      state.cash,
      state.alive,
      state.cash[buyer] - price,
      price,
      candidate.buyerValue - price,
      candidate.progress,
      candidate.completes,
      turn,
      maxTurns,
      tradeWindow,
    );
    const reserve = cashReserve(buyerBrain, state.features);
    const buyerAccept =
      state.cash[buyer] - price >= reserve * 0.45 &&
      evaluate(buyerBrain, state.features, LEAGUE_OUTPUT.trade) > -0.08;
    if (!buyerAccept) continue;

    const sellerBrain = brainFor(brains, candidate.seller);
    const sellerGroup = GROUPS[candidate.tile];
    const sellerAfterCount =
      state.groupCounts[candidate.seller * 10 + sellerGroup] - 1;
    fillFeatures(
      state.features,
      candidate.seller,
      state.cash,
      state.alive,
      state.cash[candidate.seller] + price,
      price,
      price - candidate.sellerValue,
      sellerAfterCount / GROUP_SIZES[sellerGroup],
      0,
      turn,
      maxTurns,
      tradeWindow,
    );
    const sellerAccept =
      evaluate(sellerBrain, state.features, LEAGUE_OUTPUT.trade) > -0.08;
    if (!sellerAccept) continue;

    state.cash[buyer] -= price;
    state.cash[candidate.seller] += price;
    state.owners[candidate.tile] = buyer;
    state.groupCounts[candidate.seller * 10 + sellerGroup]--;
    state.groupCounts[buyer * 10 + sellerGroup]++;
    return candidate.tile;
  }
  return -1;
}

function tryTradesAtWindow(
  state,
  initiator,
  brains,
  random,
  turn,
  maxTurns,
  tradeWindow,
  tradedTiles,
) {
  if (!state.alive[initiator]) return 0;
  let completedTrades = 0;
  const maximumTrades = brains.length;
  while (completedTrades < maximumTrades) {
    const tryBuying = () =>
      tryTrade(
        state,
        initiator,
        brains,
        random,
        turn,
        maxTurns,
        tradeWindow,
        tradedTiles,
      );
    const trySelling = () => {
      const firstBuyer = random.nextInt(brains.length);
      for (let offset = 0; offset < brains.length; offset++) {
        const buyer = (firstBuyer + offset) % brains.length;
        if (buyer === initiator || !state.alive[buyer]) continue;
        return tryTrade(
          state,
          buyer,
          brains,
          random,
          turn,
          maxTurns,
          tradeWindow,
          tradedTiles,
          initiator,
        );
      }
      return -1;
    };
    const buyFirst = random.nextDouble() < 0.5;
    let tradedTile = buyFirst ? tryBuying() : trySelling();
    if (tradedTile < 0) {
      tradedTile = buyFirst ? trySelling() : tryBuying();
    }
    if (tradedTile < 0) break;
    tradedTiles[tradedTile] = 1;
    completedTrades++;
  }
  return completedTrades;
}

function resolveDebt(state, debtor, creditor) {
  const { cash, alive, owners, houses, mortgaged, groupCounts } = state;
  if (cash[debtor] >= 0 || !alive[debtor]) return;
  for (let tile = BOARD_SIZE - 1; tile >= 0 && cash[debtor] < 0; tile--) {
    if (owners[tile] !== debtor || houses[tile] > 0 || mortgaged[tile]) continue;
    mortgaged[tile] = 1;
    cash[debtor] += Math.floor(PRICES[tile] / 2);
  }
  if (cash[debtor] >= 0) return;
  alive[debtor] = 0;
  state.aliveCount--;
  for (let tile = 0; tile < BOARD_SIZE; tile++) {
    if (owners[tile] !== debtor) continue;
    const group = GROUPS[tile];
    groupCounts[debtor * 10 + group]--;
    if (creditor >= 0 && alive[creditor]) {
      owners[tile] = creditor;
      groupCounts[creditor * 10 + group]++;
    } else {
      owners[tile] = -1;
      houses[tile] = 0;
      mortgaged[tile] = 0;
    }
  }
}

function netWorth(state, player) {
  let total = state.cash[player];
  for (let tile = 0; tile < BOARD_SIZE; tile++) {
    if (state.owners[tile] !== player) continue;
    total += state.mortgaged[tile] ? Math.floor(PRICES[tile] / 2) : PRICES[tile];
    total += Math.floor((state.houses[tile] * HOUSE_COSTS[tile]) / 2);
  }
  return total;
}

function playGame(brains, focusSeat, seed, maxTurns) {
  const playerCount = brains.length;
  const state = {
    cash: new Int32Array(playerCount).fill(1500),
    positions: new Int8Array(playerCount),
    alive: new Uint8Array(playerCount).fill(1),
    jailTurns: new Uint8Array(playerCount),
    doubles: new Uint8Array(playerCount),
    owners: new Int8Array(BOARD_SIZE).fill(-1),
    houses: new Uint8Array(BOARD_SIZE),
    mortgaged: new Uint8Array(BOARD_SIZE),
    groupCounts: new Uint8Array(playerCount * 10),
    features: new Float64Array(INPUTS),
    aliveCount: playerCount,
  };
  const random = new FastRandom(seed);
  let current = 0;
  const tradedThisTurn = new Uint8Array(BOARD_SIZE);
  for (let turn = 0; turn < maxTurns && state.aliveCount > 1; turn++) {
    if (!state.alive[current]) {
      current = (current + 1) % playerCount;
      tradedThisTurn.fill(0);
      continue;
    }
    const brain = brainFor(brains, current);
    tryTradesAtWindow(
      state,
      current,
      brains,
      random,
      turn,
      maxTurns,
      -1,
      tradedThisTurn,
    );
    const die1 = 1 + random.nextInt(6);
    const die2 = 1 + random.nextInt(6);
    const roll = die1 + die2;
    let isDouble = die1 === die2;

    if (state.jailTurns[current] > 0) {
      fillFeatures(
        state.features,
        current,
        state.cash,
        state.alive,
        state.cash[current] - 50,
        50,
        0,
        0,
        0,
        turn,
        maxTurns,
      );
      const leave =
        evaluate(brain, state.features, LEAGUE_OUTPUT.leaveJail) > 0;
      if (isDouble) {
        state.jailTurns[current] = 0;
      } else if (leave || state.jailTurns[current] >= 3) {
        state.cash[current] -= 50;
        state.jailTurns[current] = 0;
        resolveDebt(state, current, -1);
        if (!state.alive[current]) {
          current = (current + 1) % playerCount;
          tradedThisTurn.fill(0);
          continue;
        }
      } else {
        state.jailTurns[current]++;
        tryTradesAtWindow(
          state,
          current,
          brains,
          random,
          turn,
          maxTurns,
          0,
          tradedThisTurn,
        );
        current = (current + 1) % playerCount;
        tradedThisTurn.fill(0);
        continue;
      }
    }

    state.doubles[current] = isDouble ? state.doubles[current] + 1 : 0;
    if (state.doubles[current] >= 3) {
      state.positions[current] = 10;
      state.jailTurns[current] = 1;
      state.doubles[current] = 0;
      tryTradesAtWindow(
        state,
        current,
        brains,
        random,
        turn,
        maxTurns,
        0,
        tradedThisTurn,
      );
      current = (current + 1) % playerCount;
      tradedThisTurn.fill(0);
      continue;
    }

    const oldPosition = state.positions[current];
    const position = (oldPosition + roll) % BOARD_SIZE;
    state.positions[current] = position;
    if (position < oldPosition) state.cash[current] += 200;
    const type = TILE_TYPES[position];
    if (type === 1) {
      const owner = state.owners[position];
      if (owner < 0) {
        if (wantsToBuy(state, current, position, brain, turn, maxTurns)) {
          state.cash[current] -= PRICES[position];
          state.owners[position] = current;
          state.groupCounts[current * 10 + GROUPS[position]]++;
        } else {
          auction(state, position, brains, turn, maxTurns);
        }
      } else if (owner !== current && state.alive[owner] && !state.mortgaged[position]) {
        const rent = calculateRent(state, position, owner, roll);
        state.cash[current] -= rent;
        state.cash[owner] += rent;
        resolveDebt(state, current, owner);
      }
    } else if (type === 2) {
      state.cash[current] -= position === 38 ? 100 : 200;
      resolveDebt(state, current, -1);
    } else if (type === 3) {
      state.cash[current] += CARD_AMOUNTS[random.nextInt(CARD_AMOUNTS.length)];
      resolveDebt(state, current, -1);
    } else if (type === 4) {
      state.positions[current] = 10;
      state.jailTurns[current] = 1;
      state.doubles[current] = 0;
      isDouble = false;
    }
    if (state.alive[current]) {
      tryTradesAtWindow(
        state,
        current,
        brains,
        random,
        turn,
        maxTurns,
        0.5,
        tradedThisTurn,
      );
      tryBuild(state, current, brain, turn, maxTurns);
      tryTradesAtWindow(
        state,
        current,
        brains,
        random,
        turn,
        maxTurns,
        1,
        tradedThisTurn,
      );
    }
    if (!isDouble || !state.alive[current]) {
      current = (current + 1) % playerCount;
      tradedThisTurn.fill(0);
    }
  }

  const wealth = Array(playerCount).fill(-1000);
  for (let player = 0; player < playerCount; player++) {
    if (state.alive[player]) wealth[player] = netWorth(state, player);
  }
  let rank = 1;
  for (let player = 0; player < playerCount; player++) {
    if (player !== focusSeat && wealth[player] > wealth[focusSeat]) rank++;
  }
  return { rank, wealth: wealth[focusSeat] };
}

function outcomeScore(outcome, playerCount) {
  const normalizedRank = (playerCount - outcome.rank) / Math.max(1, playerCount - 1);
  const rankScore = -320 + 1520 * Math.pow(normalizedRank, 1.6);
  return rankScore + clamp(outcome.wealth, -500, 5000) * 0.055;
}

function selectOpponent(population, hall, excluded, random) {
  if (hall.length && random.nextDouble() < 0.34) {
    return hall[random.nextInt(hall.length)];
  }
  let index = random.nextInt(population.length - 1);
  if (index >= excluded) index++;
  return population[index];
}

function evaluateTrainingChunk(job) {
  const {
    population,
    hall,
    generation,
    games,
    maxTurns,
    playerCount,
    seed,
    start,
    end,
  } = job;
  const fitness = [];
  for (let candidate = start; candidate < end; candidate++) {
    let total = 0;
    for (let game = 0; game < games; game++) {
      const seat = game % playerCount;
      const random = new FastRandom(mixSeed(seed, generation, candidate, game));
      const brains = Array.from(
        { length: playerCount },
        () => selectOpponent(population, hall, candidate, random),
      );
      brains[seat] = population[candidate];
      const outcome = playGame(
        brains,
        seat,
        mixSeed(seed ^ 0x9e3779b9, generation, game),
        maxTurns,
      );
      total += outcomeScore(outcome, playerCount);
    }
    fitness.push(total / games);
  }
  return { start, fitness, gameCount: (end - start) * games };
}

function evaluateValidationChunk(job) {
  const { candidate, baselines, maxTurns, playerCount, seed, start, end } = job;
  let scoreSum = 0;
  let wins = 0;
  let rankSum = 0;
  for (let game = start; game < end; game++) {
    const seat = game % playerCount;
    const brains = Array.from(
      { length: playerCount },
      (_, index) => baselines[index % baselines.length],
    );
    brains[seat] = candidate;
    const outcome = playGame(brains, seat, mixSeed(seed, game), maxTurns);
    scoreSum += outcomeScore(outcome, playerCount);
    rankSum += outcome.rank;
    if (outcome.rank === 1) wins++;
  }
  return { scoreSum, wins, rankSum, gameCount: end - start };
}

function evaluateSeatChunk(job) {
  const { candidate, maxTurns, playerCount, seed, start, end } = job;
  const wins = Array(playerCount).fill(0);
  const counts = Array(playerCount).fill(0);
  for (let game = start; game < end; game++) {
    const focus = game % playerCount;
    const outcome = playGame(
      Array(playerCount).fill(candidate),
      focus,
      mixSeed(seed, game),
      maxTurns,
    );
    counts[focus]++;
    if (outcome.rank === 1) wins[focus]++;
  }
  return { wins, counts, gameCount: end - start };
}

function workerExecute(type, job) {
  if (type === "train") return evaluateTrainingChunk(job);
  if (type === "validate") return evaluateValidationChunk(job);
  if (type === "seat") return evaluateSeatChunk(job);
  throw new Error(`Unknown worker operation: ${type}`);
}

class WorkerPool {
  constructor(size) {
    this.size = Math.max(1, size);
    this.workers = [];
    if (this.size > 1) {
      for (let i = 0; i < this.size; i++) {
        this.workers.push(
          new Worker(new URL(import.meta.url), { workerData: { leagueWorker: true } }),
        );
      }
    }
  }

  async run(type, jobs) {
    if (this.size === 1) return jobs.map((job) => workerExecute(type, job));
    return Promise.all(
      jobs.map(
        (job, index) =>
          new Promise((resolvePromise, rejectPromise) => {
            const worker = this.workers[index];
            const onError = (error) => {
              worker.off("message", onMessage);
              rejectPromise(error);
            };
            const onMessage = (message) => {
              worker.off("error", onError);
              if (message.error) rejectPromise(new Error(message.error));
              else resolvePromise(message.result);
            };
            worker.once("error", onError);
            worker.once("message", onMessage);
            worker.postMessage({ type, job });
          }),
      ),
    );
  }

  async close() {
    await Promise.all(this.workers.map((worker) => worker.terminate()));
  }
}

function partition(total, count) {
  const jobs = [];
  for (let i = 0; i < count; i++) {
    const start = Math.floor((total * i) / count);
    const end = Math.floor((total * (i + 1)) / count);
    if (end > start) jobs.push([start, end]);
  }
  return jobs;
}

function wilsonLowerBound(wins, total, z = 1.96) {
  if (total <= 0) return 0;
  const rate = wins / total;
  const z2 = z * z;
  const denominator = 1 + z2 / total;
  const center = rate + z2 / (2 * total);
  const margin =
    z *
    Math.sqrt((rate * (1 - rate) + z2 / (4 * total)) / total);
  return (center - margin) / denominator;
}

class LeagueTrainer {
  constructor(options) {
    this.options = options;
    this.heuristic = createHeuristic();
    this.conservative = createHeuristic();
    this.aggressive = createHeuristic();
    shiftStyle(this.conservative, 0.6, -0.35);
    shiftStyle(this.aggressive, -0.55, 0.4);
    this.trainingGames = 0;
    this.validationGames = 0;
    this.pool = new WorkerPool(options.workers);
  }

  createPopulation(random) {
    const population = [
      cloneGenome(this.heuristic),
      cloneGenome(this.conservative),
      cloneGenome(this.aggressive),
    ];
    while (population.length < this.options.population) {
      const genome = cloneGenome(this.heuristic);
      mutate(genome, random, 0.35 + random.nextDouble() * 0.3);
      if (population.length % 7 === 0) {
        genome.hiddenCount = 2 + random.nextInt(MAX_HIDDEN - 1);
      }
      population.push(genome);
    }
    return population;
  }

  async evaluatePopulation(population, hall, generation) {
    const chunks = partition(population.length, Math.min(this.options.workers, population.length));
    const results = await this.pool.run(
      "train",
      chunks.map(([start, end]) => ({
        population,
        hall,
        generation,
        games: this.options.gamesPerCandidate,
        maxTurns: this.options.maxTurns,
        playerCount: this.options.playerCount,
        seed: this.options.seed,
        start,
        end,
      })),
    );
    const fitness = Array(population.length).fill(0);
    for (const result of results) {
      this.trainingGames += result.gameCount;
      for (let i = 0; i < result.fitness.length; i++) {
        fitness[result.start + i] = result.fitness[i];
      }
    }
    return fitness;
  }

  async evaluateAgainstBaselines(candidate, games, seedOffset) {
    const chunks = partition(games, Math.min(this.options.workers, games));
    const results = await this.pool.run(
      "validate",
      chunks.map(([start, end]) => ({
        candidate,
        baselines: [this.heuristic, this.conservative, this.aggressive],
        maxTurns: this.options.maxTurns,
        playerCount: this.options.playerCount,
        seed: mixSeed(this.options.seed, seedOffset),
        start,
        end,
      })),
    );
    let scoreSum = 0;
    let wins = 0;
    let rankSum = 0;
    let count = 0;
    for (const result of results) {
      scoreSum += result.scoreSum;
      wins += result.wins;
      rankSum += result.rankSum;
      count += result.gameCount;
    }
    this.validationGames += count;
    return {
      score: scoreSum / count,
      wins,
      gameCount: count,
      winRate: wins / count,
      winRateLowerBound: wilsonLowerBound(wins, count),
      averageRank: rankSum / count,
    };
  }

  async evaluateSeatFairness(candidate, games) {
    const chunks = partition(games, Math.min(this.options.workers, games));
    const results = await this.pool.run(
      "seat",
      chunks.map(([start, end]) => ({
        candidate,
        maxTurns: this.options.maxTurns,
        playerCount: this.options.playerCount,
        seed: mixSeed(this.options.seed, 1700000003),
        start,
        end,
      })),
    );
    const wins = Array(this.options.playerCount).fill(0);
    const counts = Array(this.options.playerCount).fill(0);
    for (const result of results) {
      this.validationGames += result.gameCount;
      for (let seat = 0; seat < this.options.playerCount; seat++) {
        wins[seat] += result.wins[seat];
        counts[seat] += result.counts[seat];
      }
    }
    return wins.map((value, seat) => value / Math.max(1, counts[seat]));
  }

  breed(population, fitness, order, random, generation) {
    const next = [];
    const eliteCount = Math.min(5, population.length);
    for (let i = 0; i < eliteCount; i++) next.push(cloneGenome(population[order[i]]));
    const parentPool = Math.min(12, population.length);
    const progress = generation / Math.max(1, this.options.generations);
    const sigma = 0.3 * (1 - progress) + 0.075;
    while (next.length < population.length) {
      const a = tournament(order, fitness, parentPool, random);
      const b = tournament(order, fitness, parentPool, random);
      const child = crossover(population[a], population[b], random);
      mutate(child, random, sigma * (0.8 + random.nextDouble() * 0.5));
      next.push(child);
    }
    return next;
  }

  async run() {
    const started = performance.now();
    const random = new FastRandom(this.options.seed);
    let population = this.createPopulation(random);
    const hall = [cloneGenome(this.heuristic)];
    let best = cloneGenome(this.heuristic);
    let bestValidation = Number.NEGATIVE_INFINITY;
    let stoppingReference = Number.NEGATIVE_INFINITY;
    let bestGeneration = 0;
    let bestWinRate = 0;
    let bestWinRateLowerBound = 0;
    let bestAverageRank = this.options.playerCount;
    let completedGenerations = 0;
    let plateauChecks = 0;
    let stopReason = "maximum generation budget reached";
    try {
      for (let generation = 1; generation <= this.options.generations; generation++) {
        completedGenerations = generation;
        const fitness = await this.evaluatePopulation(population, hall, generation);
        const order = Array.from(population.keys()).sort((a, b) => fitness[b] - fitness[a]);
        const champion = population[order[0]];
        if (generation === 1 || generation % this.options.validationEvery === 0) {
          const validation = await this.evaluateAgainstBaselines(
            champion,
            this.options.validationGames,
            generation * 104729,
          );
          if (validation.score > bestValidation) {
            bestValidation = validation.score;
            best = cloneGenome(champion);
            bestGeneration = generation;
            bestWinRate = validation.winRate;
            bestWinRateLowerBound = validation.winRateLowerBound;
            bestAverageRank = validation.averageRank;
          }
          if (validation.score >= stoppingReference + this.options.minimumImprovement) {
            stoppingReference = validation.score;
            plateauChecks = 0;
          } else {
            plateauChecks++;
          }
          const progressRecord = {
              type: "progress",
              generation,
              generations: this.options.generations,
              trainingGames: this.trainingGames,
              validationGames: this.validationGames,
              bestValidation: Number(bestValidation.toFixed(4)),
              currentValidation: Number(validation.score.toFixed(4)),
              currentWinRate: Number(validation.winRate.toFixed(4)),
              bestWinRate: Number(bestWinRate.toFixed(4)),
              bestWinRateLowerBound: Number(bestWinRateLowerBound.toFixed(4)),
              bestAverageRank: Number(bestAverageRank.toFixed(4)),
              targetWinRateLowerBound: this.options.targetWinRate,
              targetAverageRank: this.options.targetAverageRank,
              standardMet:
                bestWinRateLowerBound >= this.options.targetWinRate &&
                bestAverageRank <= this.options.targetAverageRank,
              hiddenNodes: champion.hiddenCount,
              plateauChecks,
              elapsedSeconds: Number(((performance.now() - started) / 1000).toFixed(1)),
            };
          const progressLine = JSON.stringify(progressRecord);
          console.log(progressLine);
          if (this.options.progressPath) {
            appendFileSync(this.options.progressPath, `${progressLine}\n`, "utf8");
          }
          if (
            generation >= this.options.minimumGenerations &&
            plateauChecks >= this.options.patience &&
            bestWinRateLowerBound >= this.options.targetWinRate &&
            bestAverageRank <= this.options.targetAverageRank
          ) {
            stopReason =
              `standard reached: 95% win-rate lower bound ${bestWinRateLowerBound.toFixed(4)} ` +
              `>= ${this.options.targetWinRate.toFixed(4)}, average rank ` +
              `${bestAverageRank.toFixed(4)} <= ${this.options.targetAverageRank.toFixed(4)}; ` +
              `${plateauChecks} stable validation checks`;
            break;
          }
        }
        if (generation % 8 === 0) {
          hall.push(cloneGenome(champion));
          if (hall.length > 18) hall.splice(1, 1);
        }
        population = this.breed(population, fitness, order, random, generation);
      }
      const finalEvaluation = await this.evaluateAgainstBaselines(
        best,
        this.options.finalEvaluationGames,
        900000001,
      );
      const seatWinRates = await this.evaluateSeatFairness(
        best,
        Math.max(4000, Math.floor(this.options.finalEvaluationGames / 2)),
      );
      return {
        version: "league-neuroevolution-v5-marginal-trade-valuation",
        algorithm: "NEAT-inspired topology mutation + symmetric marginal portfolio trade valuation + purchase-auction economic consistency + multi-window repeated bilateral trade + league self-play + hall of fame",
        playerCount: this.options.playerCount,
        seed: this.options.seed,
        generations: completedGenerations,
        maximumGenerations: this.options.generations,
        bestGeneration,
        stoppedEarly: completedGenerations < this.options.generations,
        stopReason,
        adaptiveStopping: {
          minimumGenerations: this.options.minimumGenerations,
          validationInterval: this.options.validationEvery,
          patience: this.options.patience,
          minimumImprovement: this.options.minimumImprovement,
          targetWinRateLowerBound: this.options.targetWinRate,
          targetAverageRank: this.options.targetAverageRank,
          confidenceLevel: 0.95,
        },
        stoppingStandard: {
          targetWinRateLowerBound: this.options.targetWinRate,
          targetAverageRank: this.options.targetAverageRank,
          confidenceLevel: 0.95,
          minimumGenerations: this.options.minimumGenerations,
          patience: this.options.patience,
        },
        population: this.options.population,
        trainingGames: this.trainingGames,
        validationGames: this.validationGames,
        maxTurns: this.options.maxTurns,
        workers: this.options.workers,
        trainingSeconds: Number(((performance.now() - started) / 1000).toFixed(3)),
        winRateVsHeuristic: Number(finalEvaluation.winRate.toFixed(6)),
        averageRankVsHeuristic: Number(finalEvaluation.averageRank.toFixed(6)),
        scoreVsHeuristic: Number(finalEvaluation.score.toFixed(6)),
        achievedStandard: {
          winRateLowerBound: Number(finalEvaluation.winRateLowerBound.toFixed(6)),
          averageRank: Number(finalEvaluation.averageRank.toFixed(6)),
          passed:
            finalEvaluation.winRateLowerBound >= this.options.targetWinRate &&
            finalEvaluation.averageRank <= this.options.targetAverageRank,
        },
        seatWinRates: seatWinRates.map((value) => Number(value.toFixed(6))),
        hiddenCount: best.hiddenCount,
        inputCount: INPUTS,
        outputCount: OUTPUTS,
        featureNames: [
          "cash",
          "postCash",
          "priceOrCost",
          "rentOrReturn",
          "groupProgress",
          "completesSet",
          "gamePhase",
          "relativeWealth",
          "tradeWindow",
        ],
        actionNames: [...LEAGUE_ACTION_NAMES],
        inputHidden: best.inputHidden.map(roundWeight),
        hiddenBias: best.hiddenBias.map(roundWeight),
        hiddenOutput: best.hiddenOutput.map(roundWeight),
        outputBias: best.outputBias.map(roundWeight),
      };
    } finally {
      await this.pool.close();
    }
  }
}

function tournament(order, fitness, pool, random) {
  let best = order[random.nextInt(pool)];
  for (let i = 0; i < 2; i++) {
    const challenger = order[random.nextInt(pool)];
    if (fitness[challenger] > fitness[best]) best = challenger;
  }
  return best;
}

function roundWeight(value) {
  return Number(value.toFixed(8));
}

function parseOptions(args) {
  const values = new Map();
  for (let i = 0; i < args.length - 1; i += 2) {
    if (args[i].startsWith("--")) values.set(args[i].slice(2), args[i + 1]);
  }
  const integer = (key, fallback) => {
    const parsed = Number.parseInt(values.get(key) ?? "", 10);
    return Number.isFinite(parsed) ? parsed : fallback;
  };
  const number = (key, fallback) => {
    const parsed = Number.parseFloat(values.get(key) ?? "");
    return Number.isFinite(parsed) ? parsed : fallback;
  };
  const playerCount = Math.max(3, Math.min(5, integer("players", 4)));
  const defaultMinimumGenerations =
    playerCount === 3 ? 24 : playerCount === 5 ? 40 : 32;
  const defaultPatience =
    playerCount === 3 ? 8 : playerCount === 5 ? 12 : 10;
  const defaultTargetWinRate =
    playerCount === 3 ? 0.5 : playerCount === 5 ? 0.25 : 0.35;
  const defaultTargetAverageRank =
    playerCount === 3 ? 1.65 : playerCount === 5 ? 2.45 : 2.1;
  return {
    playerCount,
    generations: integer("generations", 300),
    minimumGenerations: integer("min-generations", defaultMinimumGenerations),
    patience: integer("patience", defaultPatience),
    minimumImprovement: number("min-delta", 2),
    targetWinRate: number("target-win-rate", defaultTargetWinRate),
    targetAverageRank: number("target-average-rank", defaultTargetAverageRank),
    population: integer("population", 40),
    gamesPerCandidate: integer("games", 175),
    validationEvery: integer("validate-every", 6),
    validationGames: integer("validation-games", 1200),
    finalEvaluationGames: integer("final-games", 20000),
    maxTurns: integer("max-turns", 65 * playerCount),
    seed: integer("seed", 20260724),
    workers: Math.max(1, integer("workers", Math.min(8, availableParallelism()))),
    outputPath: resolve(values.get("output") ?? "app/pretrained-model.json"),
    progressPath: values.has("progress-output")
      ? resolve(values.get("progress-output"))
      : null,
  };
}

export {
  FastRandom,
  LeagueTrainer,
  auction,
  auctionBidCeiling,
  calculateRent,
  createEmptyGenome,
  createHeuristic,
  netWorth,
  playGame,
  resolveDebt,
  tryBuild,
  tryTrade,
  tryTradesAtWindow,
  wantsToBuy,
};

if (!isMainThread && workerData?.leagueWorker) {
  parentPort.on("message", ({ type, job }) => {
    try {
      parentPort.postMessage({ result: workerExecute(type, job) });
    } catch (error) {
      parentPort.postMessage({ error: error instanceof Error ? error.stack : String(error) });
    }
  });
}

const isDirectExecution =
  isMainThread &&
  Boolean(process.argv[1]) &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href;

if (isDirectExecution) {
  const options = parseOptions(process.argv.slice(2));
  if (options.progressPath) {
    await mkdir(dirname(options.progressPath), { recursive: true });
    await writeFile(options.progressPath, "", "utf8");
  }
  const trainer = new LeagueTrainer(options);
  const result = await trainer.run();
  await mkdir(dirname(options.outputPath), { recursive: true });
  await writeFile(options.outputPath, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  const completionRecord = {
      type: "complete",
      trainingGames: result.trainingGames,
      validationGames: result.validationGames,
      generations: result.generations,
      winRateVsHeuristic: result.winRateVsHeuristic,
      averageRankVsHeuristic: result.averageRankVsHeuristic,
      outputPath: options.outputPath,
    };
  const completionLine = JSON.stringify(completionRecord);
  console.log(completionLine);
  if (options.progressPath) {
    appendFileSync(options.progressPath, `${completionLine}\n`, "utf8");
  }
}
