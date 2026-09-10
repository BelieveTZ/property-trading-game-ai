const RUNTIME_VERSION = 1;

function normalizeSeed(seed) {
  const normalized = Number(seed) >>> 0;
  return normalized === 0 ? 0x6d2b79f5 : normalized;
}

function nextRandom(state) {
  let next = normalizeSeed(state);
  next ^= next << 13;
  next ^= next >>> 17;
  next ^= next << 5;
  next >>>= 0;
  return { state: normalizeSeed(next), value: next / 0x100000000 };
}

function physicalCardIds(cards) {
  return cards.flatMap((card) =>
    Array.from({ length: card.copies ?? 1 }, () => card.id),
  );
}

function shuffle(ids, initialState) {
  const order = [...ids];
  let state = initialState;
  for (let index = order.length - 1; index > 0; index -= 1) {
    const random = nextRandom(state);
    state = random.state;
    const swapIndex = Math.floor(random.value * (index + 1));
    [order[index], order[swapIndex]] = [order[swapIndex], order[index]];
  }
  return { order, state };
}

function shuffledDeck(cards, initialState) {
  const result = shuffle(physicalCardIds(cards), initialState);
  return {
    deck: { order: result.order, cursor: 0 },
    state: result.state,
  };
}

export function createGameRuntime(seed, cardDecks) {
  const normalizedSeed = normalizeSeed(seed);
  const chance = shuffledDeck(cardDecks.chance, normalizedSeed);
  const community = shuffledDeck(cardDecks.community, chance.state);
  return {
    version: RUNTIME_VERSION,
    seed: normalizedSeed,
    randomState: community.state,
    decks: {
      chance: chance.deck,
      community: community.deck,
    },
  };
}

export function rollRuntimeDice(runtime) {
  const first = nextRandom(runtime.randomState);
  const second = nextRandom(first.state);
  return {
    dice: [Math.floor(first.value * 6) + 1, Math.floor(second.value * 6) + 1],
    runtime: { ...structuredClone(runtime), randomState: second.state },
  };
}

export function drawRuntimeCard(runtime, deckName, cardDecks) {
  const next = structuredClone(runtime);
  let deck = next.decks[deckName];
  if (!deck || deck.cursor >= deck.order.length) {
    const shuffled = shuffledDeck(cardDecks[deckName], next.randomState);
    deck = shuffled.deck;
    next.decks[deckName] = deck;
    next.randomState = shuffled.state;
  }
  const cardId = deck.order[deck.cursor];
  deck.cursor += 1;
  const card = cardDecks[deckName].find((candidate) => candidate.id === cardId);
  if (!card) throw new Error(`Unknown ${deckName} card: ${cardId}`);
  return { card, runtime: next };
}
