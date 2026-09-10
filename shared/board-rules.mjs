import catalog from "./board-rules.json" with { type: "json" };

export const BOARD_SIZE = catalog.boardSize;
export const GROUP_ORDER = Object.freeze([...catalog.groupOrder]);
export const GROUP_TILES = Object.freeze(
  Object.fromEntries(
    Object.entries(catalog.groups).map(([group, tiles]) => [group, Object.freeze([...tiles])]),
  ),
);
export const DEED_TILE_IDS = Object.freeze(
  Object.keys(catalog.deeds).map(Number).sort((left, right) => left - right),
);

const rules = Object.freeze(
  Object.fromEntries(
    Object.entries(catalog.deeds).map(([tileId, rule]) => [
      Number(tileId),
      Object.freeze({ ...rule, rents: Object.freeze([...rule.rents]) }),
    ]),
  ),
);

export function boardRuleFor(tileId) {
  return rules[tileId];
}

export function indexedBoardRules() {
  const prices = Array(BOARD_SIZE).fill(0);
  const baseRents = Array(BOARD_SIZE).fill(0);
  const mortgages = Array(BOARD_SIZE).fill(0);
  const buildCosts = Array(BOARD_SIZE).fill(0);
  const groupIndexes = Array(BOARD_SIZE).fill(-1);
  const propertyRents = {};
  for (const tileId of DEED_TILE_IDS) {
    const rule = boardRuleFor(tileId);
    prices[tileId] = rule.price;
    baseRents[tileId] = rule.baseRent;
    mortgages[tileId] = rule.mortgage;
    buildCosts[tileId] = rule.buildCost;
    groupIndexes[tileId] = GROUP_ORDER.indexOf(rule.group);
    if (rule.rents.length) propertyRents[tileId] = [...rule.rents];
  }
  return {
    prices,
    baseRents,
    mortgages,
    buildCosts,
    groupIndexes,
    groupSizes: GROUP_ORDER.map((group) => GROUP_TILES[group].length),
    propertyRents,
  };
}
