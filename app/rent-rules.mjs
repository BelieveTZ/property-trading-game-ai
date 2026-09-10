import { DEED_DETAILS, GROUP_SIZES, TILES } from "./board-catalog.mjs";

export function propertyRent(tile, state, ownerProperties, diceTotal = 0) {
  if (state.mortgaged) return 0;
  const deed = DEED_DETAILS[tile.id];
  if (deed?.kind === "station") {
    const count = ownerProperties.filter(
      (property) => TILES[property.tileId].type === "station",
    ).length;
    return deed.rents[Math.max(0, Math.min(deed.rents.length - 1, count - 1))];
  }
  if (deed?.kind === "utility") {
    const count = ownerProperties.filter(
      (property) => TILES[property.tileId].type === "utility",
    ).length;
    return diceTotal * (count === 2 ? 10 : 4);
  }
  const groupOwned =
    Boolean(tile.group) &&
    ownerProperties.filter(
      (property) => TILES[property.tileId].group === tile.group,
    ).length === GROUP_SIZES[tile.group];
  if (deed?.kind === "property") {
    if (state.houses === 0) return deed.rents[0] * (groupOwned ? 2 : 1);
    return deed.rents[Math.max(0, Math.min(5, state.houses))];
  }
  return tile.rent ?? 0;
}
