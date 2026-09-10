export const LEAGUE_INPUT_COUNT = 9;

export const LEAGUE_OUTPUT = Object.freeze({
  buy: 0,
  auctionBid: 1,
  build: 2,
  leaveJail: 3,
  cashReserve: 4,
  trade: 5,
});

export const LEAGUE_ACTION_NAMES = Object.freeze(Object.keys(LEAGUE_OUTPUT));

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

export function encodeLeagueFeatures(
  {
    cash,
    postCash,
    price,
    outcomeValue,
    groupProgress,
    completesSet,
    turnProgress,
    advantage,
    tradeWindow = 0,
  },
  target = new Array(LEAGUE_INPUT_COUNT).fill(0),
) {
  target[0] = clamp(cash / 1500, -1, 2);
  target[1] = clamp(postCash / 1500, -1, 2);
  target[2] = clamp(price / 400, 0, 2);
  target[3] = clamp(outcomeValue / 200, -2, 2);
  target[4] = clamp(groupProgress, 0, 1);
  target[5] = completesSet ? 1 : 0;
  target[6] = clamp(turnProgress, 0, 1);
  target[7] = clamp(advantage / 2000, -1, 1);
  target[8] = clamp(tradeWindow, -1, 1);
  return target;
}
