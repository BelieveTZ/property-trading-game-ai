export const MIN_PLAYERS = 3;
export const MAX_PLAYERS = 5;

export const PLAYER_COLORS = Object.freeze([
  "#e65f55",
  "#409e84",
  "#4169a8",
  "#dd9f2f",
  "#8e65aa",
  "#dd6f9f",
]);

export const OFFICIAL_AI_NAMES = Object.freeze([
  "麦克",
  "法布里奇欧",
  "凯维娜",
  "马修",
  "奥瑞儿",
  "达米安",
  "大卫",
  "莎拉菲娜",
  "妮可",
  "安东尼",
  "艾科",
  "玛莉",
  "汤米",
  "班尼托",
  "弗雷多",
  "米山卓",
  "阿玛迪恩",
  "艾蜜莉恩",
  "李奥",
  "菲立佩",
  "麦可",
]);

const LEGACY_DEFAULT_AI_NAMES = new Set([
  "林舟",
  "小麦",
  "阿岚",
  "向北",
  "可可",
]);

export function shuffledAiNames(random = Math.random) {
  const names = [...OFFICIAL_AI_NAMES];
  for (let index = names.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [names[index], names[swapIndex]] = [names[swapIndex], names[index]];
  }
  return names;
}

export function pickUnusedAiName(players) {
  const usedNames = new Set(players.map((player) => player.name));
  return (
    shuffledAiNames().find((name) => !usedNames.has(name)) ??
    `玩家${players.length + 1}`
  );
}

export function migrateLegacyAiNames(players, myPlayerId) {
  const reservedNames = new Set(
    players
      .filter(
        (player) =>
          player.id === myPlayerId ||
          !LEGACY_DEFAULT_AI_NAMES.has(player.name),
      )
      .map((player) => player.name),
  );
  const replacements = shuffledAiNames().filter(
    (name) => !reservedNames.has(name),
  );
  let replacementIndex = 0;
  return players.map((player) => {
    if (
      player.id === myPlayerId ||
      !LEGACY_DEFAULT_AI_NAMES.has(player.name)
    ) {
      return player;
    }
    const name = replacements[replacementIndex] ?? `玩家${player.id + 1}`;
    replacementIndex += 1;
    return { ...player, name };
  });
}

export function createPlayer(id, name) {
  return {
    id,
    name:
      name ??
      (id === 0
        ? "你"
        : OFFICIAL_AI_NAMES[(id - 1) % OFFICIAL_AI_NAMES.length]),
    color: PLAYER_COLORS[id % PLAYER_COLORS.length],
    cash: 1500,
    position: 0,
    inJail: false,
    jailTurns: 0,
    jailFreeCards: 0,
    bankrupt: false,
  };
}
