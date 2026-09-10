"use client";

import {
  ChangeEvent,
  CSSProperties,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import pretrainedModel3p from "./pretrained-model-3p.json";
import pretrainedModel5p from "./pretrained-model-5p.json";
import pretrainedModel from "./pretrained-model.json";
import {
  CARD_DECK_LABELS as CATALOG_CARD_DECK_LABELS,
  CARD_DECKS as CATALOG_CARD_DECKS,
} from "./card-catalog.mjs";
import {
  evaluateTrade,
  playerNetWorth,
  recommendAuction,
  recommendBuild,
  recommendJail,
  recommendPayment,
  recommendPurchase,
  recommendTradeProposal,
  selectPlayerCountModel,
  tradeProposalKey,
} from "./ai-advisor.mjs";
import {
  applyManagementPlan,
  applyRecordedCard,
  applyTrade,
  addGamePlayer,
  advanceTurn,
  clearPaidDebt,
  declareBankruptcy,
  payJailFee as applyJailFee,
  recordDiceRoll,
  removeGamePlayer,
  resolvePendingDecision,
  settlePropertyAuction,
  skipPropertyAuction,
  useJailFreeCard as applyJailFreeCard,
} from "./game-rules.mjs";
import {
  createFreshSession,
  readSessionSnapshot,
  SESSION_STORAGE_KEY,
  writeSessionSnapshot,
} from "./session-state.mjs";
import {
  MAX_PLAYERS,
  MIN_PLAYERS,
  OFFICIAL_AI_NAMES,
  PLAYER_COLORS,
  createPlayer,
  pickUnusedAiName,
  shuffledAiNames,
} from "./player-catalog.mjs";
import { propertyRent } from "./rent-rules.mjs";
import {
  DEED_DETAILS as BOARD_DEED_DETAILS,
  GROUP_SIZES as BOARD_GROUP_SIZES,
  TILES as BOARD_TILES,
  UTILITY_COLOR,
} from "./board-catalog.mjs";
type TileType =
  | "start"
  | "property"
  | "station"
  | "utility"
  | "tax"
  | "chance"
  | "community"
  | "jail"
  | "parking"
  | "gotojail";

type Tile = {
  id: number;
  name: string;
  short: string;
  type: TileType;
  price?: number;
  rent?: number;
  group?: string;
  color?: string;
  houseCost?: number;
};

type DeedDetails =
  | {
      kind: "property";
      rents: readonly [number, number, number, number, number, number];
      mortgage: number;
      buildingCost: number;
    }
  | {
      kind: "station";
      rents: readonly [number, number, number, number];
      mortgage: number;
    }
  | {
      kind: "utility";
      mortgage: number;
    };

type PropertyState = {
  tileId: number;
  ownerId: number | null;
  houses: number;
  mortgaged: boolean;
};

type Player = {
  id: number;
  name: string;
  color: string;
  cash: number;
  position: number;
  inJail: boolean;
  jailTurns: number;
  jailFreeCards: number;
  bankrupt: boolean;
};

type TradeDraft = {
  fromId: number;
  toId: number;
  fromCash: number;
  toCash: number;
  fromCards: number;
  toCards: number;
  fromPropertyIds: number[];
  toPropertyIds: number[];
};

type ProactiveTradeProposal = {
  key: string;
  draft: TradeDraft;
  summary: string;
  score: number;
};

type CardDeck = "chance" | "community";

type CardEffect =
  | { kind: "cash"; amount: number }
  | { kind: "move"; destination: number; collectGo: boolean }
  | { kind: "nearest"; target: "station" | "utility" }
  | { kind: "back"; spaces: number }
  | { kind: "jail" }
  | { kind: "repairs"; perHouse: number; perHotel: number }
  | { kind: "payEach"; amount: number }
  | { kind: "collectEach"; amount: number }
  | { kind: "jailFree" };

type CardDefinition = {
  id: string;
  label: string;
  effect: CardEffect;
  copies?: number;
};

type Pending =
  | { kind: "property"; tileId: number }
  | { kind: "rent"; tileId: number; ownerId: number; amount: number }
  | { kind: "tax"; amount: number; label: string }
  | { kind: "card"; deck: CardDeck }
  | { kind: "notice"; label: string }
  | null;

type Debt = {
  debtorId: number;
  creditorId: number | null;
  amount: number;
  reason: string;
} | null;

type PaymentInstruction = {
  debtorId: number;
  creditorId: number;
  amount: number;
  reason: string;
};

type GameState = {
  players: Player[];
  properties: PropertyState[];
  activePlayerId: number;
  myPlayerId: number;
  tradeLockPlayerId: number | null;
  tradeLockedTileIds: number[];
  round: number;
  doublesStreak: number;
  extraTurnEligible: boolean;
  lastDice: [number, number] | null;
  pending: Pending;
  debt: Debt;
  paymentQueue: PaymentInstruction[];
  auctionQueue: number[];
  gameOver: boolean;
  winnerId: number | null;
  log: string[];
};

type Policy = {
  liquidity: number;
  roi: number;
  monopoly: number;
  pressure: number;
  reserve: number;
  games: number;
  generation: number;
  fitness: number;
};

type Suggestion = {
  action: string;
};

type JailSuggestion = Suggestion & {
  payToLeave: boolean;
};

type BuildRecommendation = Suggestion & {
  tileIds: number[];
};

type PretrainedModel = {
  version: string;
  playerCount?: number;
  generations: number;
  bestGeneration: number;
  trainingGames: number;
  validationGames: number;
  winRateVsHeuristic: number;
  averageRankVsHeuristic: number;
  hiddenCount: number;
  inputCount: number;
  outputCount: number;
  inputHidden: number[];
  hiddenBias: number[];
  hiddenOutput: number[];
  outputBias: number[];
  actionNames?: string[];
  stoppingStandard?: {
    targetWinRateLowerBound: number;
    targetAverageRank: number;
    confidenceLevel: number;
    minimumGenerations: number;
    patience: number;
  };
  achievedStandard?: {
    winRateLowerBound: number;
    averageRank: number;
    passed: boolean;
  };
};

const LEAGUE_MODELS: Record<3 | 4 | 5, PretrainedModel> = {
  3: pretrainedModel3p as PretrainedModel,
  4: pretrainedModel as PretrainedModel,
  5: pretrainedModel5p as PretrainedModel,
};
function modelForPlayerCount(playerCount: number) {
  return selectPlayerCountModel(LEAGUE_MODELS, playerCount) as PretrainedModel;
}

const TILES = BOARD_TILES as unknown as Tile[];
const DEED_DETAILS = BOARD_DEED_DETAILS as Record<number, DeedDetails>;

const TILE_TYPE_STRIP_COLORS: Record<TileType, string> = {
  start: "#277b63",
  property: "#98512a",
  station: "#252922",
  utility: UTILITY_COLOR,
  tax: "#95633c",
  chance: "#c75e42",
  community: "#4d8060",
  jail: "#84533d",
  parking: "#3c6d64",
  gotojail: "#9b463d",
};

const DEFAULT_POLICY: Policy = {
  liquidity: 0.95,
  roi: 1.25,
  monopoly: 1.5,
  pressure: 0.8,
  reserve: 240,
  games: 0,
  generation: 0,
  fitness: 0,
};

const GROUP_SIZES = BOARD_GROUP_SIZES as Record<string, number>;

const MANAGEMENT_RULES = {
  tiles: TILES,
  groupSizes: GROUP_SIZES,
  mortgageValues: Object.fromEntries(
    Object.entries(DEED_DETAILS).map(([tileId, deed]) => [
      Number(tileId),
      deed.mortgage,
    ]),
  ),
};

const CARD_DECK_LABELS = CATALOG_CARD_DECK_LABELS as Record<CardDeck, string>;
const CARD_DECKS = CATALOG_CARD_DECKS as unknown as Record<
  CardDeck,
  readonly CardDefinition[]
>;

function createGame(playerCount = 4, randomizeAiNames = false): GameState {
  const aiNames = randomizeAiNames
    ? shuffledAiNames()
    : [...OFFICIAL_AI_NAMES];
  return {
    players: Array.from({ length: playerCount }, (_, index) =>
      createPlayer(index, index === 0 ? "你" : aiNames[index - 1]),
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
    log: ["第 1 轮开始 · 所有人从出发点出发"],
  };
}

function money(value: number) {
  return `$${Math.round(value).toLocaleString("zh-CN")}`;
}

function tileGridPosition(id: number): CSSProperties {
  if (id <= 10) return { gridRow: 11, gridColumn: 11 - id };
  if (id <= 20) return { gridRow: 21 - id, gridColumn: 1 };
  if (id <= 30) return { gridRow: 1, gridColumn: id - 19 };
  return { gridRow: id - 29, gridColumn: 11 };
}

function tileEdge(id: number) {
  if (id <= 10) return "bottom";
  if (id <= 20) return "left";
  if (id <= 30) return "top";
  return "right";
}

function tileIcon(tile: Tile) {
  const icons: Record<TileType, string> = {
    start: "←",
    property: "",
    station: "",
    utility: "",
    tax: "税",
    chance: "?",
    community: "▣",
    jail: "",
    parking: "P",
    gotojail: "↙",
  };
  return icons[tile.type];
}

function tileStripColor(tile: Tile) {
  if (tile.type === "utility") return UTILITY_COLOR;
  return tile.color ?? TILE_TYPE_STRIP_COLORS[tile.type];
}

function cardDisplayLabel(card: CardDefinition, tiles: readonly Tile[]) {
  if (card.id === "chance-boardwalk") {
    return `直达${tiles[39].name}。`;
  }
  if (card.id === "chance-illinois") {
    return `直达${tiles[24].name}。如途经“起点”，可领取 $200。`;
  }
  if (card.id === "chance-st-charles") {
    return `直达${tiles[11].name}。如途经“起点”，可领取 $200。`;
  }
  if (card.id === "chance-reading") {
    return `前进${tiles[5].name}。如途经“起点”，可领取 $200。`;
  }
  return card.label;
}

function mortgageValue(tile: Tile) {
  return DEED_DETAILS[tile.id]?.mortgage ?? Math.round((tile.price ?? 0) / 2);
}

function createTradeDraft(fromId = 0, toId = 1): TradeDraft {
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

function groupHasBuildings(
  tile: Tile,
  properties: readonly PropertyState[],
) {
  if (tile.type !== "property" || !tile.group) return false;
  return properties.some(
    (property) =>
      TILES[property.tileId].group === tile.group && property.houses > 0,
  );
}

export default function GameApp() {
  const [game, setGame] = useState<GameState>(() => createGame());
  const [policy, setPolicy] = useState<Policy>(DEFAULT_POLICY);
  const [tab, setTab] = useState<"turn" | "state">("turn");
  const [dice, setDice] = useState<[number, number]>([3, 4]);
  const [auction, setAuction] = useState({ open: false, winnerId: 0, price: 60 });
  const [selectedCardId, setSelectedCardId] = useState("");
  const [cardDiceTotal, setCardDiceTotal] = useState(7);
  const [hydrated, setHydrated] = useState(false);
  const [toast, setToast] = useState("");
  const [selectedDeedTileId, setSelectedDeedTileId] = useState<number | null>(
    null,
  );
  const [selectedReferenceTileId, setSelectedReferenceTileId] = useState<
    number | null
  >(null);
  const [tradeDraft, setTradeDraft] = useState<TradeDraft>(() =>
    createTradeDraft(),
  );
  const [rejectedTradeProposalKeys, setRejectedTradeProposalKeys] = useState<
    string[]
  >([]);
  const [buildDecisionCompleted, setBuildDecisionCompleted] = useState(false);
  const importRef = useRef<HTMLInputElement>(null);

  /* eslint-disable react-hooks/set-state-in-effect -- Session hydration and turn-boundary reconciliation intentionally synchronize local UI state with external game state. */
  useEffect(() => {
    try {
      const savedSession = localStorage.getItem(SESSION_STORAGE_KEY);
      const savedGame = localStorage.getItem("deed-advisor-game-v1");
      const savedPolicy = localStorage.getItem("deed-advisor-policy-v1");
      if (savedSession || savedGame) {
        const source = savedSession ?? {
          game: JSON.parse(savedGame!),
          policy: savedPolicy ? JSON.parse(savedPolicy) : DEFAULT_POLICY,
        };
        const restored = readSessionSnapshot(source, {
          deedTileIds: TILES.filter((tile) =>
            ["property", "station", "utility"].includes(tile.type),
          ).map((tile) => tile.id),
        });
        setGame(restored.game as GameState);
        setPolicy(restored.policy as Policy);
        setAuction(restored.auction);
        setTradeDraft(restored.tradeDraft as TradeDraft);
        setRejectedTradeProposalKeys(restored.rejectedTradeProposalKeys);
        setBuildDecisionCompleted(restored.buildDecisionCompleted);
      } else {
        setGame(createGame(4, true));
      }
    } catch {
      setGame(createGame(4, true));
      setToast("旧存档无法读取，已载入新牌局");
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(
      SESSION_STORAGE_KEY,
      writeSessionSnapshot({
        game,
        policy,
        auction,
        tradeDraft,
        rejectedTradeProposalKeys,
        buildDecisionCompleted,
      }),
    );
  }, [
    auction,
    buildDecisionCompleted,
    game,
    hydrated,
    policy,
    rejectedTradeProposalKeys,
    tradeDraft,
  ]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 2400);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    if (selectedDeedTileId === null && selectedReferenceTileId === null) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSelectedDeedTileId(null);
        setSelectedReferenceTileId(null);
      }
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [selectedDeedTileId, selectedReferenceTileId]);

  useEffect(() => {
    setTradeDraft((current) => {
      const activePlayers = game.players.filter((player) => !player.bankrupt);
      const from = activePlayers.some((player) => player.id === current.fromId)
        ? current.fromId
        : activePlayers[0]?.id ?? 0;
      const to = activePlayers.some(
        (player) => player.id === current.toId && player.id !== from,
      )
        ? current.toId
        : activePlayers.find((player) => player.id !== from)?.id ?? from;
      if (from === current.fromId && to === current.toId) return current;
      return createTradeDraft(from, to);
    });
  }, [game.players]);

  const pendingCardDeck =
    game.pending?.kind === "card" ? game.pending.deck : null;

  useEffect(() => {
    setSelectedCardId("");
    setCardDiceTotal(7);
  }, [pendingCardDeck, game.activePlayerId]);

  const namedTiles = TILES;
  const selectedCard = pendingCardDeck
    ? CARD_DECKS[pendingCardDeck].find((card) => card.id === selectedCardId)
    : undefined;
  const needsUtilityDice =
    selectedCard?.effect.kind === "nearest" &&
    selectedCard.effect.target === "utility";
  const leagueModel = modelForPlayerCount(game.players.length);
  const activePlayer = game.players.find((p) => p.id === game.activePlayerId)!;
  const myPlayer = game.players.find((p) => p.id === game.myPlayerId)!;
  const currentTile = namedTiles[activePlayer.position];
  const myTurn = !game.gameOver && game.activePlayerId === game.myPlayerId;
  useEffect(() => {
    if (tab !== "turn") return;
    setTradeDraft((current) => {
      const activePlayers = game.players.filter((player) => !player.bankrupt);
      const fromId = game.activePlayerId;
      const toId = activePlayers.some(
        (player) => player.id === current.toId && player.id !== fromId,
      )
        ? current.toId
        : activePlayers.find((player) => player.id !== fromId)?.id ?? fromId;
      if (current.fromId === fromId && current.toId === toId) return current;
      return createTradeDraft(fromId, toId);
    });
  }, [game.activePlayerId, game.players, tab]);
  useEffect(() => {
    setRejectedTradeProposalKeys([]);
    setBuildDecisionCompleted(false);
  }, [game.activePlayerId, game.myPlayerId, game.players.length]);
  /* eslint-enable react-hooks/set-state-in-effect */
  const ownerMap = useMemo(
    () => new Map(game.properties.map((property) => [property.tileId, property])),
    [game.properties],
  );
  const selectedDeedTile =
    selectedDeedTileId === null ? null : namedTiles[selectedDeedTileId];
  const selectedDeed =
    selectedDeedTileId === null ? null : DEED_DETAILS[selectedDeedTileId];
  const selectedDeedState =
    selectedDeedTileId === null
      ? null
      : ownerMap.get(selectedDeedTileId) ?? null;
  const selectedReferenceTile =
    selectedReferenceTileId === null
      ? null
      : namedTiles[selectedReferenceTileId];
  const tradeFrom = game.players.find(
    (player) => player.id === tradeDraft.fromId,
  );
  const tradeTo = game.players.find((player) => player.id === tradeDraft.toId);
  const tradeHasContents =
    tradeDraft.fromCash > 0 ||
    tradeDraft.toCash > 0 ||
    tradeDraft.fromCards > 0 ||
    tradeDraft.toCards > 0 ||
    tradeDraft.fromPropertyIds.length > 0 ||
    tradeDraft.toPropertyIds.length > 0;

  const suggestion = useMemo<Suggestion | null>(() => {
    if (!myTurn || !game.pending) return null;
    if (game.pending.kind === "property") {
      const tile = namedTiles[game.pending.tileId];
      const { buy } = recommendPurchase({
        game,
        model: leagueModel,
        policy,
        player: activePlayer,
        tile,
      });
      return {
        action: buy ? `买入 ${tile.name}` : "放弃购买，进入竞拍",
      };
    }
    if (game.pending.kind === "rent") {
      const payment = recommendPayment({
        game,
        player: activePlayer,
        amount: game.pending.amount,
      });
      return {
        action:
          payment.action === "pay"
            ? "支付租金"
            : payment.action === "fundraise"
              ? "记录欠款，抵押或卖房筹款"
              : "记录欠款并宣布破产",
      };
    }
    if (game.pending.kind === "tax") {
      const payment = recommendPayment({
        game,
        player: activePlayer,
        amount: game.pending.amount,
      });
      return {
        action:
          payment.action === "pay"
            ? "确认支出"
            : payment.action === "fundraise"
              ? "记录欠款，抵押或卖房筹款"
              : "记录欠款并宣布破产",
      };
    }
    if (game.pending.kind === "card") {
      return {
        action: `录入实际抽到的${CARD_DECK_LABELS[game.pending.deck]}牌`,
      };
    }
    return {
      action: "结束回合",
    };
  }, [activePlayer, game, myTurn, namedTiles, leagueModel, policy]);

  const auctionSuggestion = useMemo<Suggestion | null>(() => {
    if (
      !auction.open ||
      game.pending?.kind !== "property"
    ) {
      return null;
    }
    const tile = namedTiles[game.pending.tileId];
    if (myPlayer.bankrupt) {
      return {
        action: "不参与竞拍",
      };
    }

    const { participate, maximumBid } = recommendAuction({
      game,
      model: leagueModel,
      policy,
      player: myPlayer,
      tile,
      currentBid: auction.price,
    });
    return {
      action: participate
        ? `参与竞拍，最高建议价 ${money(maximumBid)}`
        : "退出竞拍，不再加价",
    };
  }, [
    auction.open,
    auction.price,
    game,
    leagueModel,
    myPlayer,
    namedTiles,
    policy,
  ]);

  const jailSuggestion = useMemo<JailSuggestion | null>(() => {
    if (
      !myTurn ||
      game.lastDice !== null ||
      !activePlayer.inJail ||
      activePlayer.bankrupt
    ) {
      return null;
    }
    const { payToLeave } = recommendJail({
      game,
      model: leagueModel,
      player: activePlayer,
    });
    return {
      action: payToLeave
        ? "支付 $50 出狱后掷骰"
        : "留在监狱，尝试掷出双数",
      payToLeave,
    };
  }, [
    activePlayer,
    game,
    leagueModel,
    myTurn,
  ]);

  const buildRecommendation = useMemo<BuildRecommendation | null>(() => {
    if (
      !myTurn ||
      game.lastDice === null ||
      game.pending ||
      auction.open ||
      activePlayer.bankrupt ||
      buildDecisionCompleted
    ) {
      return null;
    }

    const { tileIds } = recommendBuild({
      game,
      model: leagueModel,
      player: activePlayer,
    });

    if (!tileIds.length) return null;
    const actions = tileIds.map((tileId) => {
      const property = ownerMap.get(tileId)!;
      return property.houses === 4
        ? `${namedTiles[tileId].short}升级为酒店`
        : `${namedTiles[tileId].short}建造 1 栋房屋`;
    });
    return {
      action: actions.join("；"),
      tileIds,
    };
  }, [
    activePlayer,
    auction.open,
    buildDecisionCompleted,
    game,
    leagueModel,
    myTurn,
    namedTiles,
    ownerMap,
  ]);

  const tradeSuggestion = useMemo<Suggestion | null>(() => {
    if (
      !tradeFrom ||
      !tradeTo ||
      !tradeHasContents ||
      (game.myPlayerId !== tradeFrom.id && game.myPlayerId !== tradeTo.id)
    ) {
      return null;
    }

    const myIsFrom = game.myPlayerId === tradeFrom.id;
    const me = myIsFrom ? tradeFrom : tradeTo;
    const propertyIdsGiven = myIsFrom
      ? tradeDraft.fromPropertyIds
      : tradeDraft.toPropertyIds;
    const propertyIdsReceived = myIsFrom
      ? tradeDraft.toPropertyIds
      : tradeDraft.fromPropertyIds;
    const cashGiven = myIsFrom ? tradeDraft.fromCash : tradeDraft.toCash;
    const cashReceived = myIsFrom ? tradeDraft.toCash : tradeDraft.fromCash;
    const cardsGiven = myIsFrom ? tradeDraft.fromCards : tradeDraft.toCards;
    const cardsReceived = myIsFrom ? tradeDraft.toCards : tradeDraft.fromCards;
    const evaluation = evaluateTrade(
      game,
      leagueModel,
      me,
      {
        propertyIdsGiven,
        propertyIdsReceived,
        cashGiven,
        cashReceived,
        cardsGiven,
        cardsReceived,
      },
    );
    return {
      action: evaluation.accepted ? "接受这笔交易" : "拒绝这笔交易",
    };
  }, [
    game,
    leagueModel,
    tradeDraft,
    tradeFrom,
    tradeHasContents,
    tradeTo,
  ]);

  const proactiveTradeProposal = useMemo<ProactiveTradeProposal | null>(() => {
    if (!myTurn) return null;
    return recommendTradeProposal({
      game,
      model: leagueModel,
      player: myPlayer,
      rejectedKeys: rejectedTradeProposalKeys,
    }) as ProactiveTradeProposal | null;
  }, [
    game,
    leagueModel,
    myPlayer,
    myTurn,
    rejectedTradeProposalKeys,
  ]);

  function rejectProactiveTrade(proposal: ProactiveTradeProposal) {
    setRejectedTradeProposalKeys((current) =>
      current.includes(proposal.key) ? current : [...current, proposal.key],
    );
    setTradeDraft((current) =>
      tradeProposalKey(current) === proposal.key
        ? createTradeDraft(proposal.draft.fromId, proposal.draft.toId)
        : current,
    );
  }

  function appendLog(next: GameState, message: string) {
    next.log = [message, ...next.log].slice(0, 60);
  }

  function recordDice() {
    if (game.pending) {
      setToast("请先处理当前地块的决策");
      return;
    }
    setGame((previous) => {
      const [d1, d2] = dice;
      const result = recordDiceRoll(previous, {
        dieOne: d1,
        dieTwo: d2,
      }, {
        tiles: namedTiles,
        rentFor: ({ tile, property, ownerProperties, diceTotal }: {
          tile: Tile;
          property: PropertyState;
          ownerProperties: PropertyState[];
          diceTotal: number;
        }) => propertyRent(tile, property, ownerProperties, diceTotal),
      }) as { ok: boolean; game: GameState };
      return result.ok ? result.game : previous;
    });
  }

  function resolvePending(action: "buy" | "skip" | "confirm") {
    const result = resolvePendingDecision(game, action, {
      tiles: namedTiles,
    }) as {
      ok: boolean;
      reason: string | null;
      game: GameState;
      auction?: { winnerId: number; price: number };
    };
    if (!result.ok) {
      if (result.reason === "insufficient-cash") {
        setToast("现金不足，需先抵押资产或进入竞拍");
      }
      return;
    }
    if (result.auction) {
      setAuction({ open: true, ...result.auction });
      return;
    }
    setGame(result.game);
  }

  function resolveSelectedCard() {
    if (game.pending?.kind !== "card") return;
    const selectedCard = CARD_DECKS[game.pending.deck].find(
      (card) => card.id === selectedCardId,
    );
    if (!selectedCard) {
      setToast("请先选择实体牌局实际抽到的牌");
      return;
    }
    if (
      selectedCard.effect.kind === "nearest" &&
      selectedCard.effect.target === "utility" &&
      (cardDiceTotal < 2 || cardDiceTotal > 12)
    ) {
      setToast("公用事业牌的额外骰子合计应为 2–12");
      return;
    }

    setGame((previous) => {
      const pending = previous.pending;
      if (pending?.kind !== "card") return previous;
      const card = CARD_DECKS[pending.deck].find(
        (item) => item.id === selectedCardId,
      );
      if (!card) return previous;
      const result = applyRecordedCard(
        previous,
        {
          deck: pending.deck,
          card: {
            ...card,
            label: cardDisplayLabel(card, namedTiles),
          },
          diceTotal: cardDiceTotal,
        },
        {
          tiles: namedTiles,
          rentFor: ({ tile, property, ownerProperties, diceTotal }: {
            tile: Tile;
            property: PropertyState;
            ownerProperties: PropertyState[];
            diceTotal: number;
          }) => propertyRent(tile, property, ownerProperties, diceTotal),
        },
      ) as { ok: boolean; game: GameState };
      return result.ok ? result.game : previous;
    });
    setSelectedCardId("");
    setCardDiceTotal(7);
  }

  function useJailFreeCard() {
    const result = applyJailFreeCard(game, game.activePlayerId) as {
      ok: boolean;
      game: GameState;
    };
    if (result.ok) {
      setGame(result.game);
      setToast("已使用监狱通行证，现在可以录入骰子");
    }
  }

  function payToLeaveJail() {
    const result = applyJailFee(game, game.activePlayerId) as {
      ok: boolean;
      game: GameState;
    };
    if (!result.ok) {
      setToast("现金不足以支付 $50 出狱费");
      return;
    }
    setGame(result.game);
  }

  function finishAuction() {
    if (!game.pending || game.pending.kind !== "property") return;
    const result = settlePropertyAuction(
      game,
      { winnerId: auction.winnerId, price: auction.price },
      { tiles: namedTiles },
    ) as {
      ok: boolean;
      game: GameState;
      nextTileId?: number;
    };
    if (!result.ok) {
      setToast("请选择未破产且现金充足的得主，并输入正整数成交价");
      return;
    }
    setGame(result.game);
    setAuction((value) =>
      result.nextTileId === undefined
        ? { ...value, open: false }
        : {
            open: true,
            winnerId: result.game.players.find((player) => !player.bankrupt)?.id ?? 0,
            price: Math.max(10, Math.round((namedTiles[result.nextTileId].price ?? 60) * 0.6)),
          },
    );
  }

  function finishAuctionWithoutWinner() {
    const result = skipPropertyAuction(game) as {
      ok: boolean;
      game: GameState;
      nextTileId?: number;
    };
    if (!result.ok) return;
    setGame(result.game);
    setAuction((value) =>
      result.nextTileId === undefined
        ? { ...value, open: false }
        : {
            open: true,
            winnerId: result.game.players.find((player) => !player.bankrupt)?.id ?? 0,
            price: Math.max(10, Math.round((namedTiles[result.nextTileId].price ?? 60) * 0.6)),
          },
    );
  }

  function endTurn() {
    const result = advanceTurn(game, game.activePlayerId) as {
      ok: boolean;
      reason: string | null;
      game: GameState;
    };
    if (!result.ok) {
      setToast(
        result.reason === "unpaid-debt"
          ? "请先偿清债务或宣布破产"
          : "请先完成当前决策",
      );
      return;
    }
    setRejectedTradeProposalKeys([]);
    setBuildDecisionCompleted(false);
    setGame(result.game);
  }

  function addPlayer() {
    if (game.players.length >= MAX_PLAYERS) {
      setToast("最多支持 5 名玩家");
      return;
    }
    const usedIds = new Set(game.players.map((player) => player.id));
    const id = PLAYER_COLORS.findIndex((_, candidate) => !usedIds.has(candidate));
    const player = createPlayer(id, pickUnusedAiName(game.players));
    const result = addGamePlayer(game, player, MAX_PLAYERS) as {
      ok: boolean;
      game: GameState;
    };
    if (result.ok) setGame(result.game);
  }

  function removePlayer(id: number) {
    if (game.players.length <= MIN_PLAYERS) {
      setToast("牌局至少需要 3 名玩家");
      return;
    }
    const result = removeGamePlayer(game, id, MIN_PLAYERS) as {
      ok: boolean;
      game: GameState;
      fallbackId: number;
    };
    if (!result.ok) return;
    setGame(result.game);
    setAuction((previous) => ({
      ...previous,
      open: result.game.gameOver ? false : previous.open,
      winnerId:
        previous.winnerId === id ? result.fallbackId : previous.winnerId,
    }));
  }

  function updatePlayer(id: number, patch: Partial<Player>) {
    setGame((previous) =>
      clearPaidDebt({
        ...previous,
        players: previous.players.map((player) =>
          player.id === id ? { ...player, ...patch } : player,
        ),
      }) as GameState,
    );
  }

  function changeTradeParticipant(side: "from" | "to", playerId: number) {
    const activePlayers = game.players.filter((player) => !player.bankrupt);
    if (side === "from") {
      const toId =
        playerId === tradeDraft.toId
          ? activePlayers.find((player) => player.id !== playerId)?.id ??
            playerId
          : tradeDraft.toId;
      setTradeDraft(createTradeDraft(playerId, toId));
    } else {
      const fromId =
        playerId === tradeDraft.fromId
          ? activePlayers.find((player) => player.id !== playerId)?.id ??
            playerId
          : tradeDraft.fromId;
      setTradeDraft(createTradeDraft(fromId, playerId));
    }
  }

  function toggleTradeProperty(side: "from" | "to", tileId: number) {
    setTradeDraft((current) => {
      const key =
        side === "from" ? "fromPropertyIds" : "toPropertyIds";
      const selected = current[key];
      return {
        ...current,
        [key]: selected.includes(tileId)
          ? selected.filter((id) => id !== tileId)
          : [...selected, tileId],
      };
    });
  }

  function executeTrade() {
    if (!tradeFrom || !tradeTo || tradeFrom.id === tradeTo.id) {
      setToast("请选择两位不同的玩家");
      return;
    }
    if (!tradeHasContents) {
      setToast("请先录入至少一项交易内容");
      return;
    }

    const fromCash = Math.max(0, Math.floor(tradeDraft.fromCash));
    const toCash = Math.max(0, Math.floor(tradeDraft.toCash));
    const fromCards = Math.max(0, Math.floor(tradeDraft.fromCards));
    const toCards = Math.max(0, Math.floor(tradeDraft.toCards));
    const fromProperties = tradeDraft.fromPropertyIds
      .map((tileId) => ownerMap.get(tileId))
      .filter((property): property is PropertyState => !!property);
    const toProperties = tradeDraft.toPropertyIds
      .map((tileId) => ownerMap.get(tileId))
      .filter((property): property is PropertyState => !!property);
    if (
      fromProperties.length !== tradeDraft.fromPropertyIds.length ||
      toProperties.length !== tradeDraft.toPropertyIds.length ||
      fromProperties.some((property) => property.ownerId !== tradeFrom.id) ||
      toProperties.some((property) => property.ownerId !== tradeTo.id)
    ) {
      setToast("地契归属已经变化，请重新选择交易内容");
      return;
    }
    const blockedProperty = [...fromProperties, ...toProperties].find(
      (property) =>
        groupHasBuildings(namedTiles[property.tileId], game.properties),
    );
    if (blockedProperty) {
      setToast(
        `请先卖出「${namedTiles[blockedProperty.tileId].name}」同色组的全部房屋`,
      );
      return;
    }
    if (tradeFrom.jailFreeCards < fromCards || tradeTo.jailFreeCards < toCards) {
      setToast("监狱通行证数量不足");
      return;
    }
    const interestForFrom = toProperties.reduce(
      (total, property) =>
        property.mortgaged
          ? total + Math.ceil(mortgageValue(namedTiles[property.tileId]) * 0.1)
          : total,
      0,
    );
    const interestForTo = fromProperties.reduce(
      (total, property) =>
        property.mortgaged
          ? total + Math.ceil(mortgageValue(namedTiles[property.tileId]) * 0.1)
          : total,
      0,
    );
    if (tradeFrom.cash - fromCash + toCash < interestForFrom) {
      setToast(`${tradeFrom.name}的现金不足以完成交易`);
      return;
    }
    if (tradeTo.cash - toCash + fromCash < interestForTo) {
      setToast(`${tradeTo.name}的现金不足以完成交易`);
      return;
    }

    const offerSummary = (
      propertyIds: number[],
      cash: number,
      cards: number,
    ) => {
      const parts = propertyIds.map((tileId) => namedTiles[tileId].name);
      if (cash > 0) parts.push(money(cash));
      if (cards > 0) parts.push(`${cards} 张监狱通行证`);
      return parts.length ? parts.join("、") : "无";
    };
    const fromOffer = offerSummary(
      tradeDraft.fromPropertyIds,
      fromCash,
      fromCards,
    );
    const toOffer = offerSummary(
      tradeDraft.toPropertyIds,
      toCash,
      toCards,
    );

    setGame((previous) => {
      const selectedIds = [
        ...tradeDraft.fromPropertyIds,
        ...tradeDraft.toPropertyIds,
      ];
      const mortgageValues = Object.fromEntries(
        previous.properties.map((property) => [
          property.tileId,
          mortgageValue(namedTiles[property.tileId]),
        ]),
      );
      const colorGroups = Object.fromEntries(
        selectedIds.map((tileId) => {
          const group = namedTiles[tileId]?.group;
          return [
            tileId,
            group
              ? previous.properties
                  .filter(
                    (property) => namedTiles[property.tileId]?.group === group,
                  )
                  .map((property) => property.tileId)
              : [tileId],
          ];
        }),
      );
      const result = applyTrade(
        previous,
        {
          ...tradeDraft,
          fromCash,
          toCash,
          fromCards,
          toCards,
        },
        { mortgageValues, colorGroups },
      ) as {
        ok: boolean;
        reason: string | null;
        game: GameState;
        interestTotal?: number;
      };
      if (!result.ok) {
        setToast("交易条件已经变化，请重新录入");
        return previous;
      }
      const next = result.game;
      const from = next.players.find((player) => player.id === tradeFrom.id)!;
      const to = next.players.find((player) => player.id === tradeTo.id)!;
      appendLog(
        next,
        `${from.name} 与 ${to.name} 完成交易：${from.name}交出${fromOffer}；${to.name}交出${toOffer}${(result.interestTotal ?? 0) > 0 ? `；另付抵押利息 ${money(result.interestTotal ?? 0)}` : ""}`,
      );
      return next;
    });
    setTradeDraft(createTradeDraft(tradeFrom.id, tradeTo.id));
    setToast("交易已记录并完成结算");
  }

  function setOwner(tileId: number, ownerId: number | null) {
    setGame((previous) => ({
      ...previous,
      properties: previous.properties.map((property) =>
        property.tileId === tileId
          ? { ...property, ownerId, houses: ownerId === null ? 0 : property.houses }
          : property,
      ),
    }));
  }

  function toggleMortgage(tileId: number) {
    setGame((previous) => {
      const property = previous.properties.find((item) => item.tileId === tileId);
      if (!property) return previous;
      const kind = property.mortgaged ? "unmortgage" : "mortgage";
      const result = applyManagementPlan(
        previous,
        [{ kind, tileId }],
        MANAGEMENT_RULES,
      ) as {
        game: GameState;
        outcomes: Array<{
          ok: boolean;
          reason: string | null;
          playerId?: number;
          amount?: number;
        }>;
      };
      const outcome = result.outcomes[0];
      if (!outcome?.ok) {
        setToast(
          outcome?.reason === "insufficient-cash"
            ? "现金不足以赎回"
            : "请先出售该同色组的全部房屋",
        );
        return previous;
      }
      const owner = result.game.players.find(
        (player) => player.id === outcome.playerId,
      )!;
      const tile = namedTiles[tileId];
      appendLog(
        result.game,
        kind === "unmortgage"
          ? `${owner.name} 以 ${money(outcome.amount ?? 0)} 赎回「${tile.name}」`
          : `${owner.name} 抵押「${tile.name}」，获得 ${money(outcome.amount ?? 0)}`,
      );
      return result.game;
    });
  }

  function changeHouse(tileId: number, delta: number) {
    setGame((previous) => {
      const kind = delta > 0 ? "build" : "sell-building";
      const result = applyManagementPlan(
        previous,
        [{ kind, tileId }],
        MANAGEMENT_RULES,
      ) as {
        game: GameState;
        outcomes: Array<{
          ok: boolean;
          playerId?: number;
          houses?: number;
        }>;
      };
      const outcome = result.outcomes[0];
      if (!outcome?.ok) {
        setToast(delta > 0 ? "当前不符合均衡建房条件" : "当前不符合均衡售房条件");
        return previous;
      }
      const owner = result.game.players.find(
        (player) => player.id === outcome.playerId,
      )!;
      const tile = namedTiles[tileId];
      appendLog(
        result.game,
        `${owner.name}${delta > 0 ? "建造" : "出售"}「${tile.name}」的${outcome.houses === 5 ? "酒店" : "房屋"}`,
      );
      return result.game;
    });
  }

  function executeRecommendedBuilds(tileIds: readonly number[]) {
    setGame((previous) => {
      const result = applyManagementPlan(
        previous,
        tileIds.map((tileId) => ({ kind: "build", tileId })),
        MANAGEMENT_RULES,
      ) as {
        game: GameState;
        outcomes: Array<{
          ok: boolean;
          tileId: number;
          playerId?: number;
          houses?: number;
        }>;
      };
      for (const outcome of result.outcomes.filter((item) => item.ok)) {
        const player = result.game.players.find(
          (item) => item.id === outcome.playerId,
        )!;
        const tile = namedTiles[outcome.tileId];
        appendLog(
          result.game,
          `${player.name} 在「${tile.name}」${outcome.houses === 5 ? "建造酒店" : "建造房屋"}`,
        );
      }
      return result.game;
    });
    setBuildDecisionCompleted(true);
  }

  function markBankrupt(id: number) {
    const player = game.players.find((item) => item.id === id);
    if (!player) return;
    if (player.bankrupt) {
      setToast("破产结算不可直接撤销；如需纠错，请导入此前保存的对局");
      return;
    }
    const mortgageValues = Object.fromEntries(
      game.properties.map((property) => [
        property.tileId,
        mortgageValue(namedTiles[property.tileId]),
      ]),
    );
    const buildingCosts = Object.fromEntries(
      game.properties.map((property) => [
        property.tileId,
        namedTiles[property.tileId].houseCost ?? 0,
      ]),
    );
    const result = declareBankruptcy(game, id, {
      mortgageValues,
      buildingCosts,
    }) as {
      ok: boolean;
      reason: string | null;
      game: GameState;
      auctionTileIds: number[];
    };
    if (!result.ok) {
      setToast("玩家仍可正常结算，不能宣布破产");
      return;
    }
    const next = result.game;
    const bankruptPlayer = next.players.find((item) => item.id === id)!;
    appendLog(next, `${bankruptPlayer.name} 宣布破产`);
    if (next.activePlayerId === id) {
      const alive = next.players.filter((item) => !item.bankrupt);
      const replacement = alive.find((item) => item.id > id) ?? alive[0];
      if (replacement) next.activePlayerId = replacement.id;
    }
    setGame(next);
    if (next.gameOver) {
      setAuction((previous) => ({ ...previous, open: false }));
    } else if (result.auctionTileIds.length) {
      const tileId = result.auctionTileIds[0];
      setAuction({
        open: true,
        winnerId: next.players.find((item) => !item.bankrupt)?.id ?? 0,
        price: Math.max(10, Math.round((namedTiles[tileId].price ?? 60) * 0.6)),
      });
    }
  }

  function exportSession() {
    const payload = writeSessionSnapshot(
      {
        exportedAt: new Date().toISOString(),
        game,
        policy,
        auction,
        tradeDraft,
        rejectedTradeProposalKeys,
        buildDecisionCompleted,
      },
      { space: 2 },
    );
    const url = URL.createObjectURL(
      new Blob([payload], { type: "application/json" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `大富翁AI-第${game.round}轮.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function importSession(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const restored = readSessionSnapshot(String(reader.result), {
          deedTileIds: TILES.filter((tile) =>
            ["property", "station", "utility"].includes(tile.type),
          ).map((tile) => tile.id),
        });
        setGame(restored.game as GameState);
        setPolicy(restored.policy as Policy);
        setAuction(restored.auction);
        setTradeDraft(restored.tradeDraft as TradeDraft);
        setRejectedTradeProposalKeys(restored.rejectedTradeProposalKeys);
        setBuildDecisionCompleted(restored.buildDecisionCompleted);
        setToast("局面与 AI 模型已载入");
      } catch {
        setToast("文件格式不正确");
      }
    };
    reader.readAsText(file);
    event.target.value = "";
  }

  function renderTurnTradePanel() {
    return (
      <section className="state-section trade-section turn-trade-section">
        <div className="section-title compact">
          <strong className="filled-section-label">本回合交易</strong>
        </div>
        {myTurn &&
          (proactiveTradeProposal ? (
            <div className="proactive-trade-proposal">
              <span>AI 主动提案</span>
              <strong>{proactiveTradeProposal.summary}</strong>
              <div className="proactive-trade-actions">
                <button
                  type="button"
                  onClick={() => {
                    setTradeDraft(proactiveTradeProposal.draft);
                    setToast("AI 主动交易提案已填入");
                  }}
                >
                  填入报价
                </button>
                <button
                  type="button"
                  onClick={() =>
                    rejectProactiveTrade(proactiveTradeProposal)
                  }
                >
                  对方拒绝
                </button>
              </div>
            </div>
          ) : (
            <p className="proactive-trade-idle">
              AI 暂不建议主动发起交易
            </p>
          ))}
        <div className="trade-parties">
          {(["from", "to"] as const).map((side) => {
            const player = side === "from" ? activePlayer : tradeTo;
            const cash =
              side === "from" ? tradeDraft.fromCash : tradeDraft.toCash;
            const cards =
              side === "from" ? tradeDraft.fromCards : tradeDraft.toCards;
            const propertyIds =
              side === "from"
                ? tradeDraft.fromPropertyIds
                : tradeDraft.toPropertyIds;
            const ownedProperties = game.properties.filter(
              (property) => property.ownerId === player?.id,
            );
            return (
              <div className="trade-party" key={side}>
                {side === "from" ? (
                  <div className="trade-player-select">
                    <span>甲方</span>
                    <strong className="trade-player-fixed">
                      {activePlayer.name}
                    </strong>
                  </div>
                ) : (
                  <label className="trade-player-select">
                    <span>乙方</span>
                    <select
                      value={player?.id ?? ""}
                      onChange={(event) =>
                        changeTradeParticipant(
                          "to",
                          Number(event.target.value),
                        )
                      }
                    >
                      {game.players
                        .filter(
                          (candidate) =>
                            !candidate.bankrupt &&
                            candidate.id !== activePlayer.id,
                        )
                        .map((candidate) => (
                          <option key={candidate.id} value={candidate.id}>
                            {candidate.name}
                          </option>
                        ))}
                    </select>
                  </label>
                )}
                <div className="trade-value-grid">
                  <label>
                    <span>现金</span>
                    <input
                      type="number"
                      min={0}
                      value={cash}
                      onChange={(event) => {
                        const value = Math.max(
                          0,
                          Number(event.target.value),
                        );
                        setTradeDraft((current) => ({
                          ...current,
                          [side === "from" ? "fromCash" : "toCash"]: value,
                        }));
                      }}
                    />
                  </label>
                  <label>
                    <span>监狱通行证</span>
                    <input
                      type="number"
                      min={0}
                      max={player?.jailFreeCards ?? 0}
                      value={cards}
                      onChange={(event) => {
                        const value = Math.max(
                          0,
                          Number(event.target.value),
                        );
                        setTradeDraft((current) => ({
                          ...current,
                          [side === "from" ? "fromCards" : "toCards"]: value,
                        }));
                      }}
                    />
                  </label>
                </div>
                <div className="trade-property-list">
                  {ownedProperties.length ? (
                    ownedProperties.map((property) => {
                      const tile = namedTiles[property.tileId];
                      const blocked = groupHasBuildings(
                        tile,
                        game.properties,
                      );
                      return (
                        <label
                          key={tile.id}
                          className={blocked ? "blocked" : ""}
                        >
                          <input
                            type="checkbox"
                            checked={propertyIds.includes(tile.id)}
                            disabled={blocked}
                            onChange={() => toggleTradeProperty(side, tile.id)}
                          />
                          <i
                            style={{
                              background: tileStripColor(tile),
                            }}
                          />
                          <span>
                            {tile.short}
                            {property.mortgaged ? " · 已抵押" : ""}
                          </span>
                          {blocked && <em>先卖房</em>}
                        </label>
                      );
                    })
                  ) : (
                    <p>暂无地契</p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
        {tradeSuggestion ? (
          <div className="trade-ai-advice">
            <span>AI 交易建议</span>
            <strong>{tradeSuggestion.action}</strong>
          </div>
        ) : (
          <p className="trade-advice-placeholder">
            将“我”设为交易一方并录入报价后，AI 会给出建议。
          </p>
        )}
        <p className="trade-note">
          有房屋或酒店的同色组不可交易；接收已抵押地契时会自动支付抵押价 10% 的利息。
        </p>
        <button
          className="primary-button trade-submit"
          onClick={executeTrade}
          disabled={!tradeHasContents}
        >
          记录并执行交易
        </button>
      </section>
    );
  }

  function renderDeedModal() {
    if (!selectedDeedTile || !selectedDeed) return null;
    return (
      <div
        className="deed-modal-backdrop"
        onMouseDown={() => setSelectedDeedTileId(null)}
      >
        <section
          className={`deed-card deed-${selectedDeed.kind}`}
          role="dialog"
          aria-modal="true"
          aria-labelledby="deed-card-title"
          onMouseDown={(event) => event.stopPropagation()}
        >
          <button
            className="deed-card-close"
            aria-label="关闭地契卡"
            onClick={() => setSelectedDeedTileId(null)}
          >
            ×
          </button>
          {selectedDeedState?.mortgaged && (
            <span
              className="deed-mortgage-cross"
              role="img"
              aria-label="已抵押"
            />
          )}
          <header style={{ background: tileStripColor(selectedDeedTile) }}>
            <h2 id="deed-card-title">{selectedDeedTile.short}</h2>
          </header>
          <div className="deed-rent-table">
            {selectedDeed.kind === "property" && (
              <>
                {selectedDeed.rents.map((rent, index) => (
                  <div key={index}>
                    {index === 0 ? (
                      <span>地租</span>
                    ) : (
                      <span
                        className="deed-building-label"
                        role="img"
                        aria-label={
                          index === 5
                            ? "有 1 间酒店"
                            : `有 ${index} 栋房屋`
                        }
                      >
                        {Array.from(
                          { length: index === 5 ? 1 : index },
                          (_, buildingIndex) => (
                            <i
                              key={buildingIndex}
                              className={`deed-building-icon ${
                                index === 5 ? "hotel" : "house"
                              }`}
                              aria-hidden="true"
                            />
                          ),
                        )}
                      </span>
                    )}
                    <strong>{money(rent)}</strong>
                  </div>
                ))}
                <p>拥有完整同色组且尚未建房时，地租加倍。</p>
              </>
            )}
            {selectedDeed.kind === "station" &&
              selectedDeed.rents.map((rent, index) => (
                <div key={rent}>
                  <span>拥有 {index + 1} 条铁路</span>
                  <strong>{money(rent)}</strong>
                </div>
              ))}
            {selectedDeed.kind === "utility" && (
              <>
                <div>
                  <span>拥有 1 家公用事业</span>
                  <strong>骰子点数 × 4</strong>
                </div>
                <div>
                  <span>拥有 2 家公用事业</span>
                  <strong>骰子点数 × 10</strong>
                </div>
              </>
            )}
          </div>
          <footer>
            <div>
              <span>地块售价</span>
              <strong>{money(selectedDeedTile.price ?? 0)}</strong>
            </div>
            <div>
              <span>抵押价格</span>
              <strong>{money(selectedDeed.mortgage)}</strong>
            </div>
            {selectedDeed.kind === "property" && (
              <div>
                <span>每栋房屋／酒店价格</span>
                <strong>{money(selectedDeed.buildingCost)}</strong>
              </div>
            )}
          </footer>
        </section>
      </div>
    );
  }

  function renderReferenceModal() {
    if (!selectedReferenceTile) return null;
    const deck: CardDeck | null =
      selectedReferenceTile.type === "chance"
        ? "chance"
        : selectedReferenceTile.type === "community"
          ? "community"
          : null;
    const isIncomeTax = selectedReferenceTile.id === 4;
    if (!deck && !isIncomeTax) return null;
    const cards = deck
      ? CARD_DECKS[deck].flatMap((card) =>
          Array.from({ length: card.copies ?? 1 }, (_, copyIndex) => ({
            card,
            copyIndex,
          })),
        )
      : [];
    const title = deck ? CARD_DECK_LABELS[deck] : selectedReferenceTile.name;

    return (
      <div
        className="deed-modal-backdrop reference-modal-backdrop"
        onMouseDown={() => setSelectedReferenceTileId(null)}
      >
        <section
          className={`reference-modal ${deck ? `reference-modal-${deck}` : "reference-modal-tax"}`}
          role="dialog"
          aria-modal="true"
          aria-labelledby="reference-modal-title"
          onMouseDown={(event) => event.stopPropagation()}
        >
          <button
            className="deed-card-close"
            aria-label={`关闭${title}卡面`}
            onClick={() => setSelectedReferenceTileId(null)}
          >
            ×
          </button>
          <header>
            {deck && <span>卡牌总览</span>}
            <h2 id="reference-modal-title">{title}</h2>
          </header>
          {deck ? (
            <div className="reference-card-list">
              {cards.map(({ card, copyIndex }) => (
                <article
                  className={`reference-card reference-card-${deck}`}
                  key={`${card.id}-${copyIndex}`}
                >
                  <strong>{CARD_DECK_LABELS[deck]}</strong>
                  <p>{cardDisplayLabel(card, namedTiles)}</p>
                </article>
              ))}
            </div>
          ) : (
            <div className="tax-reference-card">
              <span>支付给银行</span>
              <strong>$200</strong>
              <p>落在所得税格时，向银行支付 $200。</p>
            </div>
          )}
        </section>
      </div>
    );
  }

  const pendingTile =
    game.pending?.kind === "property" ||
    game.pending?.kind === "rent"
      ? namedTiles[game.pending.tileId]
      : null;
  const pendingRentOwnerId =
    game.pending?.kind === "rent" ? game.pending.ownerId : null;
  const pendingRentOwner =
    pendingRentOwnerId === null
      ? null
      : game.players.find((player) => player.id === pendingRentOwnerId)?.name;

  return (
    <main className="app-shell">
      <section className="workspace">
        <section className="board-column" aria-label="棋盘与玩家">
          <div
            className="player-strip"
            style={
              {
                "--player-count": game.players.length,
                minWidth: `${Math.max(590, game.players.length * 118)}px`,
              } as CSSProperties
            }
          >
            {game.players.map((player) => {
              const isActive = player.id === game.activePlayerId;
              return (
                <button
                  key={player.id}
                  className={`player-card ${isActive ? "active" : ""} ${player.bankrupt ? "bankrupt" : ""}`}
                  onClick={() => setGame((state) => ({ ...state, activePlayerId: player.id }))}
                  disabled={player.bankrupt}
                  style={{ "--player-color": player.color } as CSSProperties}
                >
                  <span className="player-token">{player.name.slice(0, 1)}</span>
                  <span className="player-copy">
                    <strong>
                      {player.name}
                      {player.id === game.myPlayerId && <em>我</em>}
                    </strong>
                    <small>{money(player.cash)} · 净值 {money(playerNetWorth(player, game.properties))}</small>
                  </span>
                  {isActive && <span className="turn-dot">行动中</span>}
                </button>
              );
            })}
          </div>

          <div className="board-wrap">
            <div className="board">
              {namedTiles.map((tile) => {
                const property = ownerMap.get(tile.id);
                const owner =
                  property?.ownerId !== null && property?.ownerId !== undefined
                    ? game.players.find((p) => p.id === property.ownerId)
                    : null;
                const occupants = game.players.filter(
                  (p) => !p.bankrupt && p.position === tile.id,
                );
                const isCorner = [0, 10, 20, 30].includes(tile.id);
                const isDeedStyle =
                  tile.type === "property" ||
                  tile.type === "station" ||
                  tile.type === "utility";
                const isReferenceStyle =
                  tile.type === "chance" ||
                  tile.type === "community" ||
                  tile.id === 4;
                const isInspectable = isDeedStyle || isReferenceStyle;
                const visitingOccupants =
                  tile.type === "jail"
                    ? occupants.filter((player) => !player.inJail)
                    : [];
                const jailedOccupants =
                  tile.type === "jail"
                    ? occupants.filter((player) => player.inJail)
                    : [];
                return (
                  <div
                    key={tile.id}
                    className={`tile edge-${tileEdge(tile.id)} ${isCorner ? "corner" : ""} ${isDeedStyle ? "deed-tile" : "non-property"} ${isReferenceStyle ? "reference-tile" : ""} ${owner ? "owned" : ""} ${tile.id === 25 ? "long-name" : ""} type-${tile.type} ${activePlayer.position === tile.id ? "current" : ""}`}
                    style={tileGridPosition(tile.id)}
                    title={tile.name}
                    role={isInspectable ? "button" : undefined}
                    tabIndex={isInspectable ? 0 : undefined}
                    aria-label={
                      isDeedStyle
                        ? `查看${tile.short}地契卡`
                        : isReferenceStyle
                          ? `查看${tile.short}卡面`
                          : undefined
                    }
                    onClick={() => {
                      if (isDeedStyle) {
                        setSelectedReferenceTileId(null);
                        setSelectedDeedTileId(tile.id);
                      } else if (isReferenceStyle) {
                        setSelectedDeedTileId(null);
                        setSelectedReferenceTileId(tile.id);
                      }
                    }}
                    onKeyDown={(event) => {
                      if (
                        isInspectable &&
                        (event.key === "Enter" || event.key === " ")
                      ) {
                        event.preventDefault();
                        if (isDeedStyle) {
                          setSelectedReferenceTileId(null);
                          setSelectedDeedTileId(tile.id);
                        } else {
                          setSelectedDeedTileId(null);
                          setSelectedReferenceTileId(tile.id);
                        }
                      }
                    }}
                  >
                    {isDeedStyle && (
                      <span
                        className="color-band"
                        style={{ background: tileStripColor(tile) }}
                      >
                        {property?.houses ? (
                          <span
                            className="board-buildings"
                            aria-label={
                              property.houses === 5
                                ? "1 间酒店"
                                : `${property.houses} 栋房屋`
                            }
                          >
                            {Array.from({
                              length: property.houses === 5 ? 1 : property.houses,
                            }).map((_, index) => (
                              <i
                                key={index}
                                className={`board-building-icon ${
                                  property.houses === 5 ? "hotel" : "house"
                                }`}
                                aria-hidden="true"
                              />
                            ))}
                          </span>
                        ) : null}
                      </span>
                    )}
                    <span className="tile-copy">
                      {tile.type === "jail" ? (
                        <span className="jail-layout">
                          <span className="jail-label jail-label-visiting">探监</span>
                          <span className="jail-label jail-label-jailed">坐牢</span>
                        </span>
                      ) : (
                        <>
                          {!isDeedStyle && (
                            <span className="tile-icon" aria-hidden="true">{tileIcon(tile)}</span>
                          )}
                          <strong>
                            {tile.id === 25 ? (
                              <>
                                <span className="rail-name-line">巴尔的摩</span>
                                <span className="rail-name-line">与俄亥俄</span>
                                <span className="rail-name-line">铁路</span>
                              </>
                            ) : tile.short}
                          </strong>
                        </>
                      )}
                    </span>
                    {property?.mortgaged && (
                      <span
                        className="tile-mortgage-cross"
                        role="img"
                        aria-label="已抵押"
                      />
                    )}
                    {owner && (
                      <span
                        className="owner-mark"
                        style={{ "--owner-color": owner.color } as CSSProperties}
                        aria-label={`属于 ${owner.name}`}
                      />
                    )}
                    {tile.type === "jail" ? (
                      <>
                        {!!visitingOccupants.length && (
                          <span className="tokens jail-visiting-tokens" aria-label="探监玩家">
                            {visitingOccupants.map((p) => (
                              <i key={p.id} style={{ background: p.color }} title={`${p.name} · 探监`}>
                                {p.name.slice(0, 1)}
                              </i>
                            ))}
                          </span>
                        )}
                        {!!jailedOccupants.length && (
                          <span className="tokens jail-jailed-tokens" aria-label="坐牢玩家">
                            {jailedOccupants.map((p) => (
                              <i key={p.id} style={{ background: p.color }} title={`${p.name} · 坐牢`}>
                                {p.name.slice(0, 1)}
                              </i>
                            ))}
                          </span>
                        )}
                      </>
                    ) : !!occupants.length && (
                      <span className="tokens">
                        {occupants.map((p) => (
                          <i key={p.id} style={{ background: p.color }} title={p.name}>
                            {p.name.slice(0, 1)}
                          </i>
                        ))}
                      </span>
                    )}
                  </div>
                );
              })}
              <div className="board-center">
                <div className="board-hud">
                  <span aria-hidden="true" />
                  <div className="round-chip">
                    <span>ROUND</span>
                    <strong>{String(game.round).padStart(2, "0")}</strong>
                  </div>
                  <span className="save-state"><i /> 已在本机保存</span>
                </div>
                <div className="board-seal">
                  <strong>大富翁AI</strong>
                </div>
              </div>
            </div>
            {renderDeedModal()}
            {renderReferenceModal()}
          </div>
        </section>

        <aside className="control-panel">
          <nav className="tabs" aria-label="控制面板">
            <button className={tab === "turn" ? "active" : ""} onClick={() => setTab("turn")}>当前回合</button>
            <button className={tab === "state" ? "active" : ""} onClick={() => setTab("state")}>全局校准</button>
          </nav>

          {tab === "turn" && (
            <div className="panel-scroll">
              <section className="turn-heading">
                <span className="eyebrow">现在行动</span>
                <div className="turn-player">
                  <span style={{ background: activePlayer.color }}>{activePlayer.name.slice(0, 1)}</span>
                  <div>
                    <h1>{activePlayer.name} 的回合</h1>
                    <p>{currentTile.name} · 现金 {money(activePlayer.cash)}</p>
                  </div>
                </div>
              </section>

              <section className="model-status turn-model-status">
                <div className="model-orb"><strong>{leagueModel.generations}</strong><span>代</span></div>
                <div>
                  <span>当前部署策略</span>
                  <h2>
                    {game.players.length} 人专用神经进化模型
                  </h2>
                  <p>
                    训练 {leagueModel.trainingGames.toLocaleString("zh-CN")} 局 ·
                    独立验证胜率 {(leagueModel.winRateVsHeuristic * 100).toFixed(1)}%
                  </p>
                  <p className="standard-pass">
                    {leagueModel.achievedStandard?.passed &&
                    leagueModel.stoppingStandard ? (
                      <>
                        ✓ 自适应停止 · 95% 胜率下界{" "}
                        {(leagueModel.achievedStandard.winRateLowerBound * 100).toFixed(1)}%
                        {" · 平均名次 "}
                        {leagueModel.achievedStandard.averageRank.toFixed(2)}
                      </>
                    ) : (
                      <>
                        ✓ 自适应停止 · 独立验证胜率{" "}
                        {(leagueModel.winRateVsHeuristic * 100).toFixed(1)}%
                        {" · 平均名次 "}
                        {leagueModel.averageRankVsHeuristic.toFixed(2)}
                      </>
                    )}
                  </p>
                </div>
              </section>

              {game.gameOver && (
                <section className="turn-action-advice" role="status">
                  <span>牌局结束</span>
                  <strong>
                    {game.players.find((player) => player.id === game.winnerId)?.name ?? "最后一名玩家"}
                    获胜
                  </strong>
                </section>
              )}

              {game.debt && (
                <section className="turn-action-advice" role="alert">
                  <span>债务待处理</span>
                  <strong>
                    {game.players.find((player) => player.id === game.debt?.debtorId)?.name}
                    尚欠 {money(game.debt.amount)}，请在“全局校准”中抵押、售房、交易或宣布破产
                  </strong>
                </section>
              )}

              {!game.gameOver && !game.lastDice && !auction.open && (
                <section className="input-card">
                  <div className="section-title">
                    <strong className="filled-section-label">录入骰子</strong>
                  </div>
                  <div className="dice-row">
                    {[0, 1].map((index) => (
                      <label key={index} className="die-input">
                        <span>骰子 {index + 1}</span>
                        <select
                          value={dice[index]}
                          onChange={(event) => {
                            const next: [number, number] = [...dice];
                            next[index] = Number(event.target.value);
                            setDice(next);
                          }}
                          aria-label={`骰子 ${index + 1}`}
                        >
                          {[1, 2, 3, 4, 5, 6].map((value) => (
                            <option key={value}>{value}</option>
                          ))}
                        </select>
                      </label>
                    ))}
                    <div className="dice-total">
                      <span>合计</span>
                      <strong>{dice[0] + dice[1]}</strong>
                    </div>
                  </div>
                  {activePlayer.inJail && activePlayer.jailFreeCards > 0 && (
                    <button
                      className="jail-free-button"
                      onClick={useJailFreeCard}
                    >
                      使用监狱通行证（持有 {activePlayer.jailFreeCards} 张）
                    </button>
                  )}
                  {jailSuggestion && (
                    <div className="turn-action-advice">
                      <span>AI 建议</span>
                      <strong>{jailSuggestion.action}</strong>
                      {jailSuggestion.payToLeave && (
                        <button type="button" onClick={payToLeaveJail}>
                          支付 $50 出狱
                        </button>
                      )}
                    </div>
                  )}
                  <button className="primary-button" onClick={recordDice}>
                    确认并移动 <span>→</span>
                  </button>
                </section>
              )}

              {!game.gameOver && (game.lastDice || auction.open) && (
                <section className="landing-card">
                  <div className="landing-top">
                    {game.lastDice && (
                      <div className="mini-dice">
                        <span>{game.lastDice[0]}</span>
                        <span>{game.lastDice[1]}</span>
                      </div>
                    )}
                    <div>
                      <small>本回合落点</small>
                      <h2>{pendingTile?.name ?? namedTiles[activePlayer.position].name}</h2>
                    </div>
                  </div>
                  {game.pending?.kind === "property" && pendingTile && (
                    <>
                      <div className="property-facts">
                        <div><span>售价</span><strong>{money(pendingTile.price ?? 0)}</strong></div>
                        {pendingTile.type === "utility" ? (
                          <div><span>租金规则</span><strong>骰子点数 × 4</strong></div>
                        ) : (
                          <div><span>基础租金</span><strong>{money(pendingTile.rent ?? 0)}</strong></div>
                        )}
                        <div><span>购后现金</span><strong>{money(activePlayer.cash - (pendingTile.price ?? 0))}</strong></div>
                      </div>
                      {!auction.open ? (
                        <div className="decision-actions">
                          <button className="primary-button" onClick={() => resolvePending("buy")}>买入地块</button>
                          <button className="secondary-button" onClick={() => resolvePending("skip")}>放弃 / 竞拍</button>
                        </div>
                      ) : (
                        <div className="auction-box">
                          <div className="section-title compact">
                            <div><span>竞</span><strong>记录竞拍结果</strong></div>
                          </div>
                          {auctionSuggestion && (
                            <div className="auction-ai-advice">
                              <div className="auction-ai-head">
                                <strong>AI 拍卖建议</strong>
                              </div>
                              <h3>{auctionSuggestion.action}</h3>
                            </div>
                          )}
                          <div className="auction-fields">
                            <label>
                              <span>得主</span>
                              <select value={auction.winnerId} onChange={(e) => setAuction({ ...auction, winnerId: Number(e.target.value) })}>
                                {game.players.filter((p) => !p.bankrupt).map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                              </select>
                            </label>
                            <label>
                              <span>成交价</span>
                              <input type="number" min={1} value={auction.price} onChange={(e) => setAuction({ ...auction, price: Number(e.target.value) })} />
                            </label>
                          </div>
                          <div className="decision-actions">
                            <button className="primary-button" onClick={finishAuction}>确认成交</button>
                            <button className="secondary-button" onClick={finishAuctionWithoutWinner}>流拍</button>
                          </div>
                        </div>
                      )}
                    </>
                  )}
                  {game.pending?.kind === "rent" && (
                    <>
                      <p className="pending-copy">
                        向 {pendingRentOwner} 支付租金 <strong>{money(game.pending.amount)}</strong>
                      </p>
                      <button className="primary-button" onClick={() => resolvePending("confirm")}>确认支付</button>
                    </>
                  )}
                  {game.pending?.kind === "tax" && (
                    <>
                      <p className="pending-copy">{game.pending.label}：<strong>{money(game.pending.amount)}</strong></p>
                      <button className="primary-button" onClick={() => resolvePending("confirm")}>确认支出</button>
                    </>
                  )}
                  {game.pending?.kind === "card" && (
                    <>
                      <p className="pending-copy">
                        请选择桌面上实际抽到的{CARD_DECK_LABELS[game.pending.deck]}牌。程序不会随机代抽。
                      </p>
                      <label className="card-picker">
                        <span>实际牌面</span>
                        <select
                          value={selectedCardId}
                          onChange={(event) => setSelectedCardId(event.target.value)}
                        >
                          <option value="">请选择牌面…</option>
                          {CARD_DECKS[game.pending.deck].map((card) => (
                            <option key={card.id} value={card.id}>
                              {cardDisplayLabel(card, namedTiles)}
                            </option>
                          ))}
                        </select>
                      </label>
                      {needsUtilityDice && (
                        <label className="card-picker compact-card-picker">
                          <span>按牌面重新掷骰子的合计点数</span>
                          <input
                            type="number"
                            min={2}
                            max={12}
                            value={cardDiceTotal}
                            onChange={(event) =>
                              setCardDiceTotal(Number(event.target.value))
                            }
                          />
                        </label>
                      )}
                      <button
                        className="primary-button"
                        onClick={resolveSelectedCard}
                        disabled={!selectedCardId}
                      >
                        按所选牌面结算
                      </button>
                    </>
                  )}
                  {game.pending?.kind === "notice" && (
                    <>
                      <p className="pending-copy">{game.pending.label}</p>
                      <button className="primary-button" onClick={() => resolvePending("confirm")}>记下并继续</button>
                    </>
                  )}
                  {!game.pending && (
                    <>
                      {buildRecommendation && (
                        <div className="turn-action-advice">
                          <span>AI 建议</span>
                          <strong>{buildRecommendation.action}</strong>
                          <button
                            type="button"
                            onClick={() =>
                              executeRecommendedBuilds(
                                buildRecommendation.tileIds,
                              )
                            }
                          >
                            按建议建造
                          </button>
                        </div>
                      )}
                      <button className="primary-button" onClick={endTurn}>
                        {game.extraTurnEligible && !activePlayer.inJail
                          ? "开始额外回合"
                          : "结束回合"}
                      </button>
                    </>
                  )}
                </section>
              )}

              {myTurn && suggestion && !auction.open && (
                <section className="ai-card">
                  <div className="ai-label">
                    <span className="spark">✦</span>
                    <strong>AI 建议</strong>
                  </div>
                  <h2>{suggestion.action}</h2>
                </section>
              )}

              {!game.gameOver && renderTurnTradePanel()}

            </div>
          )}

          {tab === "state" && (
            <div className="panel-scroll state-panel">
              <section className="state-section">
                <div className="section-title compact">
                  <strong className="filled-section-label">玩家</strong>
                  <button
                    className="add-player-button"
                    onClick={addPlayer}
                    disabled={game.gameOver || game.players.length >= MAX_PLAYERS}
                  >
                    ＋ 添加玩家
                  </button>
                </div>
                <p className="player-count-note">
                  当前 {game.players.length} 人 · 选择“我”后启用建议
                </p>
                <div className="player-edit-list">
                  {game.players.map((player) => (
                    <article key={player.id} className={player.bankrupt ? "bankrupt" : ""}>
                      <div className="edit-player-head">
                        <span className="edit-avatar" style={{ background: player.color }}>{player.name.slice(0, 1)}</span>
                        <input
                          aria-label="玩家姓名"
                          autoComplete="off"
                          placeholder="输入名称"
                          value={player.name}
                          onChange={(e) => updatePlayer(player.id, { name: e.target.value })}
                        />
                        <select
                          className="player-name-preset"
                          aria-label={`为${player.name}选择预设姓名`}
                          title="选择 Monopoly Plus 中文姓名"
                          value=""
                          onChange={(e) => {
                            if (e.target.value) {
                              updatePlayer(player.id, { name: e.target.value });
                            }
                          }}
                        >
                          <option value="" disabled>选择</option>
                          {OFFICIAL_AI_NAMES.map((name) => (
                            <option key={name} value={name}>{name}</option>
                          ))}
                        </select>
                        <button
                          className={game.myPlayerId === player.id ? "me-button active" : "me-button"}
                          onClick={() => setGame((s) => ({ ...s, myPlayerId: player.id }))}
                        >{game.myPlayerId === player.id ? "这是我" : "设为我"}</button>
                        <button
                          className="remove-player-button"
                          onClick={() => removePlayer(player.id)}
                          disabled={game.gameOver || game.players.length <= MIN_PLAYERS}
                          aria-label={`移除${player.name}`}
                        >
                          ×
                        </button>
                      </div>
                      <div className="edit-grid">
                        <label><span>现金</span><input type="number" value={player.cash} onChange={(e) => updatePlayer(player.id, { cash: Number(e.target.value) })} /></label>
                        <label><span>位置</span><select value={player.position} onChange={(e) => updatePlayer(player.id, { position: Number(e.target.value) })}>{namedTiles.map((tile) => <option key={tile.id} value={tile.id}>{tile.id} · {tile.short}</option>)}</select></label>
                        <label><span>监狱通行证</span><input type="number" min={0} value={player.jailFreeCards} onChange={(e) => updatePlayer(player.id, { jailFreeCards: Math.max(0, Number(e.target.value)) })} /></label>
                      </div>
                      <button className="text-danger" disabled={game.gameOver || player.bankrupt} onClick={() => markBankrupt(player.id)}>标记破产</button>
                    </article>
                  ))}
                </div>
              </section>

              <section className="state-section trade-section">
                <div className="section-title compact">
                  <strong className="filled-section-label">交易</strong>
                </div>
                <div className="trade-parties">
                  {(["from", "to"] as const).map((side, sideIndex) => {
                    const player = side === "from" ? tradeFrom : tradeTo;
                    const cash =
                      side === "from"
                        ? tradeDraft.fromCash
                        : tradeDraft.toCash;
                    const cards =
                      side === "from"
                        ? tradeDraft.fromCards
                        : tradeDraft.toCards;
                    const propertyIds =
                      side === "from"
                        ? tradeDraft.fromPropertyIds
                        : tradeDraft.toPropertyIds;
                    const ownedProperties = game.properties.filter(
                      (property) => property.ownerId === player?.id,
                    );
                    return (
                      <div className="trade-party" key={side}>
                        <label className="trade-player-select">
                          <span>{sideIndex === 0 ? "甲方" : "乙方"}</span>
                          <select
                            value={player?.id ?? ""}
                            onChange={(event) =>
                              changeTradeParticipant(
                                side,
                                Number(event.target.value),
                              )
                            }
                          >
                            {game.players
                              .filter((candidate) => !candidate.bankrupt)
                              .map((candidate) => (
                                <option
                                  key={candidate.id}
                                  value={candidate.id}
                                  disabled={
                                    side === "from"
                                      ? candidate.id === tradeDraft.toId
                                      : candidate.id === tradeDraft.fromId
                                  }
                                >
                                  {candidate.name}
                                </option>
                              ))}
                          </select>
                        </label>
                        <div className="trade-value-grid">
                          <label>
                            <span>现金</span>
                            <input
                              type="number"
                              min={0}
                              value={cash}
                              onChange={(event) => {
                                const value = Math.max(
                                  0,
                                  Number(event.target.value),
                                );
                                setTradeDraft((current) => ({
                                  ...current,
                                  [side === "from"
                                    ? "fromCash"
                                    : "toCash"]: value,
                                }));
                              }}
                            />
                          </label>
                          <label>
                            <span>监狱通行证</span>
                            <input
                              type="number"
                              min={0}
                              max={player?.jailFreeCards ?? 0}
                              value={cards}
                              onChange={(event) => {
                                const value = Math.max(
                                  0,
                                  Number(event.target.value),
                                );
                                setTradeDraft((current) => ({
                                  ...current,
                                  [side === "from"
                                    ? "fromCards"
                                    : "toCards"]: value,
                                }));
                              }}
                            />
                          </label>
                        </div>
                        <div className="trade-property-list">
                          {ownedProperties.length ? (
                            ownedProperties.map((property) => {
                              const tile = namedTiles[property.tileId];
                              const blocked = groupHasBuildings(
                                tile,
                                game.properties,
                              );
                              return (
                                <label
                                  key={tile.id}
                                  className={blocked ? "blocked" : ""}
                                >
                                  <input
                                    type="checkbox"
                                    checked={propertyIds.includes(tile.id)}
                                    disabled={blocked}
                                    onChange={() =>
                                      toggleTradeProperty(side, tile.id)
                                    }
                                  />
                                  <i
                                    style={{
                                      background: tileStripColor(tile),
                                    }}
                                  />
                                  <span>
                                    {tile.short}
                                    {property.mortgaged ? " · 已抵押" : ""}
                                  </span>
                                  {blocked && <em>先卖房</em>}
                                </label>
                              );
                            })
                          ) : (
                            <p>暂无地契</p>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
                {tradeSuggestion && (
                  <div className="trade-ai-advice">
                    <span>AI 建议</span>
                    <strong>{tradeSuggestion.action}</strong>
                  </div>
                )}
                <p className="trade-note">
                  有房屋或酒店的同色组不可交易；接收已抵押地契时会自动支付抵押价 10% 的利息。
                </p>
                <button
                  className="primary-button trade-submit"
                  onClick={executeTrade}
                  disabled={!tradeHasContents}
                >
                  记录并执行交易
                </button>
              </section>

              <section className="state-section">
                <div className="section-title compact">
                  <strong className="filled-section-label">地契与房屋</strong>
                </div>
                <div className="asset-list">
                  {game.properties.map((property) => {
                    const tile = namedTiles[property.tileId];
                    const owner = game.players.find((p) => p.id === property.ownerId);
                    return (
                      <article key={tile.id}>
                        <span className="asset-band" style={{ background: tileStripColor(tile) }} />
                        <div className="asset-name"><strong>{tile.name}</strong><small>{money(tile.price ?? 0)}{property.mortgaged ? " · 已抵押" : ""}</small></div>
                        <select aria-label={`${tile.name}所有者`} value={property.ownerId ?? ""} onChange={(e) => setOwner(tile.id, e.target.value === "" ? null : Number(e.target.value))}>
                          <option value="">银行</option>
                          {game.players.filter((p) => !p.bankrupt).map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                        </select>
                        {owner && (
                          <div className="asset-actions">
                            {tile.type === "property" && (
                              <>
                                <button onClick={() => changeHouse(tile.id, -1)} disabled={property.houses === 0}>−</button>
                                <span>{property.houses === 5 ? "酒店" : `${property.houses} 房`}</span>
                                <button onClick={() => changeHouse(tile.id, 1)} disabled={property.houses === 5}>＋</button>
                              </>
                            )}
                            <button className="mortgage-button" onClick={() => toggleMortgage(tile.id)}>{property.mortgaged ? "赎回" : "抵押"}</button>
                          </div>
                        )}
                      </article>
                    );
                  })}
                </div>
              </section>

              <section className="file-actions">
                <button className="secondary-button" onClick={exportSession}>导出局面与模型</button>
                <button className="secondary-button" onClick={() => importRef.current?.click()}>载入存档</button>
                <input ref={importRef} type="file" accept=".json,application/json" hidden onChange={importSession} />
                <button className="danger-button" onClick={() => {
                  if (window.confirm("确定开始一局新游戏？当前存档会被覆盖。")) {
                    const fresh = createFreshSession(
                      createGame(game.players.length, true),
                      policy,
                    );
                    setGame(fresh.game as GameState);
                    setAuction(fresh.auction);
                    setTradeDraft(fresh.tradeDraft as TradeDraft);
                    setRejectedTradeProposalKeys(fresh.rejectedTradeProposalKeys);
                    setBuildDecisionCompleted(fresh.buildDecisionCompleted);
                    setToast("已开始新牌局");
                  }
                }}>新游戏</button>
              </section>
            </div>
          )}

        </aside>
      </section>
      {toast && <div className="toast" role="status">{toast}</div>}
    </main>
  );
}
