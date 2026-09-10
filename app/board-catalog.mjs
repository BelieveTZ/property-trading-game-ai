import {
  DEED_TILE_IDS,
  GROUP_ORDER,
  GROUP_TILES,
  boardRuleFor,
} from "../shared/board-rules.mjs";

export const UTILITY_COLOR = "#ffffff";

const GROUP_COLORS = Object.freeze({
  brown: "#98512a",
  sky: "#c0e5f6",
  pink: "#d8378a",
  orange: "#f39808",
  red: "#e30921",
  yellow: "#ffed07",
  green: "#0a943c",
  navy: "#096fb4",
  station: "#252922",
  utility: UTILITY_COLOR,
});

const TILE_PRESENTATION = [
  { name: "启程", type: "start" },
  { name: "旧港巷", type: "property" },
  { name: "城市基金", type: "community" },
  { name: "石桥街", type: "property" },
  { name: "城市维护费", type: "tax" },
  { name: "北港枢纽", type: "station" },
  { name: "晨雾路", type: "property" },
  { name: "转机", type: "chance" },
  { name: "白帆街", type: "property" },
  { name: "晴湾大道", type: "property" },
  { name: "暂留所／访客", type: "jail" },
  { name: "榆光广场", type: "property" },
  { name: "城市能源", type: "utility" },
  { name: "花汀路", type: "property" },
  { name: "南栈大道", type: "property" },
  { name: "东环枢纽", type: "station" },
  { name: "陶谷广场", type: "property" },
  { name: "城市基金", type: "community" },
  { name: "炉心街", type: "property" },
  { name: "金穗大道", type: "property" },
  { name: "城市广场", type: "parking" },
  { name: "枫桥大道", type: "property" },
  { name: "转机", type: "chance" },
  { name: "绯云路", type: "property" },
  { name: "中央大道", type: "property" },
  { name: "南湾枢纽", type: "station" },
  { name: "日曜大道", type: "property" },
  { name: "银杏路", type: "property" },
  { name: "城市水务", type: "utility" },
  { name: "琥珀花园", type: "property" },
  { name: "前往暂留所", type: "gotojail" },
  { name: "松涛大道", type: "property" },
  { name: "环湖大道", type: "property" },
  { name: "城市基金", type: "community" },
  { name: "青岚大道", type: "property" },
  { name: "西岭枢纽", type: "station" },
  { name: "转机", type: "chance" },
  { name: "星河园区", type: "property" },
  { name: "发展附加费", type: "tax" },
  { name: "天际大道", type: "property" },
];

export const TILES = Object.freeze(
  TILE_PRESENTATION.map((presentation, id) => {
    const rule = boardRuleFor(id);
    return Object.freeze({
      id,
      name: presentation.name,
      short: presentation.name,
      type: presentation.type,
      ...(rule
        ? {
            price: rule.price,
            rent: rule.baseRent,
            group: rule.group,
            color: GROUP_COLORS[rule.group],
            ...(rule.buildCost ? { houseCost: rule.buildCost } : {}),
          }
        : {}),
    });
  }),
);

export const DEED_DETAILS = Object.freeze(
  Object.fromEntries(
    DEED_TILE_IDS.map((tileId) => {
      const rule = boardRuleFor(tileId);
      if (rule.type === "property") {
        return [tileId, {
          kind: "property",
          rents: rule.rents,
          mortgage: rule.mortgage,
          buildingCost: rule.buildCost,
        }];
      }
      if (rule.type === "station") {
        return [tileId, { kind: "station", rents: rule.rents, mortgage: rule.mortgage }];
      }
      return [tileId, { kind: "utility", mortgage: rule.mortgage }];
    }),
  ),
);

export const GROUP_SIZES = Object.freeze(
  Object.fromEntries(GROUP_ORDER.map((group) => [group, GROUP_TILES[group].length])),
);

export const PROPERTY_GROUP_ORDER = Object.freeze(GROUP_ORDER.slice(0, 8));
