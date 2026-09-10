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
  { name: "起点", type: "start" },
  { name: "地中海大道", type: "property" },
  { name: "社会基金", type: "community" },
  { name: "巴尔提克大道", type: "property" },
  { name: "所得税", type: "tax" },
  { name: "雷丁铁路", type: "station" },
  { name: "东方大道", type: "property" },
  { name: "机会", type: "chance" },
  { name: "佛蒙特大道", type: "property" },
  { name: "康乃狄克大道", type: "property" },
  { name: "坐牢／探监", type: "jail" },
  { name: "圣查尔斯广场", type: "property" },
  { name: "电力公司", type: "utility" },
  { name: "史代兹大道", type: "property" },
  { name: "维吉尼亚大道", type: "property" },
  { name: "宾夕法尼亚铁路", type: "station" },
  { name: "圣詹姆斯广场", type: "property" },
  { name: "社会基金", type: "community" },
  { name: "田纳西大道", type: "property" },
  { name: "纽约大道", type: "property" },
  { name: "免费停车", type: "parking" },
  { name: "肯塔基大道", type: "property" },
  { name: "机会", type: "chance" },
  { name: "印第安那大道", type: "property" },
  { name: "伊利诺大道", type: "property" },
  { name: "巴尔的摩与俄亥俄铁路", type: "station" },
  { name: "大西洋大道", type: "property" },
  { name: "文特诺大道", type: "property" },
  { name: "自来水公司", type: "utility" },
  { name: "马文花园", type: "property" },
  { name: "进监狱", type: "gotojail" },
  { name: "太平洋大道", type: "property" },
  { name: "北卡罗莱纳大道", type: "property" },
  { name: "社会基金", type: "community" },
  { name: "宾夕法尼亚大道", type: "property" },
  { name: "短程铁路", type: "station" },
  { name: "机会", type: "chance" },
  { name: "园区", type: "property" },
  { name: "奢侈税", type: "tax" },
  { name: "海滨大道", type: "property" },
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
