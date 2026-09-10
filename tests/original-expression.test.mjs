import assert from "node:assert/strict";
import test from "node:test";

import { TILES } from "../app/board-catalog.mjs";
import { CARD_DECK_LABELS, CARD_DECKS } from "../app/card-catalog.mjs";
import { OFFICIAL_AI_NAMES } from "../app/player-catalog.mjs";

test("the public game uses one original fictional-city expression", () => {
  assert.deepEqual(
    TILES.map((tile) => tile.name),
    [
      "启程", "旧港巷", "城市基金", "石桥街", "城市维护费",
      "北港枢纽", "晨雾路", "转机", "白帆街", "晴湾大道",
      "暂留所／访客", "榆光广场", "城市能源", "花汀路", "南栈大道",
      "东环枢纽", "陶谷广场", "城市基金", "炉心街", "金穗大道",
      "城市广场", "枫桥大道", "转机", "绯云路", "中央大道",
      "南湾枢纽", "日曜大道", "银杏路", "城市水务", "琥珀花园",
      "前往暂留所", "松涛大道", "环湖大道", "城市基金", "青岚大道",
      "西岭枢纽", "转机", "星河园区", "发展附加费", "天际大道",
    ],
  );
  assert.deepEqual(CARD_DECK_LABELS, { chance: "转机", community: "城市基金" });

  const publicText = [
    ...TILES.map((tile) => tile.name),
    ...CARD_DECKS.chance.map((card) => card.label),
    ...CARD_DECKS.community.map((card) => card.label),
    ...OFFICIAL_AI_NAMES,
  ].join("\n");

  assert.doesNotMatch(
    publicText,
    /海滨大道|伊利诺|圣查尔斯|雷丁铁路|巴尔提克|康乃狄克|宾夕法尼亚|大富翁|Monopoly|\$/i,
  );
  assert.match(publicText, /¤200/);
  assert.equal(new Set(OFFICIAL_AI_NAMES).size, OFFICIAL_AI_NAMES.length);
});
