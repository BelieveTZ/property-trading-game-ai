import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { TILES } from "../app/board-catalog.mjs";
import { CARD_DECKS } from "../app/card-catalog.mjs";

test("draws Chance and Community cards automatically from the seeded runtime", async () => {
  const source = await readFile(
    new URL("../app/standalone-session.mjs", import.meta.url),
    "utf8",
  );
  const rules = await readFile(
    new URL("../app/game-rules.mjs", import.meta.url),
    "utf8",
  );

  assert.match(
    rules,
    /next\.pending = \{ kind: "card", deck: tile\.type \}/,
  );
  assert.match(source, /function drawPendingCards\(session\)/);
  assert.match(source, /drawRuntimeCard\(next\.runtime, deck, CARD_DECKS\)/);
  assert.doesNotMatch(source, /桌面上实际抽到|手工录入/);
});

test("includes the complete fixed-rule card effects", async () => {
  const rules = await readFile(
    new URL("../app/game-rules.mjs", import.meta.url),
    "utf8",
  );
  const cardText = [...CARD_DECKS.chance, ...CARD_DECKS.community]
    .map((card) => card.label)
    .join("\n");

  for (const expectedText of [
    "直达天际大道",
    "最近的交通枢纽",
    "退后 3 格",
    "年度检修",
    "公共账户复核",
    "邻里开放日",
    "街区维护评估",
    "暂留所通行证",
  ]) {
    assert.match(cardText, new RegExp(expectedText));
  }
  assert.match(rules, /export function applyRecordedCard/);
});

test("uses original fictional-city terminology and currency", async () => {
  const source = await readFile(
    new URL("../app/StandaloneGameApp.tsx", import.meta.url),
    "utf8",
  );

  const boardText = TILES.map((tile) => tile.name).join("\n");
  const cardText = [...CARD_DECKS.chance, ...CARD_DECKS.community]
    .map((card) => card.label)
    .join("\n");
  for (const originalText of [
    "城市基金",
    "旧港巷",
    "晴湾大道",
    "南栈大道",
    "南湾枢纽",
    "环湖大道",
    "西岭枢纽",
    "星河园区",
    "天际大道",
  ]) {
    assert.ok(boardText.includes(originalText));
  }

  for (const originalText of [
    "项目红利，领取 ¤50。",
    "城市节庆补助到账，领取 ¤100。",
    "暂留所通行证",
  ]) {
    assert.ok(cardText.includes(originalText));
  }

  for (const previousText of [
    "公益金",
    "波罗的海大道",
    "公园广场",
    "木板路",
    "免费出狱卡",
  ]) {
    assert.doesNotMatch(`${boardText}\n${cardText}\n${source}`, new RegExp(previousText));
  }

  assert.match(source, /return `¤\$\{Math\.round\(value\)\.toLocaleString\("zh-CN"\)\}`/);
  assert.doesNotMatch(source, /M\d+/);
});
