import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { TILES } from "../app/board-catalog.mjs";
import { CARD_DECKS } from "../app/card-catalog.mjs";

test("requires manual entry for Chance and Community Chest cards", async () => {
  const source = await readFile(
    new URL("../app/GameApp.tsx", import.meta.url),
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
  assert.match(source, /请选择桌面上实际抽到的/);
  assert.match(source, /程序不会随机代抽/);
  assert.doesNotMatch(
    source,
    /const event = CARD_EVENTS\[/,
  );
});

test("includes the complete classic physical card effects", async () => {
  const rules = await readFile(
    new URL("../app/game-rules.mjs", import.meta.url),
    "utf8",
  );
  const cardText = [...CARD_DECKS.chance, ...CARD_DECKS.community]
    .map((card) => card.label)
    .join("\n");

  for (const expectedText of [
    "直达海滨大道",
    "前进至最近的铁路",
    "退后 3 格",
    "全面维修",
    "银行疏忽对你有利",
    "今天是你的生日",
    "维修街道",
    "监狱通行证",
  ]) {
    assert.match(cardText, new RegExp(expectedText));
  }
  assert.match(rules, /export function applyRecordedCard/);
});

test("uses the Monopoly Plus Simplified Chinese terminology", async () => {
  const source = await readFile(
    new URL("../app/GameApp.tsx", import.meta.url),
    "utf8",
  );

  const boardText = TILES.map((tile) => tile.name).join("\n");
  const cardText = [...CARD_DECKS.chance, ...CARD_DECKS.community]
    .map((card) => card.label)
    .join("\n");
  for (const officialText of [
    "社会基金",
    "巴尔提克大道",
    "康乃狄克大道",
    "史代兹大道",
    "维吉尼亚大道",
    "巴尔的摩与俄亥俄铁路",
    "北卡罗莱纳大道",
    "短程铁路",
    "园区",
    "海滨大道",
  ]) {
    assert.ok(boardText.includes(officialText));
  }

  for (const officialText of [
    "银行支付红利 $50。",
    "旅游基金到期，领取 $100。",
    "监狱通行证",
  ]) {
    assert.ok(cardText.includes(officialText));
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

  assert.match(
    source,
    /return `\$\$\{Math\.round\(value\)\.toLocaleString\("zh-CN"\)\}`/,
  );
  assert.doesNotMatch(source, /M\d+/);
});
