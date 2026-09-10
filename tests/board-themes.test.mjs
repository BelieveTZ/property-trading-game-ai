import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { TILES } from "../app/board-catalog.mjs";

test("keeps only the classic Atlantic City board", async () => {
  const source = await readFile(
    new URL("../app/GameApp.tsx", import.meta.url),
    "utf8",
  );
  assert.deepEqual(
    Array.from(TILES, (tile) => tile.name),
    [
      "起点",
      "地中海大道",
      "社会基金",
      "巴尔提克大道",
      "所得税",
      "雷丁铁路",
      "东方大道",
      "机会",
      "佛蒙特大道",
      "康乃狄克大道",
      "坐牢／探监",
      "圣查尔斯广场",
      "电力公司",
      "史代兹大道",
      "维吉尼亚大道",
      "宾夕法尼亚铁路",
      "圣詹姆斯广场",
      "社会基金",
      "田纳西大道",
      "纽约大道",
      "免费停车",
      "肯塔基大道",
      "机会",
      "印第安那大道",
      "伊利诺大道",
      "巴尔的摩与俄亥俄铁路",
      "大西洋大道",
      "文特诺大道",
      "自来水公司",
      "马文花园",
      "进监狱",
      "太平洋大道",
      "北卡罗莱纳大道",
      "社会基金",
      "宾夕法尼亚大道",
      "短程铁路",
      "机会",
      "园区",
      "奢侈税",
      "海滨大道",
    ],
  );
  assert.doesNotMatch(source, /BoardTheme/);
  assert.doesNotMatch(source, /CITY_TILE_NAMES/);
  assert.doesNotMatch(source, /BOARD_THEME_ORDER/);
  assert.doesNotMatch(source, /boardTheme/);
  assert.equal(TILES[12].short, "电力公司");
  assert.equal(TILES[28].short, "自来水公司");
  assert.equal(TILES[12].color, "#ffffff");
  assert.equal(TILES[28].color, "#ffffff");
  const officialColors = {
    brown: "#98512a",
    sky: "#c0e5f6",
    pink: "#d8378a",
    orange: "#f39808",
    red: "#e30921",
    yellow: "#ffed07",
    green: "#0a943c",
    navy: "#096fb4",
  };
  for (const tile of TILES.filter((tile) => tile.type === "property")) {
    assert.equal(tile.color, officialColors[tile.group]);
  }
});

test("makes the color band and tile labels substantially larger", async () => {
  const css = await readFile(
    new URL("../app/globals.css", import.meta.url),
    "utf8",
  );
  assert.match(css, /grid-template-rows: 20% 80%/);
  assert.match(
    css,
    /\.tile\.edge-left\.deed-tile,\s*\.tile\.edge-right\.deed-tile \{\s*grid-template-rows: 27% 73%;/,
  );
  assert.match(css, /font-size: clamp\(11px, 1\.16vw, 18px\)/);
  assert.doesNotMatch(css, /\.tile-copy small/);
  assert.doesNotMatch(css, /\.tile-index\b/);
});

test("keeps board color bands free of tile numbers", async () => {
  const source = await readFile(
    new URL("../app/GameApp.tsx", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(source, /className="tile-index"/);
});
