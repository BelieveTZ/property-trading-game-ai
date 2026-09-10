import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { TILES } from "../app/board-catalog.mjs";

test("keeps one fixed original-city board", async () => {
  const source = await readFile(
    new URL("../app/StandaloneGameApp.tsx", import.meta.url),
    "utf8",
  );
  assert.equal(TILES.length, 40);
  assert.doesNotMatch(source, /BoardTheme/);
  assert.doesNotMatch(source, /CITY_TILE_NAMES/);
  assert.doesNotMatch(source, /BOARD_THEME_ORDER/);
  assert.doesNotMatch(source, /boardTheme/);
  assert.equal(TILES[12].short, "城市能源");
  assert.equal(TILES[28].short, "城市水务");
  assert.equal(TILES[12].color, "#ffffff");
  assert.equal(TILES[28].color, "#ffffff");
  const fixedColors = {
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
    assert.equal(tile.color, fixedColors[tile.group]);
  }
});

test("makes the color band and tile labels substantially larger", async () => {
  const css = await readFile(
    new URL("../app/globals.css", import.meta.url),
    "utf8",
  );
  assert.match(css, /\.standalone-band\s*\{[\s\S]*?height: 22%/);
  assert.match(css, /font-size: clamp\(0\.55rem, 0\.76vw, 0\.8rem\)/);
});

test("keeps board color bands free of tile numbers", async () => {
  const source = await readFile(
    new URL("../app/StandaloneGameApp.tsx", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(source, /className="tile-index"/);
});
