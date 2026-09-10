import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { DEED_DETAILS, TILES } from "../app/board-catalog.mjs";
import { propertyRent } from "../app/rent-rules.mjs";
import { boardRuleFor } from "../shared/board-rules.mjs";

const source = await readFile(
  new URL("../app/StandaloneGameApp.tsx", import.meta.url),
  "utf8",
);
const styles = await readFile(
  new URL("../app/globals.css", import.meta.url),
  "utf8",
);

test("opens a physical-style deed card from every purchasable board space", () => {
  const deeds = Object.values(DEED_DETAILS);
  assert.equal(deeds.filter((deed) => deed.kind === "property").length, 22);
  assert.equal(deeds.filter((deed) => deed.kind === "station").length, 4);
  assert.equal(deeds.filter((deed) => deed.kind === "utility").length, 2);
  assert.match(source, /onClick=\{\(\) => setSelectedTileId\(tile\.id\)\}/);
  assert.match(source, /role="dialog"/);
  assert.match(source, /aria-modal="true"/);
  assert.match(source, /className=\{property\.houses === 5 \? "hotel" : "houses"\}/);
  assert.match(styles, /\.standalone-tile > small\.hotel\s*\{[\s\S]*?#c9232d/);
  assert.doesNotMatch(source, /持有人：|未抵押/);
  assert.match(source, /className="deed-mortgage-cross"/);
  assert.match(source, /property\?\.mortgaged && <b aria-label="已抵押">×<\/b>/);
  assert.match(styles, /\.deed-dialog \.deed-mortgage-cross/);
});

test("uses exact classic rents and mortgage values in both display and settlement", () => {
  assert.deepEqual(boardRuleFor(1).rents, [2, 10, 30, 90, 160, 250]);
  assert.equal(boardRuleFor(1).mortgage, 30);
  assert.deepEqual(boardRuleFor(39).rents, [50, 200, 600, 1400, 1700, 2000]);
  assert.equal(boardRuleFor(39).mortgage, 200);
  assert.deepEqual(boardRuleFor(5).rents, [25, 50, 100, 200]);
  assert.equal(boardRuleFor(5).mortgage, 100);
  const deedState = (tileId, houses = 0, mortgaged = false) => ({
    tileId,
    ownerId: 0,
    houses,
    mortgaged,
  });
  assert.equal(
    propertyRent(
      TILES[12],
      deedState(12),
      [deedState(12), deedState(28)],
      7,
    ),
    70,
  );
  assert.equal(
    propertyRent(
      TILES[1],
      deedState(1),
      [deedState(1), deedState(3)],
    ),
    4,
  );
  assert.match(source, /地块售价 \{money\(Number\(selectedTile\.price \?\? 0\)\)\}/);
  assert.match(source, /抵押价值 \{money\(selectedDeed\.mortgage\)\}/);
  assert.match(source, /selectedDeed\.rents\.map/);
});
