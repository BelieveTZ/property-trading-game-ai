import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { DEED_DETAILS, TILES } from "../app/board-catalog.mjs";
import { propertyRent } from "../app/rent-rules.mjs";
import { boardRuleFor } from "../shared/board-rules.mjs";

const source = await readFile(
  new URL("../app/GameApp.tsx", import.meta.url),
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
  assert.match(
    source,
    /onClick=\{\(\) => \{[\s\S]*?if \(isDeedStyle\) \{[\s\S]*?setSelectedDeedTileId\(tile\.id\)/,
  );
  assert.match(source, /role="dialog"/);
  assert.match(source, /aria-modal="true"/);
  assert.doesNotMatch(source, /<small>地契<\/small>/);
  assert.match(styles, /\.deed-modal-backdrop/);
  assert.match(styles, /\.deed-card/);
  assert.match(
    source,
    /<div className="board-wrap">[\s\S]*?\{renderDeedModal\(\)\}/,
  );
  assert.match(
    styles,
    /\.deed-modal-backdrop\s*\{[\s\S]*?position:\s*absolute/,
  );
  assert.match(source, /className="deed-building-label"/);
  assert.match(source, /index === 5 \? "hotel" : "house"/);
  assert.match(styles, /\.deed-building-icon\.house\s*\{[\s\S]*?#16864b/);
  assert.match(styles, /\.deed-building-icon\.hotel\s*\{[\s\S]*?#c9463b/);
  const houseIconRule = styles.match(
    /\.deed-building-icon\.house\s*\{[^}]*\}/,
  )?.[0];
  const hotelIconRule = styles.match(
    /\.deed-building-icon\.hotel\s*\{[^}]*\}/,
  )?.[0];
  assert.match(houseIconRule ?? "", /width:\s*14px/);
  assert.match(houseIconRule ?? "", /height:\s*10px/);
  assert.match(hotelIconRule ?? "", /width:\s*14px/);
  assert.match(hotelIconRule ?? "", /height:\s*10px/);
  assert.match(
    styles,
    /\.deed-card\.deed-station > header h2\s*\{[\s\S]*?color: #fff;/,
  );
  assert.doesNotMatch(source, /className="deed-card-status"/);
  assert.doesNotMatch(source, /持有人：|未抵押/);
  assert.match(source, /className="deed-mortgage-cross"/);
  assert.match(source, /className="tile-mortgage-cross"/);
  assert.match(
    styles,
    /\.deed-mortgage-cross\s*\{[\s\S]*?#cf2424/,
  );
  assert.match(
    styles,
    /\.tile-mortgage-cross\s*\{[\s\S]*?rgba\(210, 35, 35, 0\.88\)/,
  );
  const deedCardRule = styles.match(/\.deed-card\s*\{[^}]*\}/)?.[0];
  assert.ok(deedCardRule);
  assert.doesNotMatch(deedCardRule, /box-shadow/);
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
  assert.match(source, /<span>地块售价<\/span>/);
  assert.match(source, /money\(selectedDeedTile\.price \?\? 0\)/);
  assert.match(source, /抵押价格/);
  assert.match(source, /每栋房屋／酒店价格/);
});
