import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const css = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");

test("keeps the board flat and player fields contained", () => {
  const boardRule = css.match(/\.board\s*\{([^}]*)\}/)?.[1] ?? "";
  assert.doesNotMatch(boardRule, /box-shadow/);
  const modelStatusRule =
    css.match(/\.model-status\s*\{([^}]*)\}/)?.[1] ?? "";
  assert.doesNotMatch(modelStatusRule, /box-shadow/);

  const gridRule = css.match(
    /\.auction-fields,\s*\.edit-grid\s*\{([^}]*)\}/,
  )?.[1] ?? "";
  assert.match(
    gridRule,
    /grid-template-columns:\s*minmax\(0, 1fr\) minmax\(0, 1fr\)/,
  );

  const fieldRule = css.match(
    /\.auction-fields input,[\s\S]*?\.edit-grid select\s*\{([^}]*)\}/,
  )?.[1] ?? "";
  assert.match(fieldRule, /width:\s*100%/);
  assert.match(fieldRule, /max-width:\s*100%/);
});

test("keeps every trade control inside its party and section borders", () => {
  assert.match(css, /\.trade-section,\s*\.trade-section \* \{\s*box-sizing: border-box;/);
  assert.match(css, /\.trade-parties \{[\s\S]*?grid-template-columns: repeat\(2, minmax\(0, 1fr\)\);/);
  assert.match(css, /\.trade-party \{[\s\S]*?width: 100%;[\s\S]*?overflow: hidden;/);
  assert.match(css, /\.trade-player-select select,[\s\S]*?width: 100%;[\s\S]*?max-width: 100%;/);
  assert.match(css, /\.trade-value-grid \{[\s\S]*?grid-template-columns: minmax\(0, 1fr\);/);
});
