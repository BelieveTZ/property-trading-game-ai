import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("keeps player avatars, board tokens and dice flat without shadows", async () => {
  const styles = await readFile(
    new URL("../app/globals.css", import.meta.url),
    "utf8",
  );
  const playerTokenRule =
    styles.match(/\.player-token\s*\{([^}]*)\}/)?.[1] ?? "";
  const boardTokenRule =
    styles.match(/\.tokens i\s*\{([^}]*)\}/)?.[1] ?? "";
  const diceRule =
    styles.match(/\.mini-dice span\s*\{([^}]*)\}/)?.[1] ?? "";
  const turnPlayerRule =
    styles.match(/\.turn-player > span\s*\{([^}]*)\}/)?.[1] ?? "";

  assert.match(playerTokenRule, /box-shadow:\s*none/);
  assert.match(playerTokenRule, /filter:\s*none/);
  assert.match(boardTokenRule, /box-shadow:\s*none/);
  assert.match(boardTokenRule, /filter:\s*none/);
  assert.match(diceRule, /box-shadow:\s*none/);
  assert.match(turnPlayerRule, /box-shadow:\s*none/);
  assert.match(turnPlayerRule, /filter:\s*none/);
});
