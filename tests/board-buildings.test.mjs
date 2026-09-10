import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("shows green houses and a same-size red hotel directly on property bands", async () => {
  const [source, styles] = await Promise.all([
    readFile(new URL("../app/GameApp.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.match(
    source,
    /className="color-band"[\s\S]*?className="board-buildings"/,
  );
  assert.match(source, /property\.houses === 5 \? "hotel" : "house"/);
  assert.match(source, /length: property\.houses === 5 \? 1 : property\.houses/);
  assert.match(styles, /\.board-building-icon\s*\{[\s\S]*?#16864b/);
  assert.match(styles, /\.board-building-icon\.hotel\s*\{[\s\S]*?#c9463b/);
  assert.doesNotMatch(source, /"▰"|"▪"/);
});
