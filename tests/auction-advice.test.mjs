import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("shows AI advice throughout a property auction", async () => {
  const source = await readFile(
    new URL("../app/GameApp.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /const auctionSuggestion = useMemo/);
  assert.match(source, /AI 拍卖建议/);
  assert.match(source, /参与竞拍，最高建议价/);
  assert.match(source, /退出竞拍，不再加价/);
  assert.doesNotMatch(source, /auctionSuggestion\.confidence/);
  assert.doesNotMatch(source, /auctionSuggestion\.reasons/);
  assert.match(source, /suggestion && !auction\.open/);
  assert.match(
    source,
    /!game\.gameOver && !game\.lastDice && !auction\.open/,
  );
  assert.match(
    source,
    /!game\.gameOver && \(game\.lastDice \|\| auction\.open\)/,
  );
});

test("keeps every visible AI recommendation action-only", async () => {
  const source = await readFile(
    new URL("../app/GameApp.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /type Suggestion = \{\s*action: string;\s*\}/);
  assert.doesNotMatch(source, /suggestion\.confidence/);
  assert.doesNotMatch(source, /suggestion\.reasons/);
  assert.doesNotMatch(source, /className="ai-foot"/);
});
