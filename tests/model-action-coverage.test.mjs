import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL("../app/StandaloneGameApp.tsx", import.meta.url),
  "utf8",
);
const trainer = await readFile(
  new URL("../training/league-trainer.mjs", import.meta.url),
  "utf8",
);
const advisor = await readFile(
  new URL("../app/ai-advisor.mjs", import.meta.url),
  "utf8",
);
const standalonePolicy = await readFile(
  new URL("../app/standalone-policy.mjs", import.meta.url),
  "utf8",
);

test("exposes every trained neural action in the turn workflow", () => {
  assert.match(
    trainer,
    /actionNames: \[\.\.\.LEAGUE_ACTION_NAMES\]/,
  );

  assert.match(advisor, /LEAGUE_OUTPUT\.buy/);
  assert.match(advisor, /auctionCeiling\(/);
  assert.match(advisor, /LEAGUE_OUTPUT\.build/);
  assert.match(advisor, /LEAGUE_OUTPUT\.leaveJail/);
  assert.match(advisor, /LEAGUE_OUTPUT\.cashReserve/);
  assert.match(standalonePolicy, /automaticNegotiationAction\(/);
  assert.match(source, /legalManagementActions\(/);
  assert.match(source, />资产管理</);
  assert.match(source, /支付 ¤50 后离开/);
  assert.match(source, /trade-counter/);
});
