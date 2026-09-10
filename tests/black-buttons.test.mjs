import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL("../app/GameApp.tsx", import.meta.url),
  "utf8",
);
const css = await readFile(
  new URL("../app/globals.css", import.meta.url),
  "utf8",
);

test("keeps black action buttons flat and gives them one green hover state", () => {
  const primaryRule = css.match(/\.primary-button\s*\{([^}]*)\}/)?.[1] ?? "";
  const proposalRule =
    css.match(/\.proactive-trade-actions\s*>\s*button\s*\{([^}]*)\}/)?.[1] ??
    "";

  assert.match(primaryRule, /background:\s*var\(--ink\)/);
  assert.match(primaryRule, /box-shadow:\s*none/);
  assert.match(proposalRule, /background:\s*var\(--ink\)/);
  assert.match(proposalRule, /box-shadow:\s*none/);
  assert.match(
    css,
    /\.primary-button:hover:not\(:disabled\)[\s\S]*?background:\s*var\(--green\)/,
  );
  assert.match(
    css,
    /\.proactive-trade-actions\s*>\s*button:hover[\s\S]*?background:\s*var\(--green\)/,
  );
});

test("ends a turn without a decorative arrow", () => {
  const endTurnButton =
    source.match(
      /<button className="primary-button" onClick=\{endTurn\}>([\s\S]*?)<\/button>/,
    )?.[1] ?? "";

  assert.match(endTurnButton, /结束回合/);
  assert.doesNotMatch(endTurnButton, /→/);
});
