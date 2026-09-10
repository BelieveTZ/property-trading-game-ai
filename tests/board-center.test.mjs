import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../app/GameApp.tsx", import.meta.url), "utf8");
const css = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");

test("board center is a clean, level title without decorative copy or frames", () => {
  const centerMarkup = source.match(
    /<div className="board-center">([\s\S]*?)<\/div>\s*<\/div>\s*<\/div>/,
  )?.[1] ?? "";

  assert.match(centerMarkup, /<strong>MonopolyAI<\/strong>/);
  assert.match(source, /<div className="board-hud">/);
  assert.match(source, /className="round-chip"/);
  assert.match(source, /className="save-state"/);
  assert.doesNotMatch(source, /className="topbar"/);
  assert.doesNotMatch(centerMarkup, /LOCAL STRATEGY ENGINE|记录局面|训练策略|解释每一步/);
  assert.doesNotMatch(source, /className="board-metrics"/);
  assert.doesNotMatch(css, /\.board-center::before|\.board-center::after/);

  const sealRule = css.match(/\.board-seal\s*\{([^}]*)\}/)?.[1] ?? "";
  assert.doesNotMatch(sealRule, /\bborder\s*:|\bbox-shadow\s*:|\btransform\s*:/);

  const hudRule = css.match(/\.board-hud\s*\{([^}]*)\}/)?.[1] ?? "";
  assert.match(hudRule, /top:\s*clamp\(38px,\s*4\.6vw,\s*64px\)/);
  assert.match(hudRule, /left:\s*50%/);
  assert.match(hudRule, /transform:\s*translateX\(-50%\)/);
  assert.doesNotMatch(hudRule, /\bright\s*:/);

  const workspaceRule = css.match(/\.workspace\s*\{([^}]*)\}/)?.[1] ?? "";
  const boardWrapRule = css.match(/\.board-wrap\s*\{([^}]*)\}/)?.[1] ?? "";
  const controlPanelRule =
    css.match(/\.control-panel\s*\{([^}]*)\}/)?.[1] ?? "";
  assert.match(workspaceRule, /min-height:\s*100vh/);
  assert.match(workspaceRule, /align-items:\s*stretch/);
  assert.match(boardWrapRule, /height:\s*calc\(100vh - 176px\)/);
  assert.match(controlPanelRule, /min-height:\s*100%/);
  assert.match(controlPanelRule, /display:\s*flex/);
});
