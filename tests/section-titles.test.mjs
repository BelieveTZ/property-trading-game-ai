import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../app/GameApp.tsx", import.meta.url), "utf8");
const css = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");

test("uses full black labels for dice entry, players and property management", () => {
  for (const label of ["录入骰子", "玩家", "地契与房屋"]) {
    assert.match(
      source,
      new RegExp(`<strong className="filled-section-label">${label}</strong>`),
    );
  }

  assert.doesNotMatch(
    source,
    /<span>01<\/span>|<span>记<\/span>|<span>人<\/span>|<span>契<\/span>/,
  );
  assert.doesNotMatch(source, /className="history"|>历史记录<\/strong>/);

  const labelRule = css.match(
    /\.section-title > \.filled-section-label\s*\{([^}]*)\}/,
  )?.[1] ?? "";
  assert.match(labelRule, /background:\s*var\(--ink\)/);
  assert.match(labelRule, /color:\s*#fff/);
});
