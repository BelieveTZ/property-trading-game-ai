import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("renders the property-trading game application shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>地产交易游戏AI<\/title>/);
  assert.match(html, /原创的单机地产交易棋盘游戏/);
  assert.match(html, /StandaloneGameApp-/);
  assert.match(html, /正在恢复本机牌局/);
  assert.doesNotMatch(html, /MonopolyAI|大富翁决策助手|Monopoly Plus/);
});
