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

test("renders the Deed Advisor application shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>大富翁AI<\/title>/);
  assert.match(html, /<strong>大富翁AI<\/strong>/);
  assert.match(html, /当前回合/);
  assert.match(html, /当前部署策略/);
  assert.match(html, /专用神经进化模型/);
  assert.match(html, /起点/);
  assert.match(html, /机会/);
  assert.match(html, /社会基金/);
  assert.match(html, /免费停车/);
  assert.match(html, /电力公司/);
  assert.match(html, /自来水公司/);
  assert.match(html, /园区/);
  assert.match(html, /海滨大道/);
  assert.doesNotMatch(html, /地块名称主题/);
  assert.doesNotMatch(html, /上海|深圳|悉尼|合肥|贵阳/);
  assert.doesNotMatch(html, /brand-mark/);
  assert.doesNotMatch(html, /brand-name|brand-sub/);
  assert.doesNotMatch(html, /city-theme-chip/);
  assert.doesNotMatch(html, /局面管理/);
  assert.doesNotMatch(html, /AI 实验室/);
  assert.doesNotMatch(html, /react-loading-skeleton/);
});
