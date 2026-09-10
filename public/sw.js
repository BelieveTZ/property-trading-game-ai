const CACHE_NAME = "property-trading-game-ai-shell-v2";
const STATIC_SHELL = ["/favicon.svg", "/manifest.webmanifest"];

async function precacheShell() {
  const cache = await caches.open(CACHE_NAME);
  const response = await fetch("/", { cache: "reload" });
  if (!response.ok) throw new Error("application shell could not be loaded");
  await cache.put("/", response.clone());
  const html = await response.text();
  const referencedAssets = [...html.matchAll(/(?:src|href)=["']([^"']+)["']/g)]
    .map((match) => new URL(match[1], self.location.origin))
    .filter((url) => url.origin === self.location.origin)
    .map((url) => `${url.pathname}${url.search}`);
  const assets = [...new Set([...STATIC_SHELL, ...referencedAssets])];
  await Promise.all(assets.map((path) => cache.add(path)));
}

self.addEventListener("install", (event) => {
  event.waitUntil(precacheShell());
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))),
    ),
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET" || new URL(request.url).origin !== self.location.origin) return;
  event.respondWith((async () => {
    const cached = await caches.match(request);
    if (cached) return cached;
    try {
      const response = await fetch(request);
      if (response.ok) {
        const cache = await caches.open(CACHE_NAME);
        await cache.put(request, response.clone());
      }
      return response;
    } catch {
      if (request.mode === "navigate") {
        return (await caches.match("/")) ?? Response.error();
      }
      return Response.error();
    }
  })());
});
