import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "地产交易游戏AI",
    short_name: "地产交易AI",
    description: "原创的离线地产交易棋盘游戏与可复现自对弈研究项目。",
    start_url: "/",
    display: "standalone",
    background_color: "#101914",
    theme_color: "#101914",
    lang: "zh-CN",
    icons: [
      {
        src: "/favicon.svg",
        sizes: "any",
        type: "image/svg+xml",
      },
    ],
  };
}
