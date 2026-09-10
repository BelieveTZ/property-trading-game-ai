import type { Metadata } from "next";
import StandaloneGameApp from "./StandaloneGameApp";

export const metadata: Metadata = {
  description:
    "原创的单机地产交易棋盘游戏，支持实时决策建议、全 AI 观战与可复现的自对弈研究。",
};

export default function Home() {
  return <StandaloneGameApp />;
}
