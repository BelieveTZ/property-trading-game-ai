import type { Metadata } from "next";
import GameApp from "./GameApp";

export const metadata: Metadata = {
  description:
    "录入实体牌局的骰子和决策，在关键回合获得明确的 AI 操作建议，并用本地自对弈持续训练策略。",
};

export default function Home() {
  return <GameApp />;
}
