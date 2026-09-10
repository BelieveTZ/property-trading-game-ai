# MonopolyAI · 大富翁决策助手

`MonopolyAI` 是面向实体《Monopoly Plus》牌局的中文决策助手。玩家在网页中录入骰子、购买、拍卖、交易、抵押、建房和卡牌结果，应用维护完整局面，并使用对应人数的神经进化模型给出当前可执行的建议。

## 主要功能

- 支持 3–5 名玩家、监狱、连续双数、租金、税费、拍卖、破产和回合推进
- 支持地产交易、报价拒绝、同回合交易锁、抵押与均匀建房规则
- 可查看完整地产、机会、社会基金和所得税卡面
- 根据 3、4、5 人牌局自动选择独立训练的冻结模型
- 浏览器自动保存，也可导出和导入带版本的完整对局文件
- 提供可暂停、可恢复的零知识自博弈训练环境

## 架构

- `app/game-rules.mjs`：可测试的游戏状态转换与合法性校验
- `app/rent-rules.mjs` 与 `app/card-catalog.mjs`：可表驱动验证的租金算法和实体牌目录
- `app/session-state.mjs`：存档解析、迁移、校验和规范化
- `app/ai-advisor.mjs`：购买、拍卖、建造、出狱和交易决策
- `shared/board-rules.json`：网页、经典训练器和零知识环境共用的棋盘规则
- `shared/league-model.mjs`：冻结模型兼容的特征与动作定义
- `training/zero_knowledge/`：从规则与奖励出发的自博弈环境、模型和训练流程

重要架构决定记录在 [`docs/adr`](docs/adr)，领域术语与边界记录在 [`CONTEXT.md`](CONTEXT.md)。

## 本地运行

需要 Node.js `>=22.13.0`。

```bash
npm install
npm run dev
```

零知识训练依赖和命令见 [`training/zero_knowledge/README.md`](training/zero_knowledge/README.md)。训练产生的检查点、运行日志和本地工作目录不会提交到仓库。

## 质量检查

```bash
npm test
npm run lint
```

`npm test` 会依次执行 TypeScript 类型检查、生产构建、Node 行为测试以及 Python 环境与模型测试。也可分别运行 `npm run test:js` 和 `npm run test:python`。
