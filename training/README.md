# 训练与评测

训练主线以一个月约 450–600 GPU 小时内的独立评测表现为目标。策略只能看到公开牌局状态和当前合法动作，不能读取已洗好但尚未揭示的牌序、骰子结果或随机数状态。

## 当前组成

- `search_pipeline/`：随机搜索教师、公开信息信念采样、席位隔离状态、均衡座位评测与 100/300/500 GPU 小时冻结流程。
- `zero_knowledge/`：从终局结果学习的 PPO 基准；其现有检查点冻结为对照组。
- `league-trainer.mjs`：早期神经进化基准；继续用于回归评测，不再代表主训练路线。

## 主训练路线

1. 在 3、4、5 人固定种子牌局上运行随机搜索教师，保存公开观测、合法候选动作、教师分布和终局结果。
2. 用教师分布和终局价值蒸馏快速策略／价值网络。
3. 将候选策略加入历史联盟，以均衡座位继续自对弈强化学习。
4. 在约 100、300、500 GPU 小时冻结候选并独立评测；只有通过门禁的候选才进入应用。
5. 共享模型权重，但每个席位使用独立记忆、奖励累计和采样随机流。

版本化参数见 [`search_pipeline/config.json`](search_pipeline/config.json)。`SearchTeacher` 在每次模拟前重新采样未知卡序，不会沿用环境内已洗好的隐藏顺序。教师数据逐条保存游戏种子、教师种子、版本、搜索预算、决策序号和所选动作，可从同一配置复现。

可先用小预算生成一份可复现教师数据，验证完整管线：

```powershell
python -m training.search_pipeline.generate `
  --output training/runs/search-teacher/sample.jsonl `
  --run-dir training/runs/search-teacher `
  --players 4 `
  --games 1 `
  --simulations 16 `
  --depth 24
```

生成器每局后原子保存进度；在运行目录创建 `pause.request` 后会在当前牌局结束时安全暂停，再次执行同一命令会从已完成局数继续。

随后可把教师分布蒸馏为兼容现有公开观测／候选动作接口的快速策略：

```powershell
python -m training.search_pipeline.distill `
  --dataset training/runs/search-teacher/sample.jsonl `
  --output training/runs/search-teacher/student.pt `
  --steps 200
```

将蒸馏策略与一个或多个冻结检查点进行联盟自对弈微调；每个席位拥有独立的循环记忆、累计奖励和采样随机流：

```powershell
python -m training.search_pipeline.train_league `
  --checkpoint training/runs/search-teacher/student.pt `
  --opponent neuroevolution-v5=app/pretrained-model.json `
  --opponent zero-knowledge-ppo-v1=training/runs/baselines/zero-knowledge-ppo-v1.pt `
  --output training/runs/search-main/league-v1.pt `
  --games 24 `
  --players 3,4,5
```

使用固定种子、3/4/5 人局和均衡候选座位运行独立评测，并在达到 100、300、500 GPU 小时时冻结检查点：

```powershell
python -m training.search_pipeline.evaluate `
  --checkpoint training/runs/search-teacher/student.pt `
  --run-dir training/runs/search-main `
  --gpu-hours 100 `
  --players 3,4,5 `
  --seeds 20260910,20260911,20260912,20260913 `
  --opponent neuroevolution-v5=app/pretrained-model.json `
  --opponent zero-knowledge-ppo-v1=training/runs/baselines/zero-knowledge-ppo-v1.pt
```

评测会写出完整席位安排、候选及对手文件哈希、模型版本、胜率、平均名次、Wilson 95% 区间、平均／P95 决策时间和非法动作数；候选座位与各对手所在的物理座位都会轮换，非法动作不为 0 的候选不会成为最佳候选。历史对手清单由 `LeagueRoster` 版本化保存。旧神经进化 JSON 会按文件内的 3/4/5 人元数据加载对应冻结模型，缺少匹配人数时直接拒绝评测；零知识 PPO 基准的本地冻结副本和 SHA-256 清单位于被 Git 忽略的 `training/runs/baselines/`。没有提供 `--opponent` 时只运行合法性烟雾基线，不能作为正式强度报告。

上限策略由 `config.json` 的 `policies.upperBound` 搜索预算直接驱动，可用相同的固定种子和冻结对手单独评测：

```powershell
python -m training.search_pipeline.evaluate_upper_bound `
  --output training/runs/search-main/upper-bound.json `
  --opponent neuroevolution-v5=app/pretrained-model.json `
  --opponent zero-knowledge-ppo-v1=training/runs/baselines/zero-knowledge-ppo-v1.pt
```

上限报告还会固化教师随机种子、搜索模拟次数、深度、配置版本，以及所有对手检查点的路径和哈希，因而可核对到确切输入产物。

## 基准训练

本机环境安装及 PPO 的启动、暂停、恢复命令见 [`zero_knowledge/README.md`](zero_knowledge/README.md)。训练目录、检查点和运行日志均在忽略列表中，不进入源码仓库。

## 评测报告要求

每次正式比较都记录模型版本、对手版本、玩家人数、固定种子集合、座位分布与搜索预算，并报告：

- 胜率与 95% Wilson 置信区间；
- 平均名次；
- 平均与高分位决策时间；
- 非法动作数（发布门槛为 0）。

训练内胜率不等同于对真人或任意外部策略的通用胜率。
