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

版本化参数见 [`search_pipeline/config.json`](search_pipeline/config.json)。批量数据生成使用每次决策 32 次模拟、深度 24 的候选搜索；当合法动作超过预算时，按动作类型保留覆盖后再从其余动作中进行有种子的抽样，因此总搜索量不会随交易候选数量失控。高预算上限策略仍使用独立配置，不受批量生成预算影响。`SearchTeacher` 在每次模拟前重新采样未知卡序，不会沿用环境内已洗好的隐藏顺序。教师数据逐条保存游戏种子、教师种子、版本、搜索预算、决策序号和所选动作，可从同一配置复现。

可先用小预算生成一份可复现教师数据，验证完整管线：

```powershell
python -m training.search_pipeline.generate `
  --output training/runs/search-teacher/sample.jsonl `
  --run-dir training/runs/search-teacher `
  --players 4 `
  --games 8 `
  --simulations 32 `
  --depth 24 `
  --workers 4 `
  --checkpoint-every 32
```

生成器把每局保存为独立分片，由多个 CPU 进程并行生成，并按游戏编号确定性合并。每 32 个决策保存一次局内检查点；在运行目录创建 `pause.request` 后，各工作进程会在当前决策结束时安全暂停。再次执行同一命令会恢复未完成牌局并跳过已完成分片。搜索教师阶段累计在 `teacher_cpu_seconds`，不会误计入 GPU 训练里程碑。

随后可把教师分布蒸馏为兼容现有公开观测／候选动作接口的快速策略：

```powershell
python -m training.search_pipeline.distill `
  --dataset training/runs/search-teacher/sample.jsonl `
  --output training/runs/search-teacher/student.pt `
  --run-dir training/runs/search-teacher `
  --steps 20000 `
  --batch-size 256 `
  --checkpoint-every 100
```

蒸馏器先为 JSONL 建立小型随机访问索引，再按固定种子抽取批次；不会把完整教师数据同时展开到内存或显存。`distill-checkpoint.pt` 保存模型与优化器，重复执行相同命令会校验数据哈希、模型结构、批量大小、种子、抽样版本和优化器配置后继续未完成步数。`distill-state.json` 与 `distill-pid.txt` 用于查看进度。默认 20,000 步、批量 256，相当于在 306,753 条样本上约 16.7 个样本轮次；需要更多训练时可以提高总步数并从同一检查点延长。


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

正式里程碑训练使用实际运行时间预算，并把局数设置为只起安全上限作用的足够大数值。例如，从已完成的联盟候选继续训练到第一个 100 GPU 小时里程碑：

```powershell
python -m training.search_pipeline.train_league `
  --checkpoint training/runs/search-main/league-v1.pt `
  --opponent neuroevolution-v5=app/pretrained-model.json `
  --opponent zero-knowledge-ppo-v1=training/runs/baselines/zero-knowledge-ppo-v1.pt `
  --output training/runs/search-main-100h/league-100h.pt `
  --run-dir training/runs/search-main-100h `
  --games 1000000 `
  --players 3,4,5 `
  --gpu-hours 100
```

联盟训练每 24 局执行一次策略更新并释放该批决策样本；`league-state.json` 持续记录已完成局数和本次里程碑累计秒数。收到暂停请求时，检查点同时保存候选权重、待更新样本和累计时间，恢复后不会重算已经完成的牌局。达到时间预算后状态变为 `budget-reached`，并原子写出可直接用于独立评测的候选模型。

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
