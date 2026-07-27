# Wan2.2 8×USP1 调度算法迭代交接

更新时间：2026-07-27

## 当前目标与统一口径

- 目标：最小化 Wan2.2 的纯 E2E P95。
- 当前只搜索 8×USP1 上的调度算法；先仿真，再做 req=50 NPU 验证，最后才做 req=100。
- 调度器只能使用在线可见信息：已到达请求、Wan2.2 estimator、backend 当前状态和有界历史；不能知道未来请求或最终请求总数。
- 当前不做 Normal 抢占，Tail 预算规则不变。
- 直接基准统一为实测 2×USP4：
  - req=50：P95 = 2191.560s。
  - req=100：P95 = 3010.310s。
- 假设的原始 baseline 为 `P95(2×USP4) / 0.7`：
  - req=50：3130.800s。
  - req=100：4300.443s。

## 已完成结果

| 方案 | 请求数 | 类型 | P95 | 相对 2×USP4 P95 减少 | 相对 2×USP4 加速比 | 相对假设 baseline P95 减少 |
|---|---:|---|---:|---:|---:|---:|
| Central Pull + β=0.85 + Tail Pack + Tail Gate | 50 | NPU | 1746.806s | 20.294% | 1.255× | 44.206% |
| Central Pull + Queue-Band Risk + Tail Pack | 50 | NPU | 1716.099s | 21.695% | 1.277× | 45.187% |
| Central Pull + Tail-aware Release-Calendar Beam | 50 | NPU | 1714.909s | 21.749% | 1.278× | 45.225% |
| Central Pull + β=0.85 + Tail Pack + Tail Gate | 100 | NPU | 2177.259s | 27.673% | 1.383× | 49.372% |

Queue-Band req=50 的完整实验：

- 实验名：`wan22_queue_band_req50_20260727_164742`
- 50/50 成功，无抢占。
- Mean 1249.033s，Median 1349.571s，P95 1716.099s，P99 2916.994s。
- 对此前 Tail Gate 的 P95 额外改善 1.758%。
- 同一 trace 的仿真 P95 为 1715.462s，与实测相差 0.637s（0.037%）。
- 本地完整 artifact：`/tmp/wan22_queue_band_req50_20260727_164742_artifact.tar.gz`
- 本地解包目录：`/tmp/wan22_queue_band_req50_20260727_164742_artifact_local`

## 当前 NPU 结论

当前真实 incumbent 仍是 **Central Pull + Queue-Band Risk + Tail Pack**。
Release-Calendar Beam 在仿真中明显更好，但 req=50 NPU 只把 P95 从
1716.099s 降到 1714.909s：

- 相对 Queue-Band 只减少 1.190s / 0.069%，加速比 1.0007×。
- 50/50 成功；48 Normal / 2 Tail；没有抢占。
- Mean 1237.171s，Median 1282.497s，P99 2917.232s。
- 相对 2×USP4 减少 21.749%，加速比 1.278×。
- 相对假设 baseline 减少 45.225%，加速比 1.826×。

实验名为 `wan22_tail_aware_release_calendar_beam_req50_20260727_103414`。
本地 artifact 为
`/tmp/wan22_tail_aware_release_calendar_beam_req50_20260727_103414_artifact.tar.gz`，
SHA256 为
`5bf672d915aebdc6bdbf4eb7139caa27b9ce215050ab3a540c5134270937e7e7`。

该次启动曾命中 Queue-Band 实验残留的 8 个 backend。benchmark 使用的模型、
拓扑和 backend scheduler 相同，按本次 client 时间窗恢复出的 48 个 Normal
和 2 个 Tail trace 完整，结果可用于比较；残留进程随后已精确清理。为防止
再次发生，提交 `a53639c6` 增加了启动前端口占用检查，已推送并在服务器通过
26 项 dispatcher 测试。

## 仿真与真实差距

输入并没有错：

- 50 个请求类型、顺序和全局到达时间与 benchmark 对齐；最大到达误差 0.044s。
- estimator 都是 110.724s / 219.548s / 612.299s。
- 两个 Tail 都是 request-00018、request-00038。

差距来自逐请求服务残差和 Beam 对释放顺序的敏感性。把真实 Normal 占用时间
回放进仿真后，两种策略各自的 48 个 Normal dispatch 顺序和 backend 都与
真实 trace 100% 一致：

| 回放口径 | Queue-Band P95 | Beam P95 | Beam P95 减少 |
|---|---:|---:|---:|
| synthetic seed=42 | 1718.310s | 1686.013s | 1.880% |
| 各自真实服务序列 | 1716.288s | 1715.175s | 0.0648% |
| NPU 实测 | 1716.099s | 1714.909s | 0.0693% |

首个分叉发生在第 15 次 pull：真实执行中一个 backend 的 estimator ETA 已被
截为 0，但它实际上还需要约 21.9s 才释放。只给仿真中的一个 long 请求增加
3s，就会跨过这个离散边界、改变选择，并让 P95 跳高约 9.8s。因此后续候选
除了多种子仿真，还必须通过两条真实服务残差路径回放。

## 剩余排序空间

固定两个 Tail、8 个 backend 和无抢占，用 Queue-Band 实测的逐请求有效服务
时间做离线诊断：

- Queue-Band 重建 P95 为 1716.151s，与实测只差 0.052s。
- 全 trace 的 CP-SAT best feasible 为 1680.913s，相对实测可再减少
  35.187s / 2.050%；求解下界仍很松，不能把它称为最优值。
- 冻结 813.095s 之前已经开始的 17 个 Normal，只重排当时剩余的 31 个
  可见请求，仍可得到 1683.130s，相对实测减少 32.969s / 1.921%。

这说明主要空间在后半段 backlog 的 8 条 backend 链配平，而不是 Tail 预算。
该结果使用了完整真实服务时间，只用于证明存在空间，不能当作在线算法收益。

## 已否决的算法

| 算法 | 结论 |
|---|---|
| Stable P95 Pairwise Controller | req50/100 均弱于 Beam |
| Quantile-Boundary Min-Cost Matching | seed=42 不改变 Queue-Band 动作 |
| Quantile Shadow-Price | req50 回退 16.413%，req100 回退 9.781% |
| Quantile-Critical Branch-and-Bound | 稳定弱于 Beam且规划更贵 |
| P95 Critical-Chain Local Search | 与 Beam 基本相同，seed=42 req100 回退 |
| Online Scenario Rollout | req50 仅改善 0.053%，req100 无收益，开销增加 31%–39% |
| Online Residual-Calibrated Calendar | 相对 Beam 仅改善约 0.1%，无偏估时下无收益 |
| Robust Counterfactual Beam Gate | 两条真实残差路径相对 Queue-Band 均改善 0.741%–0.806%，但相对 Beam 一胜一负，未通过稳定性门槛 |
| Dynamic Tail Reselection | req50 多种子无显著收益；真实残差回放回退 2.5%–3.0% |
| Provisional Shoulder Tail | req50 无显著收益；只有 dispatcher-held Tail 可以交换 |
| Censored-ETA Release Calendar | req50 回退 0.128%；req100 置信区间跨 0；两条真实回放符号相反 |
| Near-Release Joint Matching | req50/100 相对 Beam 回退 0.359% / 0.304%，规划开销更高 |
| Release Coalescing | 有界等待不触发；强制等待约 29s 后 P95 不变 |
| P95 Deadline/Laxity Dispatch | req50 最好仅改善 0.235%；两条真实残差路径为 0% / 回退 1.412%，局部 deadline 会把风险转移给后续请求 |
| 在线 Backend×输入类型亲和性学习 | 卡间整体速度差稳定，但 class 交互仅 0.2%–0.46%；相对 Beam 的 req50/100 收益均约 0%，两条真实回放均更差 |
| 全局可见队列逐-pull 重规划 | req50/100 相对 Queue-Band 降低 2.938% / 2.509%，但两条真实残差路径只降低 0.814% / 0.061%，均弱于波次锁定；首次 8-request 计划实际位置只命中 1/8，计划抖动过大 |

这些方案均不进入 NPU。

## 当前 NPU 候选

**积压配平 + 波次顺序锁定**：

1. 当可见 Normal backlog 至少有 16 个请求时，以 Queue-Band drain
   作为初始可行计划。
2. 对全部当前可见请求做跨 8 条 release chain 的 pair-swap 配平，直接比较
   Type-7 P95；max 和 mean 只用于 P95 相同时的稳定 tie-break。
3. 只锁定下一个 8-request 全局 dispatch prefix，不锁请求到具体 backend。
   哪个 backend 实际先空闲，就取 prefix 的下一个请求。
4. 新到请求不能插入当前 wave，但最多 8 次 pull 后立即参加下一轮规划。
   Tail 预算、Tail Pack 和 protected-drain 均不变。

| 验证口径 | Queue-Band P95 | 候选 P95 | P95 减少 | 相对 fixed Beam |
|---|---:|---:|---:|---:|
| req50 / 100 paired seeds | 1865.701s | 1816.689s | 2.598%，胜率 93% | -0.089%，CI 跨 0 |
| req100 / 100 paired seeds | 2325.770s | 2267.367s | 2.502%，胜率 92% | +0.027%，CI 跨 0 |
| Queue-Band 真实服务路径回放 | 1716.102s | 1695.418s | 1.205% | +0.255% |
| Beam 真实服务路径回放 | 1718.266s | 1696.503s | 1.267% | +1.073% |

多种子上该方案与 Beam 统计持平，但 Beam 的 req50 NPU 实际只改善 0.069%；
波次顺序锁定在两条真实残差路径上都保持约 1.2% 收益，因此进入 req50 NPU。
第一个 epoch 在约 643s、24 个 pending 时建立，锁定的 8 个请求全部已经到达。
之后即使预计/实际 backend 位置只有 6/8 或 4/8 匹配，按全局顺序执行仍保持
双路径收益，说明收益来自稳定的跨链顺序，而不是假定准确的卡绑定。

生产实现使用
`central_pull_backlog_leveling_wave_commit`，启动 preset 为
`run_wan22_super_p95_dispatcher_8x1_backlog_leveling_wave_commit.sh`。
trace 会分别记录 `backlog_leveling_epoch_plan` 和
`backlog_leveling_wave_dispatch`。
该实现当前只定位为实验候选：规划仍在 dispatcher 锁内同步执行。req50 preset
把候选评估上限设为 4096；两条真实路径的 epoch 最多使用 2269 次评估，本机
约为几十毫秒。通用生产化仍需时间预算或异步快照规划。

筛选顺序仍是 req50/req100 多种子仿真、真实残差回放、req50 NPU；只有前两关
都有明确收益才运行 NPU，req100 NPU 最后再做。

代码分支为 `codex/wan22-queue-band-risk-8xusp1`。候选实现位于该分支最新
提交，父提交为 `a53639c6eaa160636ba013d0a49e8d9f0cc11ce3`，推送目标为
`git@github.com:HpSuda/vllm-omni.git`。
