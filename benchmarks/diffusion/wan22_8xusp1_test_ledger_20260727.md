# Wan2.2 8×USP1 调度测试清单

## 统一统计口径

当前只优化 **Wan2.2 的纯 E2E P95**，并按以下口径比较：

| 请求数 | 2×USP4 实测参考 | 假设原始 baseline |
|---:|---:|---:|
| 50 | 2191.560s | 3130.800s |
| 100 | 3010.310s | 4300.443s |

其中：

```text
假设原始 baseline = P95(2×USP4) / 0.7
P95 降幅 = (baseline - candidate) / baseline
加速比 = baseline / candidate
```

旧报告中直接使用 1×USP8/M8 实测值计算的百分比继续作为历史记录，但不再作为
当前主口径。

## 已完成的 NPU 测试

所有 8×USP1 实验均使用原 benchmark 请求流；表中的“增量”只比较同请求数的
相邻方案。

| 请求数 | 方案 | NPU P95 | 相对上一方案 | 相对 2×USP4 | 相对假设 baseline |
|---:|---|---:|---:|---:|---:|
| 50 | 原策略 8×USP1 | 1962.350s | — | 降 10.459%，1.117× | 降 37.321%，1.595× |
| 50 | Central Pull + Cost-Damped Risk β=0.5 + Tail Spread | 1885.154s | 降 3.934% | 降 13.981%，1.163× | 降 39.787%，1.661× |
| 50 | β=0.95 + Tail Pack | 1788.961s | 降 5.103% | 降 18.370%，1.225× | 降 42.859%，1.750× |
| 50 | β=0.85 + Tail Pack + protected-drain | 1746.806s | 降 2.356% | 降 20.294%，1.255× | 降 44.206%，1.792× |
| 50 | Queue-Band Risk + Tail Pack + protected-drain | 1716.099s | 降 1.758% | 降 21.695%，1.277× | 降 45.187%，1.824× |
| 50 | Tail-aware Release-Calendar Beam | 1714.909s | 降 0.069% | 降 21.749%，1.278× | 降 45.225%，1.826× |
| 100 | Cost-Damped Risk β=0.5 + Tail Spread | 2431.107s | — | 降 19.241%，1.238× | 降 43.468%，1.769× |
| 100 | β=0.85 + Tail Pack + protected-drain | 2177.259s | 降 10.442% | 降 27.673%，1.383× | 降 49.371%，1.975× |

当前结论：

- req50 从原策略 8×USP1 的 1962.350s 降到 1714.909s，累计降低
  12.609%，但最后一步 Beam 相对 Queue-Band 只降低 1.190s / 0.069%，属于
  噪声量级。
- 因此当前工程参考方案仍采用更简单的 Queue-Band；Beam 只作为算法对照。
- req100 没有原策略 8×USP1 和 β=0.95 + Tail Pack 的 NPU 中间点，不能把
  2177.259s 的全部收益归因于最后一个调度改动。
- Beam 实验启动时曾复用残留 backend；模型、拓扑、backend scheduler 和完整
  请求 trace 均一致，结果可比较。后续 runner 必须在启动前检查 8080、
  8091–8098 端口和残留进程。

## 最近的 CPU 仿真与真实 trace 回放

| 方案 | req50 | req100 | 两条真实服务残差回放 | 结论 |
|---|---|---|---|---|
| Tail-aware Release-Calendar Beam | 与 Wave 相同的 100 seeds 降 2.678% | 同 cohort 降 2.470% | 预测收益约 0.06%–0.95% | NPU 实际只降 0.069%，说明逐 pull 规划对 ETA 误差敏感 |
| 全局可见队列逐-pull 重规划 | 相对 Queue-Band 降 2.938% | 降 2.509% | 只降 0.814% / 0.061% | 首个 8-request 计划实际位置只命中 1/8，计划抖动过大，否决 |
| 积压配平 + 波次顺序锁定 | paired mean 降 2.598%，胜率 93% | paired mean 降 2.502%，胜率 92% | 降 1.205% / 1.267% | 当前最强 NPU 候选 |

“积压配平 + 波次顺序锁定”相对 fixed Beam 的多种子结果基本持平：

- req50：回退 0.089%，置信区间跨 0；
- req100：改善 0.027%，置信区间跨 0。

但它在两条真实残差路径上分别得到 1695.418s 和 1696.503s，均明显好于
Queue-Band，并比对应 fixed Beam 再降低 0.255% / 1.073%。其关键区别是锁定
下一波 8 个请求的全局顺序，避免 ETA 小误差导致每次 pull 都推翻计划。

这里的 2.598% / 2.502% 是逐 seed 先计算降幅再取均值；若直接用两列 P95
均值相除，会得到 2.627% / 2.511%，两者是不同统计量。其他报告中 Beam
约 2.65% / 1.95% 的数字来自不同 seed cohort，不能与本表直接混算。

## 已否决的主要方向

| 方向 | 否决原因 |
|---|---|
| Min-Cost Matching / Shadow Price | 前者不改变 Queue-Band 动作；后者 req50/100 分别回退 16.413% / 9.781% |
| Stable Pairwise / Critical-Chain / Branch-and-Bound | 没有稳定超过 Beam，部分 req100 回退，规划开销更高 |
| Dynamic Tail Reselection / Provisional Tail | 多种子无显著收益，真实残差回放回退约 2.5%–3.0% |
| Censored ETA / Near-Release Matching | 相对 Beam 收益接近 0 或回退，真实路径方向不一致 |
| Deadline/Laxity | req50 最好只降 0.235%；真实路径为持平 / 回退 1.412% |
| Backend×输入类型亲和性学习 | 类型交互只有约 0.2%–0.46%，多种子收益约 0，真实回放更差 |
| Robust Gate / Residual Calibration | 只能在部分路径改善，不能稳定超过 Beam |
| Release Coalescing | 有界等待不触发；强制等待约 29s 后 P95 不变 |

## 下一步

1. 服务器重连后，先在容器运行新策略的 dispatcher 单测。
2. 只跑一次 req50 的“积压配平 + 波次顺序锁定”NPU 实验，benchmark 参数不变。
3. 必须从 raw trace 验证确实发生了 epoch 规划和 changed dispatch，并记录每次
   planner elapsed；它当前只是实验候选，不宣称生产安全。
4. 只有 req50 NPU 明确优于 1716.099s，才安排 req100 NPU。
5. NPU 运行期间继续在 CPU 上探索波次锁定的大邻域改进，不做纯参数搜索。

当前候选实现提交为 `e3bcecdc46d178ec6cbf219bd8ac81c15ffa573c`，已推送到
`codex/wan22-queue-band-risk-8xusp1`。
