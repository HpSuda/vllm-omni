# Wan2.2 8×USP1 Tail Pack 调度探索

## 结论

下一轮 8×USP1、50 请求 NPU 实验使用：

```text
Central Pull + Cost-Damped Risk beta=0.95 + Tail Pack
```

Tail 的预算和选择规则保持不变，仍是每 20 个到达请求释放 1 个 credit，并由
满足条件的大请求消耗。新方案只修改两点：

1. Normal 从中央队列拉取时，将 Cost-Damped Risk 的 `beta` 从 0.5 调为 0.95；
2. Tail 不再分散到最空闲 backend，而是复用已有 Tail 的 backend。

调度器只使用已经到达的请求、当前等待时间、Wan2.2 估时和当前 backend 负载。
benchmark 输入、请求总数可见性和负载生成方式均未修改。

固定 benchmark seed 的 50 请求仿真中，P95 从 1841.572s 降到
1781.603s，降低 **3.256%**。200 个随机种子中，该组合在等速 backend
模型下平均降低 **2.288%**，在按真实 trace 加入 backend 速度差异后平均降低
**1.923%**。

这是纯 P95 策略：Tail 的 P99、Mean 和总 Duration 可能变差，不把这些指标作为
本轮否决条件。

## 为什么继续优化 Tail 放置

当前 NPU trace 的每个请求均已拆成：

```text
到达时间
+ 中央 Normal 等待
+ backend scheduler 等待
+ scheduler 实际计算区间
= E2E
```

其中实际计算区间按每段 `scheduler_select → scheduler_preempt/complete`
累计，不把 Tail 被抢占后的暂停时间误算为计算。

当前 8×USP1、50 请求 NPU 基线为 48 Normal + 2 Tail。两个 Tail 被分散到
backend-0 和 backend-7，分别产生 3 次和 5 次抢占：

| Backend | Normal/Tail | 实际计算(s) | 利用率 | 中央等待均值(s) | Backend 等待均值(s) | E2E 均值(s) | 抢占 |
|---|---:|---:|---:|---:|---:|---:|---:|
| backend-0 | 5/1 | 2980.551 | 99.601% | 670.798 | 162.973 | 1516.005 | 3 |
| backend-1 | 7/0 | 2469.721 | 82.531% | 686.519 | 0.005 | 1044.659 | 0 |
| backend-2 | 6/0 | 2510.283 | 83.887% | 681.085 | 0.006 | 1104.769 | 0 |
| backend-3 | 7/0 | 2437.624 | 81.458% | 684.346 | 0.006 | 1037.314 | 0 |
| backend-4 | 7/0 | 2496.692 | 83.432% | 952.234 | 0.006 | 1313.799 | 0 |
| backend-5 | 4/0 | 2515.203 | 84.051% | 598.445 | 0.006 | 1233.796 | 0 |
| backend-6 | 6/0 | 2422.686 | 80.959% | 895.136 | 0.007 | 1303.562 | 0 |
| backend-7 | 6/1 | 2502.574 | 83.629% | 694.082 | 72.857 | 1383.426 | 5 |

同类 clean Normal 在各 backend 上的实测速度因子为：

| Backend | 速度因子 |
|---|---:|
| backend-0 | 0.9845× |
| backend-1 | 1.0165× |
| backend-2 | 1.0062× |
| backend-3 | 0.9866× |
| backend-4 | 1.0077× |
| backend-5 | 0.9923× |
| backend-6 | 0.9970× |
| backend-7 | 1.0042× |

最快和最慢只相差约 3.2%，小于中央等待和 Tail 干扰造成的差异。因此本轮不为
backend 建专门的静态快慢路由，而先集中 Tail 干扰。

## 仿真校准

服务时间只使用 50/100 请求两轮 NPU trace 中未抢占的 Normal：

| 规格 | 仿真实际服务均值(s) | 调度器可见估时(s) |
|---|---:|---:|
| short | 105.612 | 110.724 |
| medium | 221.694 | 219.548 |
| long | 627.783 | 612.299 |

按规格中心化后的实际服务 log sigma 为 `0.011239`。没有用 E2E P95 反向拟合
全局 scale。

仿真还修正了一个时序差异：生产 backend 在 dispatcher 返回下一个 Normal
之前，会让本地 Tail 先执行一个 denoise 区间；随后 Normal 才在阶段边界抢占。
修正后当前策略的仿真误差为：

| 请求数 | NPU P95(s) | 仿真 P95(s) | 仿真误差 |
|---:|---:|---:|---:|
| 50 | 1885.154 | 1841.572 | −2.312% |
| 100 | 2431.107 | 2453.161 | +0.907% |

## 多种子结果

表中的提升为逐 seed 先计算相对当前 `beta=0.5 + Tail Spread` 的 P95
降低比例，再求平均。

| Backend 模型 | 请求数 | 当前策略 P95 均值(s) | beta=0.95 + Tail Pack(s) | P95 降低 | Win rate |
|---|---:|---:|---:|---:|---:|
| 等速，200 seeds | 50 | 2044.531 | 1997.360 | **2.288%** | 81.0% |
| 等速，200 seeds | 100 | 2647.517 | 2465.916 | **6.867%** | 100.0% |
| 实测 backend 速度，200 seeds | 50 | 2051.502 | 2011.066 | **1.923%** | 74.5% |
| 实测 backend 速度，200 seeds | 100 | 2673.801 | 2488.121 | **6.984%** | 100.0% |

单独改变 Normal 顺序的 Risk-Slack SRPT 在 50/100 请求下平均只改善约
0.1%/0.3%；Guarded Max Risk 会回退 2%–10%，均不进入 NPU 实验。
本轮最稳定的收益来自 Tail Pack。

## 固定 benchmark seed 的逐请求仿真

### 50 请求

| 指标 | 当前 beta=0.5 + Spread | beta=0.95 + Pack | 变化 |
|---|---:|---:|---:|
| P95(s) | 1841.572 | 1781.603 | **降低 3.256%** |
| Mean(s) | 1234.397 | 1291.742 | 增加 4.646% |
| P99(s) | 2367.373 | 2677.021 | 增加 13.080% |
| Duration(s) | 2983.686 | 3382.433 | 增加 13.364% |
| 抢占次数 | 8 | 3 | 减少 62.5% |

当前两个 Tail 分布在两个 backend；候选的两个 Tail 都在一个 backend。
候选最慢 Tail 为 3159.358s，但 P95 边界的 Normal 从约 1878s 降到约
1780s。这正是纯 P95 的取舍。

固定 seed 的完整逐请求数据：

```text
results/simulator/wan22_8xusp1_tail_pack_trace/current_50_requests.csv
results/simulator/wan22_8xusp1_tail_pack_trace/candidate_50_requests.csv
```

### 100 请求

| 指标 | 当前 beta=0.5 + Spread | beta=0.95 + Pack | 变化 |
|---|---:|---:|---:|
| P95(s) | 2453.161 | 2198.921 | **降低 10.364%** |
| Mean(s) | 1522.644 | 1576.600 | 增加 3.544% |
| P99(s) | 2537.104 | 5608.196 | 增加 121.047% |
| Duration(s) | 5651.914 | 7258.140 | 增加 28.419% |
| 抢占次数 | 23 | 10 | 减少 56.5% |

100 请求产生 5 个 Tail。Pack 将 5 个 Tail 集中到一个 backend，因此 P95
改善明显，但 P99/Duration 明显恶化。该数据用于说明纯 P95 行为，本轮服务器
不运行 100 请求。

固定 seed 的完整逐请求数据：

```text
results/simulator/wan22_8xusp1_tail_pack_trace/current_100_requests.csv
results/simulator/wan22_8xusp1_tail_pack_trace/candidate_100_requests.csv
```

## 与基准的预计关系

以下只把仿真的固定-seed相对收益乘到已测 NPU 结果上，是 NPU 实验前的预计值，
不是实测结果：

| 50 请求口径 | P95(s) | 相对降低 | Speedup |
|---|---:|---:|---:|
| 初始文档 M8 baseline | 2750.930 | — | 1.000× |
| 原始 8×USP1 方案 | 1962.350 | 28.666% | 1.402× |
| 当前 Cost-Damped Risk 实测 | 1885.154 | 31.472% | 1.459× |
| Tail Pack 预计 | 1823.766 | **33.704%** | **1.508×** |

Tail Pack 相对当前实测预计降低 3.256%，相对原始 8×USP1 预计累计降低
7.062%。

## NPU 验证

只运行 8×USP1、50 请求，配置为：

```bash
CENTRAL_PULL_RISK_BETA=0.95
TAIL_ROUTING_MODE=pack
```

代码提交：

```text
d8dbf819 Add trace-calibrated Tail Pack scheduling
```

当前 NPU 结果暂未填写：现有 macOS Terminal 保留了已登录服务器会话，但当前
桌面控制接口拒绝操作 Terminal，且不会另开 SSH 绕过 2FA。终端恢复为可控后，
先在容器运行 dispatcher/trace 测试，再执行一次 50 请求 benchmark，并把实际
P95、全量逐请求 trace、每 backend 计算/等待表及三种基准加速比补到本节。
