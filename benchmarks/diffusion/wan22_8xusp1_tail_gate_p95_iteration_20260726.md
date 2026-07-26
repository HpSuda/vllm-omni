# Wan2.2 8×USP1 Tail Gate 调度迭代

## 本轮方案

```text
Central Pull
+ Cost-Damped Risk beta=0.85
+ Tail Pack
+ Tail Gate
```

本轮只优化 8×USP1、50 requests 的纯 P95。benchmark 的请求规格、0.05 rps、
seed=42、Tail credit 规则和 backend 抢占实现均不修改。

Tail 仍由每 20 个到达请求产生的 1 个 credit 在线选出，仍通过 Tail Pack
集中到同一个 backend。Tail Gate 只改变开始时机：

1. Tail 到达后立即完成分类和 backend 预留；
2. 如果中央 Normal 队列或目标 backend 仍有 Normal，则先在 dispatcher 等待；
3. 中央 Normal 队列清空且目标 backend 的 Normal 完成后，再把 Tail 发给 backend；
4. Tail 释放后若又有 Normal 到达，仍可在原有阶段边界抢占 Tail。

该规则只使用到达历史、当前队列和 backend 状态。它保护 P95 边界附近的 Normal，
代价是两个 Tail 更晚完成，因此预期 P99 和总 Duration 变差。

## 本地仿真

服务时间使用上一轮 8×USP1 / 50-request NPU trace 中 clean Normal 的 10% trimmed
mean，调度器估时仍使用生产 Wan2.2 estimator：

| 规格 | 仿真实际服务(s) | 调度器可见估时(s) |
|---|---:|---:|
| short | 106.190 | 110.724 |
| medium | 221.959 | 219.548 |
| long | 628.089 | 612.299 |

按规格中心化后的 pooled log sigma 为 `0.011468`。seed=42 仿真使用与 benchmark
完全相同的 50 个请求类型序列；模拟到达时间与实测 client trace 的平均绝对误差
为 0.025s，最大误差为 0.049s。

固定 benchmark 流的结果：

| 指标 | 当前 `beta=0.95 + Tail Pack` | `beta=0.85 + Tail Pack + Tail Gate` | 变化 |
|---|---:|---:|---:|
| P95(s) | 1783.782 | **1744.148** | **降低 2.222%** |
| Mean(s) | 1292.981 | 1264.799 | 降低 2.180% |
| Median(s) | 1436.920 | 1407.513 | 降低 2.047% |
| P99(s) | 2679.191 | 2957.734 | 增加 10.397% |
| Duration(s) | 3384.740 | 3714.693 | 增加 9.748% |
| Throughput(req/s) | 0.014772 | 0.013460 | 降低 8.882% |
| 抢占次数 | 3 | 0 | 减少 100% |

仿真中的 P95 使用 NumPy 默认 type-7 线性插值。候选的边界请求为：

| 排名 | Request | 规格 | 队列 | Backend | E2E(s) | 插值权重 |
|---:|---|---|---|---|---:|---:|
| 47 | `request-00040` | long | Normal | backend-1 | 1719.029 | 45% |
| 48 | `request-00048` | long | Normal | backend-4 | 1764.699 | 55% |

```text
P95 = 1719.029 × 45% + 1764.699 × 55%
    = 1744.148s
```

两个 Tail 均位于 backend-7：

| Request | 首次执行(s) | 完成(s) | E2E(s) | 抢占 |
|---|---:|---:|---:|---:|
| `request-00038` | 2446.542 | 3085.187 | 2402.060 | 0 |
| `request-00018` | 3085.187 | 3714.693 | 3491.618 | 0 |

当前方案会让 Tail 提前进入 backend，再被 Normal 抢占 3 次；Tail Gate 把这三次
抢占消除，并把 Tail 干扰限制在 Normal drain 之后。图中的末尾空闲主要是其余
七个 backend 等待 backend-7 顺序完成两个 Tail，不是可供 Normal 使用却遗漏的
计算空泡。

## 1,000 seeds 稳健性

每个 seed 都先计算相对当前 `beta=0.95 + Tail Pack` 的 P95 降低比例，再求平均。
请求数固定为 50；策略不读取 seed 或最终请求总数。

| Backend 模型 | 当前 P95 均值(s) | Tail Gate P95 均值(s) | 平均 P95 降低 | Win rate |
|---|---:|---:|---:|---:|
| 等速 backend | 2000.008 | 1924.425 | **3.771%** | 95.6% |
| 实测速度差异 | 2001.021 | 1924.502 | **3.812%** | 96.3% |

在等速模型中，最差 seed 回退 5.400%；在实测速度模型中，最差 seed 回退
5.019%。若把 beta 保持为 0.95，两个模型的 win rate 都为 98.5%，但平均
P95 收益降到 3.436%/3.447%，固定 benchmark 流收益也更低。因此本轮选择
`beta=0.85` 上机验证。

## NPU 实测

待完成 8×USP1、50-request 单轮实验后补充。
