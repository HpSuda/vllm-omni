# Wan2.2 8×USP1 五个调度方案汇总

## 统一口径

本表只比较 Wan2.2 的纯 E2E P95。基准固定为：

| 请求流 | 2×USP4 实测参考 | 假设原始 baseline |
|---|---:|---:|
| 50 requests / 0.05 rps | 2191.560s | `2191.560 / 0.7` = 3130.800s |
| 100 requests / 0.03 rps | 3010.310s | `3010.310 / 0.7` = 4300.443s |

```text
P95 降幅 = (对照 P95 - 当前 P95) / 对照 P95
加速比 = 对照 P95 / 当前 P95
```

## 50 请求：全部已实测

| 方案 | P95 | 相对上一方案 | 相对 2×USP4 | 相对假设 baseline |
|---|---:|---:|---:|---:|
| 原策略 8×USP1 | 1962.350s | — | 降低 10.459%，1.117× | 降低 37.321%，1.595× |
| Central Pull + Cost-Damped Risk β=0.5 + Tail Spread | 1885.154s | 降低 3.934%，1.041× | 降低 13.981%，1.163× | 降低 39.787%，1.661× |
| Central Pull + Cost-Damped Risk β=0.95 + Tail Pack | 1788.961s | 降低 5.103%，1.054× | 降低 18.370%，1.225× | 降低 42.859%，1.750× |
| Central Pull + Cost-Damped Risk β=0.85 + Tail Pack + Tail Gate | 1746.806s | 降低 2.356%，1.024× | 降低 20.294%，1.255× | 降低 44.206%，1.792× |
| Central Pull + Tail-aware Release-Calendar Beam + Tail Pack + Tail Gate | **1714.909s** | **降低 1.826%，1.019×** | **降低 21.749%，1.278×** | **降低 45.225%，1.826×** |

从原策略到 Beam，P95 累计减少 247.441s，降低 **12.609%**。
“相对上一方案”只比较本表相邻两行，不代表中间所有收益均由一个参数贡献。

## 100 请求：已有结果与待测项

| 方案 | P95 | 相对上一已测方案 | 相对 2×USP4 | 相对假设 baseline | 状态 |
|---|---:|---:|---:|---:|---|
| 原策略 8×USP1 | 待测 | — | — | — | 待补测 |
| Central Pull + Cost-Damped Risk β=0.5 + Tail Spread | 2431.107s | — | 降低 19.241%，1.238× | 降低 43.468%，1.769× | 已实测 |
| Central Pull + Cost-Damped Risk β=0.95 + Tail Pack | 待测 | — | — | — | 待补测 |
| Central Pull + Cost-Damped Risk β=0.85 + Tail Pack + Tail Gate | 2177.259s | 比 β=0.5 降低 10.442%，1.117× | 降低 27.673%，1.383× | 降低 49.371%，1.975× | 已实测 |
| Central Pull + Tail-aware Release-Calendar Beam + Tail Pack + Tail Gate | **2122.506s** | 比 β=0.85 降低 **2.515%，1.026×** | 降低 **29.492%，1.418×** | 降低 **50.644%，2.026×** | 已实测 |

当前 req100 的最低实测 P95 是 Beam 的 **2122.506s**。原策略和
`β=0.95 + Tail Pack` 尚无同口径 NPU 实测；补测后再计算完整的逐步增益。

## 各方案只改变什么

| 方案 | 核心变化 |
|---|---|
| 原策略 8×USP1 | 两级队列；每 20 个请求产生 1 个 Tail credit；Normal FIFO，Tail LIFO。 |
| Central Pull + Cost-Damped Risk β=0.5 + Tail Spread | backend 空闲时从中心队列拉取；结合等待风险和预测耗时选择 Normal，Tail 分散。 |
| Central Pull + Cost-Damped Risk β=0.95 + Tail Pack | 提高风险权重，并把 Tail 集中到 Tail backend，减少对 Normal backend 的干扰。 |
| Central Pull + Cost-Damped Risk β=0.85 + Tail Pack + Tail Gate | Tail backend 先排空已接收的 Normal，再释放 Tail，保护 drain 窗口；代码中的模式名是 `protected_drain`。 |
| Central Pull + Tail-aware Release-Calendar Beam + Tail Pack + Tail Gate | 根据可见 backend 的预计释放时间，小范围联合规划下一批 Normal；Tail Pack 和 Tail Gate 保持不变。 |

五个方案使用相同 benchmark 请求流；新调度器不读取最终请求总数。
