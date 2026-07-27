# Wan2.2 8×USP1 / 100 请求 Beam NPU 结果

## 结果

本轮测试：

```text
Tail-aware Release-Calendar Beam
+ Tail Pack
+ Protected Drain
```

100/100 个正式请求成功，纯 E2E P95 为 **2122.506s**。

| 对照 | P95 | P95 降低 | 加速比 |
|---|---:|---:|---:|
| 2×USP4 实测参考 | 3010.310s | **29.492%** | **1.418×** |
| 假设原始 baseline：`3010.310 / 0.7` | 4300.443s | **50.644%** | **2.026×** |
| 上一版 β=0.85 + Tail Pack + Protected Drain | 2177.259s | **2.515%** | **1.026×** |

2×USP4 的 `3010.310s` 来自原始 PDF；仓库还记录过一轮同口径复跑
`2968.990s`，两者相差 1.373%。按照统一口径，本报告仍固定使用
`3010.310s`，复跑值只用于确认结果大致可复现。

## 汇总指标

| 指标 | 上一版 β=0.85 | Beam | Beam 变化 |
|---|---:|---:|---:|
| Duration | 8111.367s | **7996.688s** | 降低 1.414% |
| Throughput | 0.012328 | **0.012505** | 提高 1.437% |
| Mean | 1548.990s | **1524.095s** | 降低 1.607% |
| Median | **1513.348s** | 1522.748s | 增加 0.621% |
| P95 | 2177.259s | **2122.506s** | **降低 2.515%** |
| P99 | 5848.516s | **5736.667s** | 降低 1.912% |

Beam 没有用 P99 或 Duration 换取本轮 P95；除 Median 略高外，其余汇总指标
也有改善。但 5 个 Tail 仍集中串行执行，所以 P99 和 Duration 的绝对值仍然较大。

## P95 为什么降低

100 请求的 NumPy type-7 P95 为：

```text
P95 = 0.95 × 第 95 小 + 0.05 × 第 96 小
```

| 方案 | 第 95 小 | 第 96 小 | P95 |
|---|---:|---:|---:|
| 上一版 β=0.85 | `request-00096` Normal：2130.067s | `request-00098` Tail：3073.903s | 2177.259s |
| Beam | `request-00096` Normal：2078.518s | `request-00098` Tail：2958.283s | **2122.506s** |

收益可以拆成：

```text
Normal 边界：(2130.067 - 2078.518) × 95% = 48.972s
Tail 边界：  (3073.903 - 2958.283) ×  5% =  5.781s
合计：                                            54.753s
```

Beam 同时改善了 Normal 边界和 Tail 边界。5 个 Tail 的 Gate 释放比上一版
提前约 116.2s，实际执行顺序仍是：

```text
request-00018 → request-00098 → request-00080
→ request-00060 → request-00038
```

因此本轮收益不是改变 Tail budget 或抢占规则，而是 Beam 根据可见 backend
释放时间重新安排 Normal，使 Tail backend 更早完成 Normal drain。

## 策略是否生效

- 95 Normal / 5 Tail，输入分布为 15 short / 32 medium / 53 long；
- 共执行 96 次 Release-Calendar 规划，其中 46 次实际启用 Beam 搜索；
- 其余请求在 pending 不处于 `[10, 27]` 时安全回退到在线风险排序；
- planner 总耗时 665.116ms，约占 7996.688s Duration 的 0.008%；
- 5 个 Tail 全部使用 Tail Pack + Protected Drain；
- 抢占 0 次，调度器没有获得最终请求总数。

预运行仿真 P95 为 2087.667s，实测为 2122.506s，绝对值低估 34.840s /
1.642%。仿真预计相对上一版降低 2.013%，实测降低 2.515%，方向和主要收益均
得到验证。

## 实验与产物

| 项目 | 值 |
|---|---|
| 实验名 | `wan22_tail_aware_release_calendar_beam_req100_20260728_011309` |
| 源码提交 | `201d62eafcfaeaa66a4201e1ae5a2f73260de97c` |
| Runner SHA256 | `b4fa60b32775793f3aec81a7edf81e7e2dd50a8d0aa990be031ae7c054418ea4` |
| Artifact SHA256 | `c469b80b5f6646d36d462951d39d59e9f13fe9087091f67746cd9a9cd1dfc051` |

artifact 的 result、request trace、dispatcher/backend 原始 trace、health、
源码 manifest 和清理检查均已通过独立分析。清理后 9 个端口全部释放，8 张
NPU 均空闲，目标服务进程为空。同一进程组仍有 168 个
`[python3] <defunct>`，它们不占端口或 NPU，但容器的子进程回收问题仍需修复。
