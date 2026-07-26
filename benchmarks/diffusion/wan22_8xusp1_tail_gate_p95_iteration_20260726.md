# Wan2.2 8×USP1 Tail Gate 调度迭代

## 结论

8×USP1、50-request NPU 实验已完成，实测方案为：

```text
Central Pull
+ Cost-Damped Risk beta=0.85
+ Tail Pack
+ Tail Gate
```

50/50 个正式请求成功，实测 P95 为 **1746.806s**。相对上一轮
`beta=0.95 + Tail Pack` 的 1788.961s，P95 直接降低 **2.356%**，
加速 **1.024×**；相对最初 M8 `1×USP8` 的 2750.930s，累计降低
**36.501%**，加速 **1.575×**。

固定 benchmark 流的仿真 P95 为 1744.148s，与 NPU 实测仅相差 −0.152%；
仿真预计直接降低 2.222%，实测降低 2.356%。两个 Tail 在真实 trace 中分别于
中央 Normal 队列深度 9 和 23 时进入 Gate，均预留到 backend-7；直到中央队列
清空才释放，抢占从上一轮的 3 次降为 0。

这是有效的纯 P95 优化，不是所有指标同时改善：Mean、Median 和 P95 均下降，
但 P99 增加 10.383%，Duration 增加 10.456%，Throughput 降低 9.466%。

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

### 统一加速比

每一行的“相对最初 base”都使用原始文档中的 M8 `1×USP8 / 50 requests`
P95 `2750.930s` 计算。直接对照用于区分本轮调度增量和拓扑累计收益。

| 方案 | P95(s) | 相对直接对照 | 相对最初 M8 base |
|---|---:|---:|---:|
| 最初 M8 `1×USP8` | 2750.930 | — | 1.000× / 降 0.000% |
| 原文 `2×USP4` | 2191.560 | 相对 M8：1.255× / 降 20.334% | 1.255× / 降 20.334% |
| 原始 8×USP1 | 1962.350 | 相对 M8：1.402× / 降 28.666% | 1.402× / 降 28.666% |
| 8×USP1 + `beta=0.5` + Tail Spread | 1885.154 | 相对原始 8×USP1：1.041× / 降 3.934% | 1.459× / 降 31.472% |
| 8×USP1 + `beta=0.95` + Tail Pack | 1788.961 | 相对上一行：1.054× / 降 5.103% | 1.538× / 降 34.969% |
| 8×USP1 + `beta=0.85` + Tail Pack + Tail Gate | **1746.806** | 相对上一行：**1.024× / 降 2.356%** | **1.575× / 降 36.501%** |

新方案相对原始 8×USP1 累计降低 **10.984%**、加速 **1.123×**；相对原文
`2×USP4` 降低 **20.294%**、加速 **1.255×**。相对最初 M8 的
36.501% 是拓扑和多轮调度修改的累计收益，不能全部归因于 Tail Gate。

### 汇总和纯 P95 取舍

| 成功 | Duration(s) | Throughput(req/s) | Mean(s) | Median(s) | P95(s) | P99(s) |
|---:|---:|---:|---:|---:|---:|---:|
| 50/50 | 3672.485 | 0.013615 | 1267.525 | 1408.977 | **1746.806** | 2913.988 |

与上一轮同拓扑 `beta=0.95 + Tail Pack` 实测直接比较：

| 指标 | 上一轮 | Tail Gate | 变化 |
|---|---:|---:|---:|
| P95(s) | 1788.961 | 1746.806 | **降低 2.356%** |
| Mean(s) | 1298.529 | 1267.525 | 降低 2.388% |
| Median(s) | 1452.089 | 1408.977 | 降低 2.969% |
| P99(s) | 2639.887 | 2913.988 | 增加 10.383% |
| Duration(s) | 3324.839 | 3672.485 | 增加 10.456% |
| Throughput(req/s) | 0.015038 | 0.013615 | 降低 9.466% |
| 抢占次数 | 3 | 0 | 减少 100% |

Tail Gate 将 Tail 推迟到 Normal drain 之后，因此同时改善了 Mean、Median 和
P95；代价集中在最慢两个 Tail，所以 P99 和总完成时间变差。这与“纯 P95”
目标一致。

### 仿真与实测

| 数据 | 当前 P95(s) | Tail Gate P95(s) | P95 降低 |
|---|---:|---:|---:|
| 固定 benchmark 流仿真 | 1783.782 | 1744.148 | 2.222% |
| NPU 实测 | 1788.961 | 1746.806 | 2.356% |

Tail Gate 仿真与 NPU 的候选 P95 误差为 **−0.152%**；实测收益比仿真多
0.134 个百分点。仿真不仅复现了请求类型和到达时间，也复现了两个 Tail 到达时
中央队列深度 9/23 的关键状态。

### P95 边界

P95 使用 NumPy 默认 type-7 线性插值：

| 排名 | Request | 规格 | 队列 | Backend | Dispatcher 等待(s) | Scheduler(s) | Backend(s) | E2E(s) | 权重 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 47 | `request-00040` | long | Normal | backend-4 | 1091.716 | 619.769 | 625.201 | 1718.952 | 45% |
| 48 | `request-00048` | long | Normal | backend-0 | 1127.999 | 635.602 | 639.563 | 1769.595 | 55% |

```text
P95 = 1718.952455 × 45% + 1769.595497 × 55%
    = 1746.806128s
```

### Tail Gate 的真实行为

| Request | 到达序号 | Gate 时中央队列 | 目标 Backend | Gate 等待(s) | Scheduler 等待(s) | 实际执行(s) | E2E(s) | 抢占 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `request-00018` | 20 | 9 | backend-7 | 2217.026 | 0.006 | 612.314 | 2835.582 | 0 |
| `request-00038` | 40 | 23 | backend-7 | 1756.954 | 612.314 | 614.002 | 2989.319 | 0 |

两个 Tail 在同一时刻释放，当时中央 Normal 队列为 0、backend-7 的 Normal
已完成，另外两个 backend 仍各有一个 Normal 在执行。`request-00018` 先执行，
`request-00038` 在 backend-7 等待前一个 Tail 完成。两者分别为最慢第 2 和
第 1 名，不进入 P95 插值边界。

### 每 Backend 执行情况

实际计算按每段 `scheduler_select → scheduler_preempt/complete` 累计；
Scheduler wait 是 backend 内等待，不与实际计算重复相加。

| Backend | Normal/Tail | 实际计算(s) | 实际计算利用率 | Dispatcher 等待均值(s) | Scheduler wait 均值(s) | E2E 均值(s) | 抢占 |
|---|---:|---:|---:|---:|---:|---:|---:|
| backend-0 | 4/0 | 2538.532 | 69.123% | 583.048 | 0.006 | 1223.695 | 0 |
| backend-1 | 7/0 | 2356.824 | 64.175% | 742.716 | 0.005 | 1084.697 | 0 |
| backend-2 | 6/0 | 2395.390 | 65.225% | 708.256 | 0.006 | 1112.489 | 0 |
| backend-3 | 8/0 | 2364.467 | 64.383% | 912.711 | 0.007 | 1213.107 | 0 |
| backend-4 | 6/0 | 2393.523 | 65.174% | 879.179 | 0.006 | 1283.622 | 0 |
| backend-5 | 6/0 | 2431.603 | 66.211% | 954.401 | 0.005 | 1365.004 | 0 |
| backend-6 | 5/0 | 2320.208 | 63.178% | 771.247 | 0.005 | 1240.410 | 0 |
| backend-7 | 6/2 | 3512.210 | 95.636% | 1031.250 | 76.544 | 1551.876 | 0 |

backend-7 的 Scheduler wait 均值包含第二个 Tail 等待第一个 Tail 的
612.314s。其余 backend 的 scheduler wait 均约 5–7ms，主要等待已在
dispatcher 中发生。

### 本轮服务时间校准

只使用 48 个未抢占且只被选择一次的 Normal，按规格各裁掉两端 10%：

| 规格 | 原始样本 | Trim 后 | Trimmed mean(s) | Median(s) | Trimmed CV | 调度器估时(s) |
|---|---:|---:|---:|---:|---:|---:|
| short | 10 | 8 | 105.820 | 105.576 | 0.833% | 110.724 |
| medium | 14 | 12 | 222.396 | 222.096 | 0.779% | 219.548 |
| long | 24 | 20 | 628.193 | 626.504 | 1.011% | 612.299 |

按规格中心化后的 pooled log sigma 为 `0.011087`，10% trim 后为
`0.009111`。与本轮仿真使用的 106.190/221.959/628.089s 很接近，不需要因
本次实验重做策略结论。

## 50 请求全量执行 Trace

Dispatcher 等待对 Normal 表示中央队列等待，对 Tail 表示 Tail Gate 等待。
Backend 时间是请求进入 backend 到推理返回的区间，可能包含 scheduler 内等待；
Scheduler 是实际执行区间累计。Backend 与 Scheduler 不能相加得到 E2E。
到达序号 1 是 benchmark warmup，因此 50 个正式请求的到达序号为 2–51。

| 到达序号 | Request | 规格 | 队列 | Backend | 估时(s) | Dispatcher 等待(s) | Scheduler(s) | Backend(s) | 抢占 | E2E(s) |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 2 | `request-00000` | long | normal | backend-0 | 612.299 | 0.000 | 637.438 | 641.398 | 0 | 643.434 |
| 3 | `request-00001` | short | normal | backend-1 | 110.724 | 0.000 | 102.872 | 105.017 | 0 | 107.062 |
| 4 | `request-00002` | medium | normal | backend-2 | 219.548 | 0.000 | 218.003 | 220.703 | 0 | 222.751 |
| 5 | `request-00003` | medium | normal | backend-3 | 219.548 | 0.000 | 222.142 | 225.165 | 0 | 227.221 |
| 6 | `request-00004` | long | normal | backend-4 | 612.299 | 0.000 | 620.186 | 624.757 | 0 | 626.805 |
| 7 | `request-00005` | long | normal | backend-5 | 612.299 | 0.000 | 630.432 | 635.450 | 0 | 637.502 |
| 8 | `request-00006` | long | normal | backend-6 | 612.299 | 0.000 | 628.909 | 632.917 | 0 | 634.962 |
| 9 | `request-00007` | short | normal | backend-7 | 110.724 | 0.000 | 103.752 | 105.592 | 0 | 106.308 |
| 10 | `request-00008` | long | normal | backend-1 | 612.299 | 0.000 | 612.069 | 616.533 | 0 | 618.579 |
| 11 | `request-00009` | short | normal | backend-1 | 110.724 | 1223.177 | 102.766 | 104.699 | 0 | 1329.912 |
| 12 | `request-00010` | medium | normal | backend-3 | 219.548 | 754.443 | 221.028 | 224.657 | 0 | 981.152 |
| 13 | `request-00011` | long | normal | backend-7 | 612.299 | 87.956 | 620.452 | 624.291 | 0 | 714.285 |
| 14 | `request-00012` | short | normal | backend-1 | 110.724 | 1310.271 | 102.780 | 104.835 | 0 | 1417.142 |
| 15 | `request-00013` | medium | normal | backend-4 | 219.548 | 1121.598 | 217.100 | 220.225 | 0 | 1343.866 |
| 16 | `request-00014` | long | normal | backend-2 | 612.299 | 80.047 | 618.960 | 622.884 | 0 | 704.970 |
| 17 | `request-00015` | long | normal | backend-3 | 612.299 | 69.952 | 633.261 | 637.499 | 0 | 709.500 |
| 18 | `request-00016` | medium | normal | backend-6 | 219.548 | 1145.925 | 219.200 | 221.833 | 0 | 1369.800 |
| 19 | `request-00017` | long | normal | backend-0 | 612.299 | 438.105 | 631.905 | 635.923 | 0 | 1076.071 |
| 20 | `request-00018` | long | tail | backend-7 | 612.299 | 2217.026 | 612.314 | 616.510 | 0 | 2835.582 |
| 21 | `request-00019` | short | normal | backend-4 | 110.724 | 1469.112 | 103.570 | 105.472 | 0 | 1575.449 |
| 22 | `request-00020` | long | normal | backend-4 | 612.299 | 402.785 | 615.317 | 619.562 | 0 | 1024.387 |
| 23 | `request-00021` | long | normal | backend-5 | 612.299 | 407.372 | 626.655 | 630.611 | 0 | 1040.025 |
| 24 | `request-00022` | medium | normal | backend-4 | 219.548 | 1189.859 | 217.582 | 220.371 | 0 | 1412.274 |
| 25 | `request-00023` | medium | normal | backend-6 | 219.548 | 1248.647 | 219.658 | 222.369 | 0 | 1471.225 |
| 26 | `request-00024` | long | normal | backend-6 | 612.299 | 391.767 | 623.700 | 627.644 | 0 | 1021.451 |
| 27 | `request-00025` | medium | normal | backend-3 | 219.548 | 1371.224 | 222.105 | 224.812 | 0 | 1598.082 |
| 28 | `request-00026` | short | normal | backend-5 | 110.724 | 1567.094 | 105.018 | 106.794 | 0 | 1675.926 |
| 29 | `request-00027` | short | normal | backend-3 | 110.724 | 1587.894 | 105.804 | 107.748 | 0 | 1696.252 |
| 30 | `request-00028` | long | normal | backend-1 | 612.299 | 346.527 | 608.678 | 613.579 | 0 | 962.151 |
| 31 | `request-00029` | long | normal | backend-7 | 612.299 | 421.132 | 618.034 | 622.175 | 0 | 1045.350 |
| 32 | `request-00030` | long | normal | backend-2 | 612.299 | 412.351 | 617.425 | 621.379 | 0 | 1035.768 |
| 33 | `request-00031` | long | normal | backend-3 | 612.299 | 631.625 | 632.151 | 636.351 | 0 | 1270.020 |
| 34 | `request-00032` | long | normal | backend-0 | 612.299 | 766.088 | 633.586 | 637.553 | 0 | 1405.680 |
| 35 | `request-00033` | long | normal | backend-5 | 612.299 | 798.539 | 628.078 | 631.985 | 0 | 1432.563 |
| 36 | `request-00034` | medium | normal | backend-5 | 219.548 | 1469.017 | 220.932 | 223.508 | 0 | 1694.572 |
| 37 | `request-00035` | long | normal | backend-7 | 612.299 | 870.511 | 621.169 | 625.375 | 0 | 1497.928 |
| 38 | `request-00036` | long | normal | backend-2 | 612.299 | 863.388 | 619.234 | 623.195 | 0 | 1488.637 |
| 39 | `request-00037` | long | normal | backend-1 | 612.299 | 911.833 | 611.494 | 615.860 | 0 | 1529.732 |
| 40 | `request-00038` | long | tail | backend-7 | 612.299 | 1756.954 | 614.002 | 1230.328 | 0 | 2989.319 |
| 41 | `request-00039` | long | normal | backend-6 | 612.299 | 1069.899 | 628.742 | 632.683 | 0 | 1704.609 |
| 42 | `request-00040` | long | normal | backend-4 | 612.299 | 1091.716 | 619.769 | 625.201 | 0 | 1718.952 |
| 43 | `request-00041` | short | normal | backend-3 | 110.724 | 1557.597 | 105.730 | 107.663 | 0 | 1665.943 |
| 44 | `request-00042` | medium | normal | backend-3 | 219.548 | 1328.949 | 222.247 | 225.697 | 0 | 1556.686 |
| 45 | `request-00043` | medium | normal | backend-7 | 219.548 | 1339.899 | 218.727 | 221.442 | 0 | 1563.382 |
| 46 | `request-00044` | short | normal | backend-7 | 110.724 | 1556.520 | 103.760 | 105.659 | 0 | 1662.856 |
| 47 | `request-00045` | medium | normal | backend-2 | 219.548 | 1338.161 | 218.043 | 220.710 | 0 | 1560.919 |
| 48 | `request-00046` | short | normal | backend-2 | 110.724 | 1555.588 | 103.726 | 105.570 | 0 | 1661.890 |
| 49 | `request-00047` | medium | normal | backend-1 | 219.548 | 1407.206 | 216.165 | 219.064 | 0 | 1628.302 |
| 50 | `request-00048` | long | normal | backend-0 | 612.299 | 1127.999 | 635.602 | 639.563 | 0 | 1769.595 |
| 51 | `request-00049` | medium | normal | backend-5 | 219.548 | 1484.382 | 220.490 | 223.026 | 0 | 1709.437 |

## 实验完整性与产物

- 运行前在服务器容器执行定向测试：`238 passed`；
- `benchmark.exit=0`，50/50 正式请求成功、0 失败；
- 正式 ID 为连续唯一的 `request-00000` 到 `request-00049`；
- 10 个 JSONL 文件共 791 个原始事件，50 个正式请求的 client、dispatcher、
  backend 和 scheduler 生命周期完整；
- request ID 与 video ID 一一对应；
- benchmark 结束时 dispatcher 与 8 个 backend health 均为 healthy；
- 配置回读为 `beta=0.85`、`tail_routing_mode=pack`、
  `tail_dispatch_mode=protected_drain`、`wan22:8xusp1_inferred`；
- 中央 Normal 最大深度 31，两个 Tail Gate 最大深度 2，释放 2 个，
  Gate 总等待 3973.980s；
- 原始事件复算的 Mean、Median、P95、P99 与 `result.json` 一致；
- 严格产物分析在服务器和本地均为 `passed`。

本地产物：

```text
/tmp/wan22_tail_gate_8xusp1_50_20260726_194540_artifact_final.tar.gz
SHA256 4d8247aacfc110e1ee3e0ed626c4f29344915a2cf53ef00a49a153bff54a17ca
```

服务器保留：

```text
/tmp/wan22_tail_gate_8xusp1_50_20260726_194540_artifact.tar.gz
/tmp/wan22_tail_gate_8xusp1_50_20260726_194540/
```

归档包含 `result.json`、全量 JSON/CSV 请求 trace、10 个原始 JSONL、
dispatcher 日志、8 个 backend 日志、运行前后 health 和 NPU 状态。产物传回并
校验后已停止实验容器；最终容器状态为 `exited`，8 张 NPU 上的 Python 进程数
为 0。

本轮实现提交：

```text
46e681d8 Add Tail Gate scheduling for Wan2.2
```
