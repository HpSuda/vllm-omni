# Wan2.2 8×USP1 / 100 请求在线调度仿真初筛

## 口径

- 只测试 `8 backend × USP1`，100 requests，`0.03 rps`。
- benchmark 对齐结果使用单一 `seed=42`；该 trace 为 15 short、32 medium、53 long。
- 当前参考策略为滚动 Tail 预算（每 20 次到达释放 1 个）、Central Pull、Cost-Damped Risk `beta=0.5`。
- 初始 M8 base 指原始文档中的 `1×USP8 / 100 requests / 0.03 rps`，P95 为 `4264.97s`。
- 服务模型只使用 8×USP1/100 NPU trace 中 clean Normal 执行区间校准：short/medium/long 为 `105.452/221.790/627.316s`，实际服务抖动 `sigma=0.01125`。
- 调度器仍看到生产 estimator 的 `110.724/219.548/612.299s`；估时误差采样为 0，实际服务 scale 固定为 1.0。
- 没有用实测 E2E P95 反向拟合 scale。下表相对初始 M8 的数值是仿真 P95 与 `4264.97s` 的理论比值，不能当作 NPU 实测加速。

计算口径：

```text
相对当前 P95 改善 = (2112.840 - candidate_p95) / 2112.840
相对初始 M8 加速比 = 4264.97 / candidate_p95
相对初始 M8 P95 降幅 = (4264.97 - candidate_p95) / 4264.97
```

## 当前策略的校准误差

| 口径 | P95 (s) | 相对初始 M8 |
|---|---:|---:|
| 当前策略 NPU 实测 | 2431.107 | 1.754× / 降 43.00% |
| 当前策略仿真 | 2112.840 | 2.019× / 降 50.46% |

仿真比实际低 `318.267s`，相对实测误差为 `-13.09%`。这是只校准服务锚点后的自然误差；本轮没有为了让 E2E P95 对齐而再乘 `2431.107 / 2112.840 = 1.151` 的 scale。因而本报告主要用于比较同一仿真口径下的策略相对收益，不用绝对 P95 代替 NPU 结果。

## seed=42 初筛

`相对当前 P95` 为正表示 P95 更低。

| 策略 | P95 (s) | 相对当前 P95 | Mean (s) | P99 (s) | Duration (s) | 理论相对初始 M8 |
|---|---:|---:|---:|---:|---:|---:|
| Central Pull + Max Risk | 2085.89 | +1.28% | 1498.35 | 4432.64 | 6185.93 | 2.045× / 降 51.09% |
| Central Pull + Cost-Damped Risk beta=0.45 | 2098.02 | +0.70% | 1428.86 | 4432.66 | 5921.45 | 2.033× / 降 50.81% |
| Central Pull + Cost-Damped Risk beta=0.95 | 2110.94 | +0.09% | 1500.31 | 4494.82 | 6391.14 | 2.020× / 降 50.51% |
| Central Pull + Cost-Damped Risk beta=0.5（当前） | 2112.84 | +0.00% | 1429.28 | 4381.00 | 5863.45 | 2.019× / 降 50.46% |
| Central Pull beta=0.5 + 本地 Normal Arrival+Cost | 2112.84 | +0.00% | 1429.28 | 4381.00 | 5863.45 | 2.019× / 降 50.46% |
| Central Pull + Cost-Damped Risk beta=0.75 | 2118.29 | -0.26% | 1487.84 | 4844.22 | 6565.57 | 2.013× / 降 50.33% |
| Central Pull + Cost-Damped Risk beta=0.65 | 2121.56 | -0.41% | 1464.30 | 4315.97 | 6069.27 | 2.010× / 降 50.26% |
| Central Pull + Cost-Damped Risk beta=0.70 | 2139.18 | -1.25% | 1468.65 | 4969.78 | 6108.31 | 1.994× / 降 49.84% |
| Central Pull + Cost-Damped Risk beta=1.25 | 2148.56 | -1.69% | 1538.16 | 4492.67 | 6120.22 | 1.985× / 降 49.62% |
| Central Pull + Cost-Damped Risk beta=0.35 | 2162.49 | -2.35% | 1410.71 | 4346.17 | 5874.41 | 1.972× / 降 49.30% |
| Central Pull + Cost-Damped Risk beta=0.25 | 2172.13 | -2.81% | 1403.00 | 4587.38 | 5897.08 | 1.963× / 降 49.07% |
| Central Pull + Cost-Damped Risk beta=0.10 | 2192.27 | -3.76% | 1381.34 | 4371.38 | 5865.35 | 1.945× / 降 48.60% |
| 本地 Assigned-Load + Normal Arrival+Cost | 2214.66 | -4.82% | 1297.36 | 5051.45 | 6186.74 | 1.926× / 降 48.07% |
| Central Pull + Highest Response Ratio | 2224.83 | -5.30% | 1174.13 | 4325.82 | 5848.27 | 1.917× / 降 47.83% |
| Central Pull + FIFO | 2229.75 | -5.53% | 1351.72 | 4430.84 | 5881.30 | 1.913× / 降 47.72% |
| Central Pull + Arrival+Cost | 2240.34 | -6.03% | 1239.87 | 4368.18 | 6182.26 | 1.904× / 降 47.47% |
| Central Pull + Cost-Damped Risk beta=1.50 | 2264.07 | -7.16% | 1571.11 | 4318.09 | 5933.61 | 1.884× / 降 46.91% |
| 本地 Assigned-Load + Normal FIFO | 2308.16 | -9.24% | 1356.58 | 4991.25 | 6128.09 | 1.848× / 降 45.88% |
| 本地 Assigned-Load + Normal Size-Class FIFO | 2529.36 | -19.71% | 1179.41 | 4883.94 | 6015.24 | 1.686× / 降 40.69% |
| 本地 Assigned-Load + Normal Bounded Size-Class FIFO | 2529.36 | -19.71% | 1179.41 | 4883.94 | 6015.24 | 1.686× / 降 40.69% |

纯 P95 下，seed=42 的第一名变为 `Central Pull + Max Risk`，相对当前改善 `1.28%`。`beta=0.45` 在该 trace 上改善 `0.70%`，但没有通过下方独立种子检查；旧结果中的 `beta=0.95` 在 seed=42 上仅改善 `0.09%`。Max Risk 的 mean、P99 和 duration 都没有同步改善，符合本轮只优化 P95 的目标，但也说明它在移动慢请求的位置而不是普遍提速。

## 100-seed 独立鲁棒性检查

以下使用与 benchmark seed=42 分离的 `10042..10141`。它用于检查方向，不是 benchmark 对齐结果。

| 策略 | Mean P95 (s) | 配对 P95 改善 | 胜率 | 95% bootstrap CI | Mean (s) | P99 (s) | Duration (s) | 理论相对初始 M8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cost-Damped Risk beta=0.95 | 2361.28 | +1.07% | 66% | [+0.56%, +1.57%] | 1568.22 | 4893.61 | 6444.70 | 1.806× / 降 44.64% |
| Max Risk | 2362.95 | +1.00% | 65% | [+0.50%, +1.52%] | 1575.14 | 4854.38 | 6477.77 | 1.805× / 降 44.60% |
| Cost-Damped Risk beta=0.70 | 2363.61 | +0.95% | 60% | [+0.48%, +1.43%] | 1536.99 | 4879.74 | 6460.67 | 1.804× / 降 44.58% |
| Cost-Damped Risk beta=0.5（当前） | 2388.26 | +0.00% | — | — | 1512.18 | 4888.11 | 6516.51 | 1.786× / 降 44.00% |
| Cost-Damped Risk beta=0.45 | 2395.24 | -0.31% | 50% | [-0.64%, +0.01%] | 1504.08 | 4854.76 | 6473.99 | 1.781× / 降 43.84% |
| FIFO | 2432.05 | -1.98% | 30% | [-2.69%, -1.28%] | 1436.36 | 4843.70 | 6471.86 | 1.754× / 降 42.98% |

`beta=0.95`、Max Risk、`beta=0.70` 三个预注册候选的 Holm 校正后单侧 `p` 均为 `0.000210`。但 `beta=0.70` 在固定 benchmark seed=42 上回退 `1.25%`，不适合作为优先 NPU 候选。

## 等价与未生效项

- 对中央池中尚未启动的 Normal，Cost-Damped Risk `beta=0` 等价于 FIFO，`beta=1` 等价于 Max Risk。
- 启用 Central Pull 后，本地 `normal_order` 没有多个 Normal 候选可排。把本地 Normal 改为 Arrival+Cost 与当前策略逐项完全相同，不是一个有效优化维度。
- 本地 Size-Class FIFO 与 Bounded Size-Class FIFO 在 seed=42 上结果相同，只表示该 trace 没有触发边界差异，不表示二者普遍等价。
- Arrival+Cost 和 Highest Response Ratio 虽是在线策略，但当前生产 dispatcher 尚无对应 Central Pull 选项；它们本次也没有 P95 收益。

## 建议进入 NPU 的候选

1. `Central Pull + Max Risk`：seed=42 改善 `1.28%`，独立 100-seed 改善 `1.00%`；固定 benchmark 与鲁棒性结果方向一致，优先级最高。
2. `Central Pull + Cost-Damped Risk beta=0.95`：seed=42 改善仅 `0.09%`，但独立 100-seed 平均改善 `1.07%`；可作为连续权重方案的对照。

暂不建议把第三个候选送入 NPU：`beta=0.45` 未通过独立种子检查，`beta=0.70` 则在固定 seed=42 上回退。上面两个候选都只改变 Central Pull 的在线排序，不改变 benchmark 输入和 Tail 配置。

## 复现

```bash
.venv/bin/python -m benchmarks.diffusion.simulator.sweep \
  --matrix benchmarks/diffusion/simulator/configs/wan22_8xusp1_100_online_policy_screen_seed42.yaml \
  --output results/simulator/wan22_8xusp1_100_online_policy_screen_seed42.json

.venv/bin/python -m benchmarks.diffusion.simulator.sweep \
  --matrix benchmarks/diffusion/simulator/configs/wan22_8xusp1_100_online_policy_robustness.yaml \
  --output results/simulator/wan22_8xusp1_100_online_policy_robustness.json

.venv/bin/python -m benchmarks.diffusion.simulator.sweep_stats \
  --input results/simulator/wan22_8xusp1_100_online_policy_robustness.json \
  --output results/simulator/wan22_8xusp1_100_online_policy_robustness_stats.json \
  --hypothesis-variant "Central Pull + Cost-Damped Risk beta=0.95" \
  --hypothesis-variant "Central Pull + Cost-Damped Risk beta=0.70" \
  --hypothesis-variant "Central Pull + Max Risk"
```
