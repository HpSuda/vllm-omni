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
| Central Pull + β=0.85 + Tail Pack + Tail Gate | 100 | NPU | 2177.259s | 27.673% | 1.383× | 49.372% |

Queue-Band req=50 的完整实验：

- 实验名：`wan22_queue_band_req50_20260727_164742`
- 50/50 成功，无抢占。
- Mean 1249.033s，Median 1349.571s，P95 1716.099s，P99 2916.994s。
- 对此前 Tail Gate 的 P95 额外改善 1.758%。
- 同一 trace 的仿真 P95 为 1715.462s，与实测相差 0.637s（0.037%）。
- 本地完整 artifact：`/tmp/wan22_queue_band_req50_20260727_164742_artifact.tar.gz`
- 本地解包目录：`/tmp/wan22_queue_band_req50_20260727_164742_artifact_local`

## 当前算法主线

### Tail-aware Release-Calendar Beam Search

这不是参数搜索。算法在每次 backend 释放时重新规划：

1. 从当前已到达的 Normal 请求中选出接近 P95 风险边界的候选，并额外允许一个短请求候选。
2. 根据各 backend 的预计释放时间，搜索接下来 4 个释放窗口。
3. 把正在执行和等待的 Tail 作为 P95 cohort 中的高延迟占位，避免反复牺牲新的 Normal。
4. 每条搜索路径都用 Queue-Band 补全，再直接比较预计 cohort P95；Mean 只用于同 P95 时的有限 tie-break。
5. 只提交第一步，下一次 backend 释放时重新规划。
6. 如果运行中 Tail 缺少可靠 ETA，明确回退 Queue-Band。

仿真结果：

- 历史 req=50 exact trace：Queue-Band 1715.462s → Beam 1695.734s，P95 改善 1.150%。
- 共享实现 seed=42：1718.310s → 1686.013s，改善 1.879%。
- req=50 / 1000 paired seeds：平均 P95 改善 2.645%，95% CI [2.531%, 2.759%]，胜率 96.7%；P99 和 makespan 的 CI 均跨 0。
- req=100 / 500 paired seeds：平均 P95 改善 1.954%，95% CI [1.819%, 2.089%]，胜率 93.4%；P99 和 makespan 的 CI 均跨 0。
- 本地规划耗时：平均约 4.07ms，P95 约 14.96ms。

代码与验证：

- 分支：`codex/wan22-queue-band-risk-8xusp1`
- 最新提交：`4c0db84560787fed5f3ff9216088579b3a694744`
- 主要算法提交：`79f6ebaff67293e357dc4af1b4c6119003cf4e4e`
- 已推送：`git@github.com:HpSuda/vllm-omni.git`
- 服务端容器已通过相关测试：250 passed。

## 已经在 NPU 后台运行的实验

- 实验：`wan22_tail_aware_release_calendar_beam_req50_20260727_103414`
- 拓扑：8×USP1。
- benchmark：50 请求，RPS 0.05，warmup 1，seed 42；三类输入权重 0.15 / 0.25 / 0.60。
- 唯一算法变化：Queue-Band Risk → Tail-aware Release-Calendar Beam Search。
- runner PID：`14345`。
- 二次确认状态：`BENCHMARK_RUNNING`；runner 仍存活，`health_start.json` 已生成。
- runner：容器内 `/tmp/wan22_tail_aware_release_calendar_beam_req50_runner.sh`
- 状态目录：容器内 `/tmp/wan22_tail_aware_release_calendar_beam_req50_20260727_103414`
- runner 日志：容器内 `/tmp/wan22_tail_aware_release_calendar_beam_req50_20260727_103414_runner.nohup.log`
- 完成 artifact：容器内 `/tmp/wan22_tail_aware_release_calendar_beam_req50_20260727_103414_artifact.tar.gz`
- 最新实验名指针：容器内 `/tmp/wan22_tail_aware_release_calendar_beam_req50_latest`

runner 使用 `nohup` 后台运行；本机网络或 Terminal 断开不会结束实验。实验完成后会自动停止 dispatcher/backends、生成分析文件和 SHA256，并把状态写成 `COMPLETE` 或 `COMPLETE_WITH_RESIDUAL_PROCESSES`。

## 重连后的第一组检查

只能通过已经登录服务器的 macOS Terminal 操作，不要另开 SSH：

```bash
docker exec vllm-omni-qwen-branch bash -lc '
exp=$(cat /tmp/wan22_tail_aware_release_calendar_beam_req50_latest)
echo "$exp"
cat "/tmp/${exp}/status"
tail -80 "/tmp/${exp}_runner.nohup.log"
test ! -f "/tmp/${exp}/health_end.json" || python3 -m json.tool "/tmp/${exp}/health_end.json"
'
```

若已完成，优先收集：

```bash
docker exec vllm-omni-qwen-branch bash -lc '
exp=$(cat /tmp/wan22_tail_aware_release_calendar_beam_req50_latest)
cat "/tmp/${exp}/artifact.sha256"
cat "/tmp/${exp}/artifact_analysis.md"
cat "/tmp/${exp}/processes_after_cleanup.txt"
'
```

随后把 `/tmp/wan22_tail_aware_release_calendar_beam_req50_20260727_103414_artifact.tar.gz` 从容器经当前 Terminal 传回本地，并校验远端/本地 SHA256。分析时至少检查：

- 50/50 成功、无超时或失败。
- `normal_routing_policy=central_pull_tail_aware_release_calendar_beam`。
- `release_calendar_plans > 0` 且 `release_calendar_beam_plans > 0`。
- health 中 Beam 参数为 horizon=4、width=16、branch=6、risk slack=100s。
- request trace 中存在实际 Beam 决策，而不是全程 fallback。
- P95 相对 1716.099s、2191.560s 和 3130.800s 的改善。

## 正在并行探索但尚未决定实测的算法

- Quantile-Boundary Min-Cost Matching：把当前已到达请求与未来 backend 释放窗口做在线匹配，只提交最近窗口。
- Quantile Shadow-Price / P95 Deficit Controller：给即将越过当前 P95 边界的请求动态影子价格。
- Stable P95 Pairwise Controller 已验证，但弱于 Beam：req=50 平均改善约 1.129%，req=100 约 0.571%，暂不消耗 NPU。

这些探索都必须先在同种子 req=50/100 仿真上超过 Beam，并保持 P99/makespan 无显著回退，才进入下一轮 NPU 测试。
