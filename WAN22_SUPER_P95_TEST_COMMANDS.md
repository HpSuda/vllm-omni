# Wan2.2 Super P95 启动命令

本分支提供 `8×USP1` 的以下方案：

```text
释放窗口规划 + Tail 集中延迟 + 空闲回填
```

运行前需确保容器能够读取 Wan2.2 模型；当前测试服务器离线运行时，
将主机 `/data/.cache/huggingface` 挂载到容器 `/root/.cache/huggingface`。

## 1. 切换分支

```bash
cd /vllm-workspace/vllm-omni
git switch super-p95-wan22
```

## 2. 清理旧进程

```bash
pkill -f 'wan22_super_p95_dispatcher.py|super_p95_dispatcher.py|vllm_omni.entrypoints.cli.main serve Wan-AI/Wan2.2-T2V-A14B-Diffusers' || true
rm -f /dev/shm/* 2>/dev/null || true
npu-smi info
```

## 3. 启动服务

```bash
rm -rf /tmp/wan22_super_p95_beam_backfill
mkdir -p /tmp/wan22_super_p95_beam_backfill/trace

env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python3 benchmarks/diffusion/wan22_super_p95_dispatcher.py \
  --host 127.0.0.1 \
  --port 8080 \
  --num-servers 8 \
  --device-ids '0;1;2;3;4;5;6;7' \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --backend-start-port 8091 \
  --backend-hardware-profiles 910B3 \
  --wan22-estimator-profile 8xusp1_inferred \
  --backend-scheduler super_p95_step \
  --wan22-scheduling-mode release_calendar_tail_pack_backfill \
  --backend-log-dir /tmp/wan22_super_p95_beam_backfill \
  --trace-log-dir /tmp/wan22_super_p95_beam_backfill/trace \
  --request-timeout-s 1000000 \
  --backend-health-timeout-s 1800 \
  --backend-health-poll-interval-s 10 \
  --backend-args=--omni \
  --backend-args=--usp \
  --backend-args=1 \
  --backend-args=--enable-layerwise-offload \
  --backend-args=--boundary-ratio \
  --backend-args=0.875 \
  --backend-args=--flow-shift \
  --backend-args=5.0 \
  --backend-args=--vae-use-slicing \
  --backend-args=--vae-use-tiling
```

也可以直接运行已经封装好的同等命令：

```bash
bash benchmarks/diffusion/run_wan22_super_p95_dispatcher_8x1_tail_aware_release_calendar_beam.sh
```

服务地址：

```text
dispatcher: 127.0.0.1:8080
backend:    127.0.0.1:8091–8098
```

检查是否就绪：

```bash
curl -s http://127.0.0.1:8080/health | python3 -m json.tool
grep -H 'Application startup complete' /tmp/wan22_super_p95_beam_backfill/backend_*.log
```

## 4. 运行 benchmark

### 4.1 50 请求，RPS 0.05

在另一个终端运行：

```bash
mkdir -p /tmp/wan22_super_p95_beam_backfill_req50
rm -f /tmp/wan22_super_p95_beam_backfill/trace/client.jsonl

env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url http://127.0.0.1:8080 \
  --backend v1/videos \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --dataset random \
  --task t2v \
  --num-prompts 50 \
  --max-concurrency 50 \
  --request-rate 0.05 \
  --warmup-requests 1 \
  --client-timeout-s 1000000 \
  --enable-negative-prompt \
  --disable-tqdm \
  --seed 42 \
  --random-request-seed 42 \
  --arrival-seed 42 \
  --random-request-config '[
    {"width":854,"height":480,"num_inference_steps":3,"num_frames":80,"fps":16,"weight":0.15},
    {"width":854,"height":480,"num_inference_steps":4,"num_frames":120,"fps":24,"weight":0.25},
    {"width":1280,"height":720,"num_inference_steps":6,"num_frames":80,"fps":16,"weight":0.60}
  ]' \
  --trace-log-file /tmp/wan22_super_p95_beam_backfill/trace/client.jsonl \
  --trace-label client \
  --output-file /tmp/wan22_super_p95_beam_backfill_req50/result.json

python3 benchmarks/diffusion/wan22_request_trace.py \
  --trace-dir /tmp/wan22_super_p95_beam_backfill/trace \
  --output-prefix /tmp/wan22_super_p95_beam_backfill_req50/request_trace
```

封装好的同等命令：

```bash
bash benchmarks/diffusion/run_wan22_super_p95_beam_backfill_benchmark_50.sh
```

### 4.2 100 请求，RPS 0.03

先重新执行第 2、3 步，再在另一个终端运行：

```bash
mkdir -p /tmp/wan22_super_p95_beam_backfill_req100
rm -f /tmp/wan22_super_p95_beam_backfill/trace/client.jsonl

env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url http://127.0.0.1:8080 \
  --backend v1/videos \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --dataset random \
  --task t2v \
  --num-prompts 100 \
  --max-concurrency 100 \
  --request-rate 0.03 \
  --warmup-requests 1 \
  --client-timeout-s 1000000 \
  --enable-negative-prompt \
  --disable-tqdm \
  --seed 42 \
  --random-request-seed 42 \
  --arrival-seed 42 \
  --random-request-config '[
    {"width":854,"height":480,"num_inference_steps":3,"num_frames":80,"fps":16,"weight":0.15},
    {"width":854,"height":480,"num_inference_steps":4,"num_frames":120,"fps":24,"weight":0.25},
    {"width":1280,"height":720,"num_inference_steps":6,"num_frames":80,"fps":16,"weight":0.60}
  ]' \
  --trace-log-file /tmp/wan22_super_p95_beam_backfill/trace/client.jsonl \
  --trace-label client \
  --output-file /tmp/wan22_super_p95_beam_backfill_req100/result.json

python3 benchmarks/diffusion/wan22_request_trace.py \
  --trace-dir /tmp/wan22_super_p95_beam_backfill/trace \
  --output-prefix /tmp/wan22_super_p95_beam_backfill_req100/request_trace
```

封装好的同等命令：

```bash
bash benchmarks/diffusion/run_wan22_super_p95_beam_backfill_benchmark_100.sh
```

## 5. 结果位置

```text
50 请求： /tmp/wan22_super_p95_beam_backfill_req50
100 请求：/tmp/wan22_super_p95_beam_backfill_req100
服务日志：/tmp/wan22_super_p95_beam_backfill
```

每个结果目录包含 `result.json`、`request_trace.json` 和
`request_trace.csv`。
