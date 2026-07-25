# Diffusion Scheduling Simulator

This directory contains a CPU-only discrete-event simulator for quickly testing
diffusion scheduling ideas. It is intentionally independent from the HTTP
dispatcher, Torch, and device runtimes.

The first profiles target the Wan2.2 `RANDOM3` benchmark. The bundled service
times (`38.07`, `71.34`, and `119.71` seconds) are estimator anchors, not a
calibrated model of every USP/offload topology. Treat absolute latency as
theoretical until a topology-specific profile is measured. Relative policy
experiments should use the same topology and service profile.

Two timing paths are available: `fixed_anchor` preserves those existing
anchors exactly, while `analytic_wan22` derives coarse phase time from input
shape, aggregate transformer FLOPs, and per-rank USP/HSDP traffic. Neither path
simulates individual kernels.

## Quick Start

Validate and run the default current-policy configuration:

```bash
python3 -m benchmarks.diffusion.simulator --validate-only
python3 -m benchmarks.diffusion.simulator
```

Run the coarse input-derived Wan2.2 model, or toggle its HSDP4 term:

```bash
python3 -m benchmarks.diffusion.simulator \
  --config benchmarks/diffusion/simulator/configs/wan22_analytic_usp4.yaml

python3 -m benchmarks.diffusion.simulator \
  --config benchmarks/diffusion/simulator/configs/wan22_analytic_usp4.yaml \
  --set service.hsdp.enabled=true
```

For example, the same analytic workload can become `1 x USP8` entirely through
configuration:

```bash
python3 -m benchmarks.diffusion.simulator \
  --config benchmarks/diffusion/simulator/configs/wan22_analytic_usp4.yaml \
  --set name=wan22-1xusp8-analytic \
  --set service.timing_model.execution.usp_degree=8 \
  --set service.hsdp.shard_size=8 \
  --set topology.backend_count=1 \
  --set topology.devices_per_backend=8 \
  --set topology.speed_factors='[1.0]'
```

Run one traceable experiment and save all output layers:

```bash
python3 -m benchmarks.diffusion.simulator \
  --config benchmarks/diffusion/simulator/configs/wan22_super_p95_current.yaml \
  --runs 1 \
  --output results/simulator/summary.json \
  --requests-output results/simulator/requests.csv \
  --trace-output results/simulator/events.jsonl
```

Override existing YAML fields without creating another file:

```bash
python3 -m benchmarks.diffusion.simulator \
  --set workload.utilization=1.1 \
  --set service.estimate_error_sigma=0.25 \
  --set topology.backend_count=1 \
  --set topology.devices_per_backend=8 \
  --set topology.speed_factors='[1.0]'
```

List indices are supported, so a measured service anchor can be replaced in
place with `--set workload.request_types.2.nominal_service_s=105.4`.

`PyYAML` is required to load configuration files. It is installed transitively
with the repository's common dependencies; a standalone environment can use
`pip install pyyaml`.

## Trace Diagnostics

Turn one simulator run into a backend timeline, idle/bubble analysis, and a
longest-wait request chart:

```bash
python3 -m benchmarks.diffusion.simulator \
  --config benchmarks/diffusion/simulator/configs/wan22_central_pull_max_risk_50.yaml \
  --runs 1 \
  --requests-output results/simulator/wan22_requests.csv \
  --trace-output results/simulator/wan22_events.jsonl

python3 -m benchmarks.diffusion.simulator.trace_report \
  --events results/simulator/wan22_events.jsonl \
  --requests results/simulator/wan22_requests.csv \
  --output results/simulator/wan22_trace.html
```

The report writes one self-contained HTML file, two standalone SVG charts, and
a JSON summary. The backend chart renders each request as one continuous block
from its first start through completion. Blocks are colored by short/medium/long,
Tail work is hatched, and idle intervals are classified as:

- `local-pending`: the backend is idle despite assigned unfinished work;
- `global-waiting`: an unstarted request exists elsewhere;
- `arrival-gap`: no request is ready on that backend before later arrivals;
- `drain-imbalance`: other backends are still draining after all arrivals.

The request chart ranks requests by queue time before first execution and shows
that queue interval followed by one continuous execution interval. Internal
phases and preemption slices are intentionally not expanded. Use `--seed` when
the input files contain more than one run,
`--bubble-threshold-s` to suppress tiny gaps, and `--max-requests` to control
the request chart.

## Explicit Paired Sweeps

Use a sweep matrix to compare a small, reviewed list of complete policy
variants. The sweep runner does not expand parameter value lists or construct
an automatic parameter Cartesian product. Every scenario and variant is named
explicitly:

```yaml
version: 1
name: wan22-p95-local-schedulers
seed: 42
runs: 20
baseline: fifo-no-preempt

scenarios:
  - name: n50-r005-m4
    config: configs/wan22_p95_pect_sink.yaml
    overrides:
      - workload.num_requests=50
      - workload.request_rate=0.05

variants:
  - name: fifo-no-preempt
    overrides:
      - policy.scheduler.normal_order=fifo
      - policy.scheduler.preempt_normal_over_sacrificial=false
  - name: bounded-srpt-aging120
    overrides:
      - policy.scheduler.normal_order=bounded_srpt
      - policy.scheduler.max_bypass=null
      - policy.scheduler.aging_s=120.0
      - policy.scheduler.preempt_normal_over_sacrificial=true
```

Paths are resolved relative to the matrix file. A scenario normally supplies
the simulator `config`; a variant may supply its own `config` to replace it.
Scenario overrides are applied first and variant overrides second. At least one
of the two items must supply a config for every scenario/variant pair.

Run the matrix and optionally retain the per-seed pairs:

```bash
python3 -m benchmarks.diffusion.simulator.sweep \
  --matrix benchmarks/diffusion/simulator/my_sweep.yaml \
  --output results/simulator/my_sweep.json
```

The matrix-level `seed` and `runs` are forced on every experiment. Results are
paired by seed against the named baseline and ranked by mean paired P95 latency
reduction:

```text
mean_seed((baseline_p95_seed - candidate_p95_seed) / baseline_p95_seed)
```

Positive percentages mean lower P95. This paired mean weights every seed
equally and can differ slightly from the reduction computed from the two
reported mean P95 values. Duplicate names, a missing baseline, per-item
seed/run overrides, missing configs, and mismatched result seeds are rejected.

## Configuration Model

Traffic can be specified as an absolute `request_rate`, or as normalized
`utilization`. For the latter, the simulator uses
`rate = utilization * sum(backend speed) / weighted mean service time`.
The weighted service time comes from the selected timing model.

Policy-visible work uses `estimated_service_s` when present, then
`nominal_service_s`, then the timing model prediction. Actual and estimate
noise are independent mean-one lognormal multipliers controlled by their
respective `*_sigma` fields. `service.actual_service_scale` is a final
calibration multiplier on actual phase time and deliberately does not change
the estimate visible to policies.

### Fixed Anchor Timing

`service.timing_model.type: fixed_anchor` is the default, so old YAML files and
results remain unchanged. It requires `nominal_service_s`. The optional
`service.hsdp` block retains the original calibration-friendly approximation:

```text
scaled_compute = nominal_service * actual_service_scale
hsdp_communication = scaled_compute * denoise_fraction
                     * communication_overhead_weight
actual_service = scaled_compute + hsdp_communication
```

For example, weight `0.25` means communication adds 25% of denoise compute.
When all service is denoise, communication is consequently 20% of final
service time. `shard_size` records the intended deployment mode; the weight,
not the shard size alone, controls timing until device measurements are
available. Disabled HSDP ignores the configured weight, which makes toggling a
candidate cheap.

### Coarse Wan2.2 Analytic Timing

`service.timing_model.type: analytic_wan22` replaces actual anchor time with
this phase decomposition:

```text
service =
    text encode + latent preparation
  + denoise compute + denoise overhead
  + exposed USP communication + exposed HSDP communication
  + VAE decode + postprocess
```

It first applies the production Wan2.2 shape rules: width and height are
rounded down to the VAE/patch multiple, frames become `4k+1`, and the resulting
latent is patchified into transformer tokens. One transformer forward uses the
following coarse work split, where
`S = ceil(tokens / usp_degree) * usp_degree`:

```text
F_parallel = L * (12*S*D^2 + 4*S*D*FFN + 4*S^2*D)
F_replicated = L * (4*C*D^2 + 4*S*C*D)

shape_efficiency =
    clamp((S / reference_tokens)^exponent, min_efficiency, 1)

T_compute = forwards
            / (effective_device_TFLOPS * parallel_efficiency * shape_efficiency)
            * (F_parallel / usp_degree + F_replicated) / 1e12
```

The text K/V projection and cross-attention terms are repeated on each current
Ulysses rank, so they do not receive ideal `1 / usp_degree` scaling. `forwards`
is denoise steps times sequential CFG passes.
`effective_compute_tflops_per_device` and
`parallel_compute_efficiency` convert work to time; they are effective phase
rates, not hardware peak specifications.

USP traffic counts aggregate Q/K/V/output collectives for self- and
cross-attention plus the final output gather. HSDP traffic estimates each
block's BF16 parameter All-Gather. Both are per-rank remote payloads converted
with separate effective bandwidth and collective-latency settings. Configured
overlap fractions retain only exposed communication time. These calculations
produce one time per phase or denoise step; they do not create kernel- or
collective-level simulator events.

Architecture, execution, effective-rate, overlap, and coarse fixed-stage values
live under the YAML `timing_model` block; the HSDP enable/shard controls remain
under `service.hsdp`. The bundled values are explicit assumptions chosen for a
useful first model, not measured 910B performance. A small calibration set is
enough to replace them: single-request text/denoise/VAE phase timings plus
representative USP and HSDP collective rates. No full concurrent benchmark is
required for every coefficient.

With analytic timing, the legacy `encode_fraction`, `decode_fraction`, and
`hsdp.communication_overhead_weight` must be zero to prevent double counting.
`nominal_service_s` is optional, although the bundled benchmark-aligned config
keeps the production anchors as policy estimates.

Combined USP/HSDP uses the same degree for both dimensions. Standalone HSDP is
also supported by setting `usp_degree: 1`, enabling HSDP, and setting each
backend's device count to `hsdp.shard_size`. `ulysses_mode: strict` validates
that the configured attention-head count is divisible by the USP degree;
`advanced_uaa` permits non-divisible degrees.

The built-in component types are deliberately small and composable:

- classifiers: `all_normal`, `quota_tail`, `online_credit_tail`,
  `quantile_risk_tail`;
- routers: `weighted_least_load`, `projected_completion`, `round_robin`,
  `least_inflight`;
- schedulers: `fifo`, `two_queue` (`fifo`, `lifo`, `srpt`,
  `bounded_srpt`, `least_laxity`, `arrival_plus_cost`, `size_class_fifo`, or
  `bounded_size_class_fifo` normal ordering).

Copying one of the bundled YAML files is the intended path for trying a new
combination. New component logic stays isolated in `policies.py`.

`quota_tail` can optionally enforce `max_sacrificial`, restrict eligibility
with `eligible_request_types`, and release credits at explicit one-based
measured-request positions through `credit_release_requests`. Explicit release
positions are independent of the production-compatible
`initial_arrival_counter`, so warmup state cannot shift a finite benchmark's
tail budget.

`online_credit_tail` never reads `workload.num_requests`, fixed release
positions, or an end-of-cohort signal. In `prefix_safe` mode, the safe Tail
allowance after \(n\) arrivals is
`floor((1 - target_quantile) * (n - 1))`; this requires classifier state to
start with the measured stream, but not its final size. In
`smooth_token_bucket` mode, `token_rate: 0.04` and `credit_capacity: 1` keep
Tail admissions at least 25 arrivals apart, so the classifier can remain alive
across unknown measurement-window boundaries. `congestion_threshold` and
`cooldown_requests` are optional online consumption gates.

`quantile_risk_tail` keeps the same hard Tail budget and optional explicit
credit positions, but admits an eligible request only when its counterfactual
Normal completion risk exceeds a quantile of prior eligible requests. The risk
uses only current remaining backend loads and the request's estimated service;
the current request is excluded from its own threshold. A request admitted to
Tail still contributes its shadow Normal risk to later thresholds, avoiding
selection-feedback bias. `history_window`, `min_observations`, and
`history_eligible_only` control the online reference population. It never
reads actual service durations or future arrivals.

`projected_completion` routes protected requests by estimated backend work plus
the incoming request's own service. Its `tail_mode` can spread tail work, pack
it onto an existing tail backend, or use `sink` to place it behind the largest
protected backlog. The `remaining` load view is an idealized simulator view;
production needs progress telemetry or a time-decayed estimate to reproduce it.

`two_queue` can also enable protected-first work stealing. An idle backend may
rebind a `WAITING`, never-started protected request from another backend; it
never migrates running state or Tail work. `steal_hysteresis_s` requires enough
predicted drain-time benefit before moving, and `steal_cost_s` occupies the
receiving backend as explicit dispatcher overhead.

Alternatively, `global_protected_pull` implements true deferred binding.
Normal requests remain in a simulator-owned global queue with no backend
assignment or backend load accounting until an idle backend, or a preemptible
Tail step boundary, pulls one. `protected_pull_order` is `fifo`, `max_risk`
(online age plus estimated service on the target backend),
`arrival_plus_cost`, or `highest_response_ratio`, and
`protected_pull_cost_s` charges every first binding. Started Normal requests
and all Tail requests remain backend-affine. Central Pull and work stealing are
mutually exclusive.

## Included Configurations

- `wan22_super_p95_current.yaml`: current code semantics, cold-start classifier,
  normalized-utilization traffic, and multi-seed aggregation.
- `wan22_super_p95_idealized.yaml`: report-style classifier plus live remaining
  load and completion-latency feedback.
- `wan22_benchmark_rps005_current.yaml`: the documented 50-request, 0.05 rps,
  seed-42 case. It initializes classifier state left by the benchmark's default
  warmup request.
- `wan22_2xusp4_calibrated.yaml` and `wan22_4xusp2_calibrated.yaml`: the same
  seed-42 workload and original `super_p95_step` policy, with topology-specific
  actual service anchors measured from four serial E2E requests per
  short/medium/long profile. The policy-visible estimates remain the production
  Wan2.2 estimator anchors. Each topology uses one shared actual-service scale
  jointly fitted to its 50- and 100-request P95 results.
- `wan22_4xusp2_pect_calibrated.yaml`: the same measured 4×USP2 timing profile
  with an assigned-load projected-completion router for online policy
  composition.
- `wan22_4xusp2_online_policy_screen.yaml`,
  `wan22_4xusp2_central_pull_screen.yaml`, and
  `wan22_4xusp2_central_pull_holdout.yaml`: online-only 4×USP2 policy screens
  followed by an independent 400-seed validation of Central Pull + Max Risk.
- `wan22_fifo.yaml`: all-normal FIFO control on the same theoretical `2xUSP4`
  topology.
- `wan22_hsdp4_sensitivity.yaml`: `2 x USP4 + HSDP4`, using the old M4 QPS as
  a compute-time prior and an assumed HSDP communication weight of `0.25`.
- `wan22_hsdp2_sensitivity.yaml`: `4 x USP2 + HSDP2`, using an extrapolated
  compute-time prior and the same communication assumption.
- `wan22_analytic_usp4.yaml`: benchmark-aligned `2 x USP4` workload using
  input-derived FLOPs and D2D volume. HSDP4 can be enabled with one override.
- `wan22_p95_pect_sink.yaml`: exact two-request tail cap, request-aware
  projected-completion routing, soft tail sink, and FIFO protected scheduling
  on the same analytic workload.
- `wan22_centered_cohort_pect_pack_steal_50.yaml`: previous 50-request
  Work-Stealing incumbent on `4 x USP2`, with safe Tail budget 2, centered gates, PECT
  remaining-load routing, Tail-Pack, FIFO, and protected-first work stealing.
- `wan22_centered_cohort_pect_pack_steal_100.yaml`: the matching 100-request
  profile with safe Tail budget 4 and four centered gates.
- `wan22_p95_policy_comparison.yaml`: explicit 400-seed paired comparison of
  the normal `1 x USP8` baseline, current strategy, recommended core, work
  stealing, and Tail preemption ablation across `2 x USP4` and `4 x USP2` for
  both benchmark sizes.
- `wan22_quantile_risk_pect_pack_50.yaml`: configurable online
  Quantile-Risk Tail admission combined with the current PECT-Remaining,
  Tail-Pack, and FIFO core.
- `wan22_quantile_risk_training.yaml`: frozen six-candidate, 100-seed screen
  over three risk quantiles and full versus recent eligible-risk history. It
  uses seeds 562--661 and deliberately leaves the independent holdout closed.
- `wan22_central_pull_max_risk_50.yaml` and
  `wan22_central_pull_max_risk_100.yaml`: current M2 simulation recommendation,
  using a global Max-Risk Normal queue and a configurable `0.5s` pull-cost
  placeholder.
- `wan22_central_pull_training.yaml`: uniform Central Pull screen on seeds
  662--761. It establishes the zero-cost upper bound and rejects a uniform
  two-second configuration.
- `wan22_topology_conditional_central_pull_training.yaml` and
  `wan22_topology_conditional_central_pull_holdout.yaml`: independent
  100-seed training and 400-seed holdout for the topology-conditional policy:
  keep Work Stealing on M4 and enable Max-Risk Central Pull only on M2.

Run the reproducible policy comparison:

```bash
python3 -m benchmarks.diffusion.simulator.sweep \
  --matrix benchmarks/diffusion/simulator/configs/wan22_p95_policy_comparison.yaml \
  --output results/simulator/wan22_p95_policy_comparison.json
```

Run the Quantile-Risk admission screen and seed-block statistical analysis:

```bash
python3 -m benchmarks.diffusion.simulator.sweep \
  --matrix benchmarks/diffusion/simulator/configs/wan22_quantile_risk_training.yaml \
  --output results/simulator/wan22_quantile_risk_training.json

python3 -m benchmarks.diffusion.simulator.sweep_stats \
  --input results/simulator/wan22_quantile_risk_training.json \
  --output results/simulator/wan22_quantile_risk_training_stats.json
```

The statistics tool first averages the paired reduction across scenarios
within each seed, then resamples whole seed blocks. It reports a deterministic
20,000-sample paired bootstrap interval, a one-sided paired sign-flip test, and
Holm-Bonferroni adjusted p-values. By default every non-baseline variant enters
the hypothesis family; repeat `--hypothesis-variant` to declare only
pre-registered candidates and leave other variants as diagnostics. This avoids
treating the four correlated scenarios for one seed as independent samples.

Run the promoted M2 simulation configurations directly:

```bash
python3 -m benchmarks.diffusion.simulator \
  --config benchmarks/diffusion/simulator/configs/wan22_central_pull_max_risk_50.yaml

python3 -m benchmarks.diffusion.simulator \
  --config benchmarks/diffusion/simulator/configs/wan22_central_pull_max_risk_100.yaml
```

Reproduce the sealed topology-conditional holdout:

```bash
python3 -m benchmarks.diffusion.simulator.sweep \
  --matrix benchmarks/diffusion/simulator/configs/wan22_topology_conditional_central_pull_holdout.yaml \
  --output results/simulator/wan22_topology_conditional_central_pull_holdout.json

python3 -m benchmarks.diffusion.simulator.sweep_stats \
  --input results/simulator/wan22_topology_conditional_central_pull_holdout.json \
  --output results/simulator/wan22_topology_conditional_central_pull_holdout_stats.json \
  --hypothesis-variant \
  'M4-Work-Stealing+M2-Central-Pull-Max-Risk+Half-Second-Cost'
```

Run a communication sensitivity sweep without editing code:

```bash
python3 -m benchmarks.diffusion.simulator \
  --config benchmarks/diffusion/simulator/configs/wan22_hsdp4_sensitivity.yaml \
  --set service.hsdp.communication_overhead_weight=0.10
```

The current and idealized policies deliberately remain separate. Current Wan
video routing holds each request's full assigned estimate until completion and
does not update latency EMA on the asynchronous completion path. The idealized
profile uses phase-aware remaining work and completion latency instead.

## Simulation Semantics

- A run contains a finite set of requests and drains every request before
  metrics are calculated.
- The primary latency is system E2E: generated arrival to completion. P95 uses
  the same linear percentile rule as NumPy's default.
- Request profiles and Poisson inter-arrivals use separate seeded random-number
  generators, matching `diffusion_benchmark_serving.py`.
- A new request atomically executes encode and its first denoise step. Its final
  denoise step atomically flows into decode. Scheduling occurs only at
  intermediate denoise-step boundaries.
- Preemption keeps completed progress and never repeats encode or a completed
  step. Optional preemption cost occupies the backend without advancing work.
- Policies see only arrived request metadata and estimated work. True sampled
  duration and future arrivals remain inside the engine.
- Each backend has one execution slot. Communication is represented as an
  exposed per-denoise-step time term. Analytic timing estimates layer
  collective bytes and launch counts, but does not schedule the collectives as
  separate events or model shared-fabric contention.
- The simulator does not decide HBM capacity/OOM, model loading/cold start,
  layerwise offload, or batch execution. Those need separate models or device
  measurements.
- Batch-aware queues and simultaneous request groups are explicitly deferred;
  adding batch-2 will require an engine interface extension rather than a YAML
  switch. Wan2.2 remains one request per backend in this version.

The benchmark client's semaphore starts its latency stopwatch after client-side
admission. This simulator instead measures from generated arrival, as agreed for
the scheduling objective. With `max_concurrency >= num_requests` (the documented
Wan case), the distinction does not introduce an extra client queue.

## Extending A Policy

Numeric variants only require YAML edits. A new scheduling idea normally needs
one small implementation in `policies.py` and one factory entry:

- classifier: labels a newly arrived request;
- router: chooses one backend from current online views;
- local scheduler: chooses the incumbent or one pending request at a scheduling
  boundary.

The engine owns queues, actual duration, future events, and state transitions so
policy code cannot accidentally gain offline or oracle information.

## Outputs

The console prints the multi-run P95 mean, an approximate 95% confidence
interval for that mean, and throughput. `--output` stores resolved
configuration, every run's metrics, and aggregates. Per-request CSV and event
JSONL are opt-in to keep Monte Carlo runs lightweight. Request records include
all phase times plus normalized dimensions, token count, FLOPs, per-rank
communication bytes, and collective-call counts when analytic timing is used.
Run metrics include matching stage totals and communication fractions.
