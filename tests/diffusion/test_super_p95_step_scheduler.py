import heapq

from vllm_omni.diffusion.sched.super_p95_step_scheduler import _QueuedRequest


def test_super_p95_step_scheduler_keeps_normal_fifo():
    first = _QueuedRequest(
        arrival_seq=0,
        arrival_time_s=0.0,
        sched_req_id="first",
        estimated_service_s=1.0,
        is_sacrificial=False,
    )
    second = _QueuedRequest(
        arrival_seq=1,
        arrival_time_s=0.0,
        sched_req_id="second",
        estimated_service_s=1.0,
        is_sacrificial=False,
    )

    heap = [second, first]
    heapq.heapify(heap)

    assert heapq.heappop(heap).sched_req_id == "first"


def test_super_p95_step_scheduler_uses_tail_lifo():
    first = _QueuedRequest(
        arrival_seq=0,
        arrival_time_s=0.0,
        sched_req_id="first",
        estimated_service_s=1.0,
        is_sacrificial=True,
    )
    second = _QueuedRequest(
        arrival_seq=1,
        arrival_time_s=0.0,
        sched_req_id="second",
        estimated_service_s=1.0,
        is_sacrificial=True,
    )

    heap = [first, second]
    heapq.heapify(heap)

    assert heapq.heappop(heap).sched_req_id == "second"
