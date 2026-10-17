from collections import Counter

import pytest

from codesign.ventus.causal_memory import CacheRequest, CausalDataCache, predict
from codesign.ventus.config import Hardware
from codesign.ventus.ir import Operation, Workload
from codesign.ventus.online import Token
from codesign.ventus.physical_operands import with_physical_operands
from codesign.ventus.rtl_accuracy import complete_gemm
from codesign.ventus.termination import append_endprg
from codesign.ventus.timing import Builder, Line


def request(name, line=0x90000000, *, kind="load", group=0):
    op = Operation(name, kind, destination="v1" if kind == "load" else None, addresses=(line,))
    return CacheRequest(Token(op, 0, 0, 0), 0, line, group, 0)


def test_hit_has_nonflow_request_and_response_stages():
    b = Builder(Hardware())
    cache = CausalDataCache(b, 0)
    req = request("hit")
    b.l1[0, (req.line // b.hw.line_bytes) % b.hw.l1_sets][req.line] = Line(0)
    assert cache.step(0, core_ready=True) is None
    assert cache.accept(req)
    assert cache.step(1, core_ready=True) is None
    assert cache.step(2, core_ready=True) is None
    response, node = cache.step(3, core_ready=True)
    assert response.token.operation.name == "hit"
    assert b.times[node] == 3


def test_second_target_allocates_when_accepted_and_drains_exactly_once():
    cache = CausalDataCache(Builder(Hardware()), 0)
    cache.step(0, core_ready=False)
    assert cache.accept(request("a"))
    cache.step(1, core_ready=False)
    assert not cache.accept(request("b"))  # same-address missReq in this stage
    cache.step(2, core_ready=False)
    assert cache.accept(request("b"))
    received = []
    for cycle in range(3, 80):
        result = cache.step(cycle, core_ready=cycle >= 40)
        if result:
            received.append(result[0].token.operation.name)
    assert received == ["a", "b"]
    alloc = [e for e in cache.events if e["kind"] == "target_allocate"]
    assert [(e["cycle"], e["slot"]) for e in alloc] == [(1, 0), (3, 1)]
    assert not cache.primary and not cache.lower_pending and not cache.lower_returns
    drains = [e for e in cache.events if e["kind"] == "target_drain"]
    releases = [e for e in cache.events if e["kind"] == "primary_release"]
    assert len(drains) == 2 and len(releases) == 1
    assert releases[0]["cycle"] > max(e["cycle"] for e in drains)


def test_full_core_response_queue_preserves_every_request_under_backpressure():
    b = Builder(Hardware())
    cache = CausalDataCache(b, 0)
    line = 0x90000000
    b.l1[0, (line // b.hw.line_bytes) % b.hw.l1_sets][line] = Line(0)
    accepted, received = [], []
    for cycle in range(180):
        result = cache.step(cycle, core_ready=cycle >= 80)
        if result:
            received.append(result[0].token.operation.name)
        if len(accepted) < 45:
            req = request(f"load{len(accepted)}")
            if cache.accept(req):
                accepted.append(req.token.operation.name)
        assert len(cache.core_response) <= b.hw.threads
        assert len(cache.lower_returns) <= 2
    assert len(accepted) == 45 and received == accepted


def test_primary_full_blocks_other_line_until_actual_release():
    cache = CausalDataCache(Builder(Hardware().with_changes(l1_mshrs=1)), 0)
    cache.step(0, core_ready=True)
    assert cache.accept(request("a"))
    accepted_at = None
    for cycle in range(1, 80):
        cache.step(cycle, core_ready=True)
        if accepted_at is None and cache.accept(request("b", 0x90000100)):
            accepted_at = cycle
    release = next(e["cycle"] for e in cache.events if e["kind"] == "primary_release")
    assert accepted_at > release


@pytest.mark.parametrize("warps,strategy", [(1, "stream"), (3, "shared"), (6, "reload")])
def test_complete_probe_conserves_transactions_and_executes_graph(warps, strategy):
    program = complete_gemm(5, warps, strategy)
    work = with_physical_operands(append_endprg(program.workload, program.words), program.words)
    result = predict(
        work,
        Hardware(),
        endpoint_names=[f"w{w}.pc{program.final_tensor_pc:08x}" for w in range(warps)],
        trace=True,
    )
    assert result["prediction_cycles"] > 0
    for events in result["cache_events"]:
        accepts = Counter((e["name"], e["group"]) for e in events if e["kind"] == "core_accept")
        responses = Counter((e["name"], e["group"]) for e in events if e["kind"] == "core_response")
        assert accepts == responses
    assert result["counters"]["tensor"] == 5 * warps


def test_multiline_load_waits_for_all_groups_and_unsupported_endpoints_fail():
    work = Workload(
        (
            Operation("pc80000000", "load", destination="v1", addresses=(0x90000000, 0x90000100)),
            Operation("pc80000004", "endprg"),
        )
    )
    result = predict(work, Hardware(), endpoint_names=["pc80000000"], trace=True)
    responses = [e for e in result["cache_events"][0] if e["kind"] == "core_response"]
    assert len(responses) == 2
    assert result["instruction_events"]["pc80000000"]["writeback"] > max(
        e["cycle"] for e in responses
    )
    assert result["counters"]["causal_partial_returns"] == 1
    with pytest.raises(ValueError, match="endpoint"):
        predict(work, Hardware(), endpoint_names=["pc80000004"])


def test_sequential_blocks_reuse_cache_without_reusing_request_identity():
    ops = []
    for block in range(2):
        prefix = f"b{block}."
        ops.extend(
            [
                Operation(
                    prefix + "pc80000000",
                    "load",
                    block=block,
                    destination="v1",
                    addresses=(0x90000000,),
                ),
                Operation(
                    prefix + "pc80000004", "vector", block=block, sources=("v1",), destination="v2"
                ),
                Operation(prefix + "pc80000008", "endprg", block=block),
            ]
        )
    result = predict(
        Workload(tuple(ops)),
        Hardware().with_changes(sms=1, blocks_per_sm=1),
        endpoint_names=["b0.pc80000004", "b1.pc80000004"],
        trace=True,
    )
    assert result["counters"]["l1_read_miss"] == 1
    assert result["counters"]["l1_hit"] == 1
    assert result["endpoint_times"]["b1.pc80000004"] > result["endpoint_times"]["b0.pc80000004"]


def test_mixed_load_store_traffic_drains_after_variable_backpressure():
    cache = CausalDataCache(Builder(Hardware().with_changes(l1_mshrs=2, l1_sets=2, l1_ways=2)), 0)
    requests = [
        request(f"req{i}", 0x90000000 + (i % 7) * 128, kind="store" if i % 3 == 0 else "load")
        for i in range(70)
    ]
    accepted, returned = [], []
    for cycle in range(2000):
        result = cache.step(cycle, core_ready=cycle % 11 not in {0, 1, 2, 3})
        if result:
            returned.append(result[0].token.operation.name)
        if requests and cache.accept(requests[0]):
            accepted.append(requests.pop(0).token.operation.name)
        if not requests and len(returned) == len(accepted):
            break
    assert len(accepted) == 70
    assert Counter(accepted) == Counter(returned)
    assert not cache.primary and not cache.lower_pending and not cache.lower_returns
