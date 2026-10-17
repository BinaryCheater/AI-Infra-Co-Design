from codesign.ventus.cache_acceptance import AcceptanceBuilder
from codesign.ventus.config import Hardware
from codesign.ventus.ir import Operation


def builder():
    return AcceptanceBuilder(Hardware().with_changes(memory_latency=100))


def test_secondary_full_backpressures_address_path_until_return_drain():
    b = builder()
    for i in range(3):
        issued = b.node(f"issue{i}", [(0, i * 4)])
        b.memory_instruction(issued, Operation(f"load{i}", "load", addresses=(0x90000000,)), 0, i)
        if i == 1:
            pending = b.cache_pending[0][0]
            assert pending["targets"] == 2
            release = b.times[pending["release"]]
    last_accept = b.times[b.cache_request_cursor[0]]
    assert last_accept >= release
    assert b.times[b.resources[("lsu.address.release", 0)][-1]] >= last_accept + 1
    assert b.graph.counters["l1_read_miss"] == 1


def test_return_drain_stalls_unrelated_line_but_not_earlier_admission():
    b = builder()
    b.global_line(0, 0, 0x90000000, False)
    pending = b.cache_pending[0][0]
    before = b.node("before_return", [(0, b.times[pending["return"]] - 1)])
    accepted, _ = b._cache_gate(before, 0, 0x90001000)
    assert b.times[accepted] == b.times[before]
    accepted, _ = b._cache_gate(pending["return"], 0, 0x90001000)
    assert b.times[accepted] == b.times[pending["release"]]


def test_primary_full_waits_for_first_release():
    b = builder()
    for i in range(4):
        b.global_line(b.node(f"request{i}", [(0, i)]), 0, 0x90000000 + 128 * i, False)
    gate, _ = b._cache_gate(b.node("fifth", [(0, 5)]), 0, 0x90001000)
    assert b.times[gate] >= min(b.times[p["release"]] for p in b.cache_pending[0])
