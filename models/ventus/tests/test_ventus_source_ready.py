import pytest

from codesign.ventus.cache_acceptance import AcceptanceBuilder
from codesign.ventus.config import Hardware
from codesign.ventus.ir import Operation, Workload
from codesign.ventus.source_ready import SourceReadyFrontend, rr_ready


def test_rr_ready_includes_invalid_inputs_before_winner():
    # Last served input 1; input 3 wins. Empty input 2 is ready too.
    assert rr_ready([True, False, False, True], 1, True) == (False, False, True, True)
    # No masked request: wrap to input 1; empty 0, 2, 3 may also load packets.
    assert rr_ready([False, True, False, False], 1, True) == (True, True, True, True)
    assert rr_ready([False] * 4, 2, False) == (False,) * 4
    assert rr_ready([False] * 4, 2, True) == (True,) * 4


def engine():
    work = Workload(
        (
            Operation("pc80000000", "vector", destination="v1"),
            Operation("pc80000004", "scalar", destination="x2"),
        )
    )
    model = SourceReadyFrontend(AcceptanceBuilder(Hardware()), work)
    model.allocate()
    return model


def test_empty_unpack_uses_stale_control_not_next_program_hazards():
    model = engine()
    key = (0, 0)
    model.packet_loaded(key, ["pc80000000", "pc80000004"])
    model.registers[key, "v1"] = "pending"
    assert key not in model.unpack_ready()
    model.completion["pending"] = model.b.node("done", [(0, 0)])
    model.clock = 1
    assert key in model.unpack_ready()


def test_collector_full_backpressures_even_empty_unpack():
    model = engine()
    model.collectors[0] = dict.fromkeys(range(model.hw.collectors), object())
    assert (0, 0) not in model.unpack_ready()


def test_unrealized_geometry_is_not_silently_treated_as_pinned_demux():
    model = engine()
    with pytest.raises(ValueError, match="collectors"):
        SourceReadyFrontend(
            AcceptanceBuilder(Hardware().with_changes(collectors=4)), model.workload
        )
