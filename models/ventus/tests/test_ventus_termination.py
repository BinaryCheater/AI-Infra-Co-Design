"""Terminal instruction occupancy must not be replaced by a collective barrier."""

from dataclasses import replace

import pytest

from codesign.ventus.config import Hardware
from codesign.ventus.ir import Operation, Workload
from codesign.ventus.termination import append_endprg


def test_real_terminal_word_appended_to_each_interleaved_warp():
    body = Workload(
        tuple(Operation(f"w{w}.pc80000000", "scalar", warp=w) for w in range(2)), warps_per_block=2
    )
    result = append_endprg(body, [0x13, 0x400B])
    result.validate(Hardware())
    assert [op.name for op in result.operations[-2:]] == ["w0.pc80000004", "w1.pc80000004"]
    assert all(op.kind == "endprg" and not op.dependencies for op in result.operations[-2:])
    assert len(body.operations) == 2
    with pytest.raises(ValueError, match="follows ENDPRG"):
        replace(result, operations=result.operations + (Operation("late", "scalar"),)).validate(
            Hardware()
        )


def test_missing_word_gap_and_invalid_operands_rejected():
    body = Workload((Operation("pc80000000", "scalar"),))
    with pytest.raises(ValueError, match="final ENDPRG"):
        append_endprg(body, [0x13, 0x13])
    with pytest.raises(ValueError, match="cover all words"):
        append_endprg(body, [0x13, 0x13, 0x400B])
    with pytest.raises(ValueError, match="contiguous"):
        append_endprg(
            replace(body, operations=body.operations + (Operation("pc80000008", "scalar"),)),
            [0x13, 0x13, 0x400B],
        )
    with pytest.raises(ValueError, match="no register operands"):
        Workload((Operation("pc80000000", "endprg", sources=("x0",)),)).validate(Hardware())


@pytest.mark.parametrize("slots", [(0, 0), (0,), (0, 4), (False, 1), [0, 1]])
def test_physical_operand_slots_must_be_complete_and_unambiguous(slots):
    workload = Workload((Operation("op", "vector", sources=("x1", "v2"), source_slots=slots),))
    with pytest.raises(ValueError, match="source_slots"):
        workload.validate(Hardware())


def test_json_slot_metadata_survives_and_legacy_executors_reject_it():
    from codesign.ventus.__main__ import read_candidate
    from codesign.ventus.rust import encode
    from codesign.ventus.session import TimingSession

    candidate = read_candidate(
        {
            "workload": {
                "operations": [
                    {
                        "name": "pc80000000",
                        "kind": "vector",
                        "sources": ["v1", "x2"],
                        "source_slots": [1, 0],
                    }
                ]
            }
        }
    )
    assert candidate.workload.operations[0].source_slots == (1, 0)
    for workload in (candidate.workload, Workload((Operation("pc80000000", "endprg"),))):
        with pytest.raises(ValueError, match="use frontend evaluator"):
            encode(workload, Hardware())
        with pytest.raises(ValueError, match="use frontend evaluator"):
            TimingSession(Hardware()).dispatch(workload)
