"""Real compiled collective execution, private spill backing and failure probes."""

import json
from pathlib import Path

import pytest

from codesign.ventus.compiled_check import prepare
from codesign.ventus.config import Hardware
from codesign.ventus.elf import ELFProgram
from codesign.ventus.rust import PreparedProgram, RustSession

ROOT = Path(__file__).resolve().parents[1]


def decoded(buffer):
    from codesign.ventus.rust import compact_records

    for r in compact_records(buffer):
        yield r["kind"], r["block"], r["warp"], r["addresses"]


@pytest.mark.parametrize("case_index", [0, 1])
def test_compiled_repeated_exchange_tails_streaming_and_slot_reuse(case_index):
    raw = json.loads((ROOT / "examples/compiled-sync-check.json").read_text())
    case = raw["cases"][case_index]
    elf = ELFProgram.read(ROOT / "tests/fixtures/compiled-workgroup/exchange.elf")
    inputs = {
        b["address"] + 4 * i: value for b in case["inputs"] for i, value in enumerate(b["words"])
    }
    launch = elf.launch(
        case["kernel"],
        case["arguments"],
        global_size=case["global_size"],
        local_size=case["local_size"],
        functional_words=inputs,
    )
    # Three workgroups exceed one SM's four warp slots, exercising recycling
    # of per-block dependency IDs as well as repeated barriers and private spills.
    hw = Hardware(sms=1, warps_per_sm=4, blocks_per_sm=4, collectors=4, lsu_entries=4)
    with RustSession(hw) as expanded, RustSession(hw) as streamed:
        buffer = expanded.generate(elf.words, launch)
        ops = list(decoded(buffer))
        count, rounds = case["arguments"][-2:]
        output = case["outputs"][0]["address"]
        assert sorted(
            a for k, _, _, addresses in ops if k == "store" for a in addresses if a >= output
        ) == list(range(output, output + 4 * count * rounds, 4))
        for _ in range(2):
            a = expanded.dispatch(buffer)
            b = streamed.dispatch(PreparedProgram.from_elf(elf, launch))
            assert a == b
            assert a["counters"]["barrier"] == launch[3] * launch[4] * 2 * rounds
            assert expanded.cache_state() == streamed.cache_state()


def instruction(imm, rs, fun, rd, opcode=0x13):
    return (imm & 0xFFF) << 20 | rs << 15 | fun << 12 | rd << 7 | opcode


def branch(rs, offset):
    return (
        (offset >> 12 & 1) << 31
        | (offset >> 5 & 63) << 25
        | rs << 15
        | (offset >> 1 & 15) << 8
        | (offset >> 11 & 1) << 7
        | 0x63
    )


def store(rs, base):
    return rs << 20 | base << 15 | 2 << 12 | 0x23


def launch_probe(initial=None):
    elf = ELFProgram.read(ROOT / "tests/fixtures/compiled-vecadd/single.elf")
    launch = elf.launch(
        "vecadd_f32", [], global_size=(64, 1, 1), local_size=(64, 1, 1), functional_words=initial
    )
    launch[6:9] = [8, 16, 128]
    return launch


def exchange_probe(flags=1, base=0x70000000):
    return [
        0x805022F3,  # csrr x5, warp-in-group
        base | 8 << 7 | 0x37,
        instruction(2, 5, 1, 6),  # byte offset = warp * 4
        8 << 20 | 6 << 15 | 6 << 7 | 0x33,
        instruction(7, 5, 0, 7),  # known scalar value = warp + 7
        store(7, 6),
        0x0400400B | flags << 15,
        instruction(4, 6, 4, 6),  # peer address swaps two warps
        instruction(0, 6, 2, 7, 0x03),
        instruction(2, 7, 1, 7),
        0x90000437,
        7 << 20 | 8 << 15 | 8 << 7 | 0x33,
        store(5, 8),  # loaded shared value controls address
        0x00008067,
    ]


def test_barrier_publishes_other_warp_values_for_address_execution():
    with RustSession(Hardware()) as session:
        ops = decoded(session.generate(exchange_probe(), launch_probe()))
        outputs = [
            a for k, _, _, addresses in ops if k == "store" for a in addresses if a >= 0x90000000
        ]
        assert sorted(outputs) == [0x9000001C, 0x90000020]


def test_global_memory_fence_publishes_global_values():
    with RustSession(Hardware()) as session:
        ops = decoded(session.generate(exchange_probe(2, 0x91000000), launch_probe()))
        outputs = [
            a
            for k, _, _, addresses in ops
            if k == "store"
            for a in addresses
            if 0x90000000 <= a < 0x91000000
        ]
        assert sorted(outputs) == [0x9000001C, 0x90000020]


def test_actual_compiled_cooperative_gemm_preserves_tails_and_fma_work():
    case = json.loads((ROOT / "examples/compiled-gemm-check.json").read_text())["cases"][0]
    elf = ELFProgram.read(ROOT / "tests/fixtures/compiled-workgroup/gemm.elf")
    inputs = {
        b["address"] + 4 * i: value for b in case["inputs"] for i, value in enumerate(b["words"])
    }
    launch = elf.launch(
        case["kernel"],
        case["arguments"],
        global_size=case["global_size"],
        local_size=case["local_size"],
        functional_words=inputs,
    )
    hw = Hardware(sms=1, warps_per_sm=2, blocks_per_sm=2, collectors=2, lsu_entries=2)
    with RustSession(hw) as expanded, RustSession(hw) as streamed:
        compact = expanded.generate(elf.words, launch)
        ops = list(decoded(compact))
        output = case["arguments"][2]
        m, n, k = case["arguments"][3:]
        assert sorted(
            a for kind, _, _, addresses in ops if kind == "store" for a in addresses if a >= output
        ) == list(range(output, output + m * n * 4, 4))
        a = expanded.dispatch(compact)
        b = streamed.dispatch(PreparedProgram.from_elf(elf, launch))
        assert a == b
        assert a["counters"]["barrier"] == launch[3] * launch[4] * ((k + 7) // 8) * 2
        assert a["counters"]["fma"] == launch[3] * launch[4] * ((k + 7) // 8) * 8


@pytest.mark.parametrize("flags", [0, 2])
def test_wrong_memory_fence_cannot_publish_local_values(flags):
    with RustSession(Hardware()) as session:
        with pytest.raises(ValueError, match="unsynchronized cross-warp"):
            session.generate(exchange_probe(flags), launch_probe({0x70000000: 0, 0x70000004: 0}))
        assert session.cycles == 0


@pytest.mark.parametrize(
    "words,reason",
    [
        ([0x805022F3, branch(5, 8), 0x0400C00B, 0x00008067], "missing arrival"),
        (
            [0x805022F3, branch(5, 12), 0x0400C00B, 0x0080006F, 0x0400C00B, 0x00008067],
            "PC or flags differ",
        ),
        (
            [instruction(16, 0, 0, 5), 5 << 15 | 7 << 12 | 0x57, 0x0400C00B, 0x00008067],
            "divergent collective",
        ),
        ([0x700002B7, store(0, 5), 0x00008067], "unsynchronized cross-warp"),
    ],
)
def test_invalid_collectives_and_races_fail_explicitly(words, reason):
    with RustSession(Hardware()) as session:
        with pytest.raises(ValueError, match=reason):
            session.generate(words, launch_probe())
        assert session.cycles == 0


def test_rtl_private_spill_backing_covers_every_launched_thread(tmp_path):
    receipt = prepare(ROOT / "examples/compiled-sync-check.json", tmp_path / "input")
    for case in receipt["cases"]:
        private = case["private_backing"]
        # Compiler declares 16 private spill bytes/thread; all three padded
        # workgroups need backing, including inactive tail output lanes.
        assert private["bytes"] == 3 * case["launch_resources"]["warps"] * 32 * 16
        directory = tmp_path / "input" / case["name"]
        words = [int(line, 16) for line in (directory / "input.metadata").read_text().splitlines()]
        meta = [a | b << 32 for a, b in zip(words[::2], words[1::2], strict=True)]
        n = meta[13]
        bases, sizes = meta[14 : 14 + n], meta[14 + n : 14 + 2 * n]
        assert dict(zip(bases, sizes, strict=True))[private["base"]] == private["bytes"]
