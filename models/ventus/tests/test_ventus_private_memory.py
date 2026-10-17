"""Private-word isolation must match the explicitly selected driver layout."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from codesign.ventus.config import Hardware
from codesign.ventus.driver_adapter import audit_sources, verilator_headers
from codesign.ventus.elf import ELFProgram
from codesign.ventus.rust import RustSession

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/compiled-private-r1/program.elf"


def memory_ops(buffer):
    from codesign.ventus.rust import compact_records

    for r in compact_records(buffer):
        if r["kind"] in {"load", "store"}:
            yield (
                {"load": 8, "store": 9}[r["kind"]],
                r["block"],
                r["warp"],
                r["width"],
                r["addresses"],
            )


def program(profile):
    elf = ELFProgram.read(FIXTURE)
    raw = elf.launch(
        "private_spill_u32",
        [0x90002000, 128],
        global_size=(128, 1, 1),
        local_size=(64, 1, 1),
        private_profile=profile,
    )
    return elf, raw


def test_real_compiled_private_slots_are_disjoint_across_warps_and_blocks():
    elf, launch = program("r1-workgroup")
    assert elf.resources["private_spill_u32"]["pds_stack"] == 20
    with RustSession(Hardware()) as session:
        ops = list(memory_ops(session.generate(elf.words, launch)))
    owners = {}
    for _, block, warp, width, addresses in ops:
        for address in addresses:
            if 0xA0000000 <= address < 0xB0000000:
                assert address + width <= 0xA0000000 + 128 * 20
                owner = (block, warp)
                assert owners.setdefault(address, owner) == owner
    assert set(owners.values()) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    output = [
        a
        for kind, _, _, _, addresses in ops
        if kind == 9
        for a in addresses
        if 0x90002000 <= a < 0x90002000 + 384 * 4
    ]
    assert sorted(output) == list(range(0x90002000, 0x90002000 + 384 * 4, 4))


def test_upstream_per_warp_private_increment_exposes_cross_warp_alias():
    elf, launch = program("r1-upstream")
    with RustSession(Hardware()) as session:
        with pytest.raises(ValueError, match="cannot guarantee lane isolation"):
            session.generate(elf.words, launch)
        assert session.cycles == 0


def test_partial_private_word_cannot_escape_compiler_allocation():
    elf, launch = program("r1-workgroup")
    launch[-1] = 4
    with RustSession(Hardware()) as session:
        with pytest.raises(ValueError, match="private address exceeds compiler allocation"):
            session.generate(elf.words, launch)


def test_verilator_reports_build_default_and_actual_installation(tmp_path):
    real = tmp_path / "actual"
    (real / "include").mkdir(parents=True)
    (real / "include/verilated.h").write_text("// tool fixture\n")
    text = f"VERILATOR_ROOT = {tmp_path / 'build-default'}\nVERILATOR_ROOT = {real}\n"
    assert verilator_headers(text) == real
    with pytest.raises(ValueError, match="headers could not be located"):
        verilator_headers(text, str(tmp_path / "invalid-explicit-root"))


def test_subword_compiler_private_usage_reserves_a_whole_interleaved_word():
    elf, _ = program("r1-workgroup")
    metadata = {**elf.resources["private_spill_u32"], "pds_stack": 3}
    adjusted = replace(elf, resources={**elf.resources, "private_spill_u32": metadata})
    launch = adjusted.launch(
        "private_spill_u32",
        [0, 0],
        global_size=(128, 1, 1),
        local_size=(64, 1, 1),
        private_profile="r1-workgroup",
    )
    assert adjusted.resources["private_spill_u32"]["pds_stack"] == 3
    assert launch[-1] == 4


def test_private_source_derivation_is_pinned_and_fixture_is_real_compiler_output():
    audit = audit_sources(ROOT / "vendor/ventus-gpgpu")
    assert len(audit["files"]) == 5
    record = json.loads((FIXTURE.parent / "build.json").read_text())
    assert record["completed"] and record["elf_sha256"] == ELFProgram.read(FIXTURE).sha256
