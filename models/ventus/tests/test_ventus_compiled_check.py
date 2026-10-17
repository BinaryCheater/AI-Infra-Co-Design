"""Loader isolation, bank allocation and independent numerical-output checks."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from codesign.ventus.compiled_check import compare_outputs, prepare, recheck_rtl, validate_names
from codesign.ventus.elf import ELFProgram

ROOT = Path(__file__).resolve().parents[1]


def manifest(tmp_path):
    raw = json.loads((ROOT / "examples/compiled-functional-check.json").read_text())
    raw["cases"] = raw["cases"][:1]
    raw["cases"][0]["elf"] = str(ROOT / "tests/fixtures/compiled-vecadd/single.elf")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    return path


def test_rtl_loader_retains_real_code_and_poisoned_outputs(tmp_path):
    path = manifest(tmp_path)
    receipt = prepare(path, tmp_path / "prepared")
    row = receipt["cases"][0]
    case = tmp_path / "prepared" / row["name"]
    lines = [int(x, 16) for x in (case / "input.metadata").read_text().splitlines()]
    meta = [a | b << 32 for a, b in zip(lines[::2], lines[1::2], strict=True)]
    data = [int(x, 16) for x in (case / "input.data").read_text().splitlines()]
    segments = meta[13]
    bases, sizes = meta[14 : 14 + segments], meta[14 + segments : 14 + 2 * segments]
    initial, at = {}, 0
    for base, size in zip(bases, sizes, strict=True):
        for i in range(size // 4):
            initial[base + 4 * i] = data[at + i]
        at += size // 4
    assert at == len(data)
    elf = ELFProgram.read(ROOT / "tests/fixtures/compiled-vecadd/single.elf")
    assert [initial[elf.base + 4 * i] for i in range(len(elf.words))] == list(elf.words)
    for address, expected in row["expected_words"].items():
        assert initial[int(address)] == 0x7FC00000 != expected
    for buffer in json.loads(path.read_text())["cases"][0]["inputs"]:
        assert [initial[buffer["address"] + 4 * i] for i in range(len(buffer["words"]))] == buffer[
            "words"
        ]
    assert meta[0] == row["boot"]["base"] > elf.base + 4 * len(elf.words)
    assert row["launch_resources"]["vgpr"] == 4
    assert row["launch_resources"]["sgpr"] == 12
    with pytest.raises(FileExistsError):
        prepare(path, tmp_path / "prepared")


def test_changed_elf_is_rejected_before_execution(tmp_path):
    path = manifest(tmp_path)
    raw = json.loads(path.read_text())
    raw["cases"][0]["elf_sha256"] = "0" * 64
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="ELF identity"):
        prepare(path, tmp_path / "prepared")


def test_output_check_is_address_sensitive_and_rejects_missing_or_duplicate_words():
    a = "dump-mem: mem[0x90002000 +: 4] = 0x3f800000\n"
    b = "dump-mem: mem[0x90002004 +: 4] = 0x40000000\n"
    expected = {str(0x90002000): 0x3F800000, str(0x90002004): 0x40000000}
    assert compare_outputs(b + a, expected)["outputs_match"]
    assert not compare_outputs(a, expected)["outputs_match"]
    assert not compare_outputs(a + b + a, expected)["outputs_match"]
    assert not compare_outputs(a.replace("3f800000", "40000000") + b, expected)["outputs_match"]


@pytest.mark.parametrize("names", [[], ["../escape"], ["same", "same"], [None]])
def test_input_and_output_case_paths_are_bounded(names):
    with pytest.raises(ValueError):
        validate_names([{"name": n} for n in names])


@pytest.mark.parametrize("banks", [4, 8])
def test_compiler_usage_is_separate_from_bank_aligned_allocations(banks):
    elf = ELFProgram.read(ROOT / "tests/fixtures/compiled-vecadd/pairs.elf")
    usage = dict(elf.resources["vecadd_f32_pairs"])
    launch = elf.launch("vecadd_f32_pairs", [0, 0, 0], global_size=(128, 1, 1), rf_banks=banks)
    assert launch[6] == 8 and launch[7] % banks == 0
    assert elf.resources["vecadd_f32_pairs"] == usage
    addresses = [
        {((w + r) % banks, w * launch[6] // banks + r // banks) for r in range(usage["vgpr"])}
        for w in range(4)
    ]
    assert sum(map(len, addresses)) == len(set.union(*addresses))


def test_legacy_workitem_instructions_keep_upstream_identity_and_real_call_edges():
    runtime = ROOT / "codesign/ventus/runtime"
    source = (runtime / "rtl-r1-workitem.S").read_bytes()
    record = json.loads((runtime / "rtl-r1-workitem.json").read_text())
    original = source.split(b"\n// Added conservative assembler resource summaries")[0]
    assert hashlib.sha256(source).hexdigest() == record["assembly_sha256"]
    assert hashlib.sha256(original).hexdigest() == record["original_sha256"]
    assert b"csrr.v" not in original
    assert original.count(b"  call ") == 6
    assert b".long 0x5652534f,3,0,23,6,0,36," in source
    assert source.count(b".quad 7,32,4,0,0") == 7


def test_saved_rtl_reference_is_rechecked_instead_of_trusting_pass_flags(tmp_path):
    evidence = ROOT / "docs/evidence/compiled-functional-20261017"
    case = json.loads((evidence / "cases.json").read_text())["cases"][0]
    saved = tmp_path / "rtl"
    shutil.copytree(evidence / case["name"], saved)
    summary = json.loads((saved / "summary.json").read_text())
    summary["functional_ok"] = False
    (saved / "summary.json").write_text(json.dumps(summary))
    assert recheck_rtl(case, saved)["functional_ok"]
    address = next(iter(case["expected_words"]))
    case["expected_words"][address] ^= 1
    assert not recheck_rtl(case, saved)["functional_ok"]
    (saved / "stdout.log").write_text("truncated")
    with pytest.raises(ValueError, match="log identity"):
        recheck_rtl(case, saved)
