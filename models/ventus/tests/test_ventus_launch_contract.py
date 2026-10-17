"""Compiled launches cannot bypass per-workgroup capacity and driver bounds."""

import json
import shutil
from pathlib import Path

import pytest

from codesign.ventus.compiled_check import prepare
from codesign.ventus.config import Hardware
from codesign.ventus.elf import ELFProgram
from codesign.ventus.launch_contract import audit_rtl_source, resources, rtl_resources
from codesign.ventus.rust import PreparedProgram, RustSession

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/compiled-vecadd/single.elf"


def launch():
    elf = ELFProgram.read(FIXTURE)
    return elf, elf.launch(
        "vecadd_f32",
        [0x90000000, 0x90001000, 0x90002000],
        global_size=(128, 1, 1),
        local_size=(128, 1, 1),
    )


@pytest.mark.parametrize(
    "field,value,reason",
    [
        (4, 9, "warps allocation"),
        (6, 260, "vgpr allocation"),
        (7, 516, "sgpr allocation"),
        (8, 131200, "lds allocation"),
        (6, 7, "RF-bank rows"),
        (5, 8, "lane width"),
    ],
)
def test_invalid_compiled_admission_does_not_consume_or_discard_session(field, value, reason):
    elf, valid = launch()
    invalid = list(valid)
    invalid[field] = value
    with RustSession(Hardware()) as session:
        original_state = session.checkpoint()
        with pytest.raises(ValueError, match=reason):
            session.dispatch(PreparedProgram.from_elf(elf, invalid))
        assert session.checkpoint() == original_state
        assert session.dispatch(PreparedProgram.from_elf(elf, valid))["instructions"] > 0


def test_register_pool_is_charged_for_every_warp_not_just_once():
    _, raw = launch()
    raw[6] = 256  # Four warps exactly fill 1,024 vector-register slots.
    receipt = resources(raw, Hardware())
    assert receipt["per_workgroup"]["vgpr"] == receipt["per_sm"]["vgpr"] == 1024
    assert receipt["resident_blocks_upper_bound"] == 1
    with pytest.raises(ValueError, match="vgpr allocation"):
        resources(raw, Hardware(vgpr_slots=1020))


def test_rtl_private_interface_boundary_and_complete_grid():
    _, raw = launch()
    raw[-1] = 4096
    receipt = rtl_resources(raw)
    assert receipt["private_bytes_per_wavefront"] == 131072
    raw[-1] += 1
    with pytest.raises(ValueError, match="private bytes/thread"):
        rtl_resources(raw)
    raw[-1] = 0
    raw[10] = 1024 * 128
    raw[3] = 1024
    assert rtl_resources(raw)["workgroups_per_axis"] == [1024, 1, 1]
    raw[10] += 128
    raw[3] += 1
    with pytest.raises(ValueError, match="workgroups per axis"):
        rtl_resources(raw)
    raw[3] = 1
    with pytest.raises(ValueError, match="complete grid"):
        rtl_resources(raw)


def test_stack_footprint_includes_every_warp():
    _, raw = launch()
    raw[-3:] = [64, 32, 0]
    raw[8] = 256  # Four 64-byte stacks plus 32 static bytes need more.
    with pytest.raises(ValueError, match="stacks exceed"):
        resources(raw, Hardware())


def test_rtl_preparation_rejects_oversized_compiler_resources_before_emitting_input(
    tmp_path, monkeypatch
):
    original = ELFProgram.launch

    def oversized(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result[6] = 1028
        return result

    monkeypatch.setattr(ELFProgram, "launch", oversized)
    raw = json.loads((ROOT / "examples/compiled-functional-check.json").read_text())
    raw["cases"] = raw["cases"][:1]
    raw["cases"][0]["elf"] = str(FIXTURE)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="vgpr allocation"):
        prepare(manifest, tmp_path / "input")
    assert not list((tmp_path / "input").rglob("input.metadata"))


def test_reviewed_source_hashes_bind_the_resource_contract(tmp_path):
    source = ROOT / "vendor/ventus-gpgpu"
    receipt = audit_rtl_source(source)
    shutil.copyfile(source / "SOURCE.json", tmp_path / "SOURCE.json")
    for row in receipt["files"]:
        dest = tmp_path / row["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / row["path"], dest)
    assert audit_rtl_source(tmp_path) == receipt
    changed = tmp_path / receipt["files"][0]["path"]
    changed.write_text(changed.read_text() + "\n// different target\n")
    with pytest.raises(ValueError, match="differs from pinned manifest"):
        audit_rtl_source(tmp_path)
