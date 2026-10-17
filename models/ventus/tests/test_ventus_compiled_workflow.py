"""Concrete lowering integrity, self-contained workers and bounded tool failures."""

import json
import subprocess
import sys
from copy import deepcopy

import pytest

from codesign.ventus.config import Hardware
from codesign.ventus.gemm_lowering import bits, materialize, verify_binding
from codesign.ventus.process import run_bounded
from ventus_project.compiled_workflow import (
    build,
    code_snapshot,
    execute_prepared,
    fetch,
    sha,
    verify_reference,
    write,
)


def concrete(tmp_path):
    # A=[1,2], B=[[1,2,3],[4,5,6]], C=[9,12,15]. Reference is independent
    # elementary arithmetic, not output from the lowering or instruction model.
    case = {
        "name": "rectangular",
        "arguments": [0x90000000, 0x90001000, 0x90002000, 1, 3, 2],
        "inputs": [
            {"address": 0x90000000, "words": list(map(bits, [1, 2]))},
            {"address": 0x90001000, "words": list(map(bits, [1, 2, 3, 4, 5, 6]))},
        ],
        "outputs": [{"address": 0x90002000, "expected_words": list(map(bits, [9, 12, 15]))}],
    }
    spec = materialize({"tile_m": 4, "tile_n": 8, "tile_k": 1}, [case], tmp_path / "selected")
    return spec, (tmp_path / "selected/kernel.cl").read_text()


@pytest.mark.parametrize(
    "mutation", ["source", "assignment", "launch", "reference", "hash", "entry"]
)
def test_selected_assignment_cannot_drift_before_compilation(tmp_path, mutation):
    spec, source = concrete(tmp_path)
    assert verify_binding(spec, source)["assignment"]["tile_n"] == 8
    spec = deepcopy(spec)
    if mutation == "source":
        source = source.replace("row<m", "row<=m")
    elif mutation == "assignment":
        spec["binding"]["assignment"]["tile_n"] = 16
    elif mutation == "launch":
        spec["cases"][0]["global_size"] = [64, 1, 1]
    elif mutation == "reference":
        spec["cases"][0]["outputs"][0]["expected_words"][0] = bits(10)
    elif mutation == "hash":
        spec["source_sha256"] = "0" * 64
    else:
        spec["kernel"] = "unrelated"
    with pytest.raises(ValueError):
        verify_binding(spec, source)


def test_worker_snapshot_contains_public_dependencies_and_imports_in_isolation(tmp_path):
    code_snapshot(tmp_path)
    assert (tmp_path / "codesign/solver.py").is_file()
    result = subprocess.run(
        [sys.executable, "-m", "codesign.ventus", "--help"],
        cwd=tmp_path,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert not (tmp_path / "local.toml").exists()


@pytest.mark.parametrize("kind", ["timeout", "output"])
def test_external_tools_do_not_turn_budget_exhaustion_into_success(tmp_path, kind):
    code = "import time; time.sleep(10)" if kind == "timeout" else "print('x'*20000)"
    result = run_bounded(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        stdout=tmp_path / "stdout",
        stderr=tmp_path / "stderr",
        timeout=0.2 if kind == "timeout" else 10,
        output_bytes=1024,
    )
    assert result["timed_out"] if kind == "timeout" else result["output_exhausted"]


def test_cached_terminal_failure_is_observed_without_resubmission(tmp_path, monkeypatch):
    out = tmp_path
    dest = out / "reference"
    dest.mkdir()
    write(out / "submission.json", {"job_id": "existing", "remote_relative": "results/existing"})
    write(dest / "worker-failure.json", {"job_id": "existing", "type": "MissingDependency"})

    def forbidden(*_args, **_kwargs):
        raise AssertionError("resume must not submit or connect for cached terminal result")

    monkeypatch.setattr("ventus_project.compiled_workflow.ssh", forbidden)
    result = fetch(out, {"run_root": "/work/project"})
    assert result["completed"] is False


def test_changed_source_is_rejected_without_creating_a_build(tmp_path):
    spec, _ = concrete(tmp_path)
    (tmp_path / "selected/kernel.cl").write_text("changed")
    with pytest.raises(ValueError, match="source differs"):
        build(
            tmp_path / "selected/workflow.json",
            tmp_path / "no-compiler",
            tmp_path / "no-llvm",
            tmp_path / "job",
        )
    assert not (tmp_path / "job").exists()


def test_prepared_execution_never_resubmits_an_existing_job(tmp_path, monkeypatch):
    write(tmp_path / "submission.json", {"job_id": "already-submitted"})

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Existing job must be observed, not recompiled or resubmitted")

    monkeypatch.setattr("ventus_project.compiled_workflow.compile_kernel", forbidden)
    monkeypatch.setattr("ventus_project.compiled_workflow.ssh", forbidden)
    with pytest.raises(ValueError, match="already submitted"):
        execute_prepared(tmp_path)


@pytest.mark.parametrize("mutation", [None, "phase", "claimed_rtl", "elf", "completion"])
def test_native_only_receipt_cannot_become_full_rtl_validation(tmp_path, mutation):
    # Minimal receipt fixture exercises phase authority; it is not RTL evidence.
    out, dest = tmp_path / "job", tmp_path / "reference"
    (out / "work/input").mkdir(parents=True)
    (dest / "work/execution/native-point").mkdir(parents=True)
    write(out / "work/cases.json", {"cases": [{"name": "point"}]})
    write(out / "work/input/cases.json", {"cases": [{"name": "point", "elf_sha256": "elf"}]})
    write(dest / "code-hashes.json", {"source": "fixture"})
    submission = {
        "job_id": "fixture",
        "phase": "native",
        "code_manifest_sha256": sha(dest / "code-hashes.json"),
    }
    receipt = dest / "work/execution/native-point/experiment.json"
    recorded = {
        "completed": True,
        "cycles": 13,
        "hardware": Hardware().to_dict(),
        "stages": [{"program": {"elf_sha256": "elf"}}],
    }
    write(receipt, recorded)
    result = {
        **submission,
        "completed": True,
        "case_manifest_sha256": sha(out / "work/cases.json"),
        "native": [
            {"name": "point", "completed": True, "cycles": 13, "receipt_sha256": sha(receipt)}
        ],
        "rtl": {"status": "not_run"},
    }
    if mutation == "phase":
        result["phase"] = "all"
    elif mutation == "claimed_rtl":
        result["rtl"]["status"] = "passed"
    elif mutation == "elf":
        recorded["stages"][0]["program"]["elf_sha256"] = "different"
        receipt.write_text(json.dumps(recorded))
        result["native"][0]["receipt_sha256"] = sha(receipt)
    elif mutation == "completion":
        result["completed"] = False
    write(dest / "worker-result.json", result)
    if mutation is None:
        assert verify_reference(out, dest, submission)["rtl"] == {"status": "not_run"}
    else:
        with pytest.raises(ValueError):
            verify_reference(out, dest, submission)


def test_exhausted_compile_budget_does_not_launch_a_command(tmp_path, monkeypatch):
    from codesign.ventus.compiler import compile_kernel

    # Only dependency existence/hash reads are needed: exhaustion must prevent exec.
    for name in ("bin/clang", "bin/llvm-objdump", "bin/ld.lld"):
        p = tmp_path / "compiler" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fixture; must never execute")
    for name in (
        "libclc/riscv32/lib/workitem/workitem.S",
        "libclc/riscv32/lib/workitem/helper.cl",
        "libclc/riscv32/lib/ventus.h",
        "utils/ldscripts/ventus/elf32lriscv.ld",
        "libclc/generic/include/header.h",
    ):
        p = tmp_path / "llvm" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fixture")
    source = tmp_path / "kernel.cl"
    source.write_text("fixture")
    times = iter([0, 2, 2])
    monkeypatch.setattr("codesign.ventus.compiler.perf_counter", lambda: next(times))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("An exhausted host budget cannot launch another compiler command")

    monkeypatch.setattr("codesign.ventus.compiler.run_bounded", forbidden)
    with pytest.raises(ValueError, match="compilation failed"):
        compile_kernel(
            source,
            "kernel",
            tmp_path / "compiler",
            tmp_path / "llvm",
            tmp_path / "build",
            host_budget=1,
        )
    receipt = json.loads((tmp_path / "build/build.json").read_text())
    assert receipt["completed"] is False
    assert receipt["processes"][0]["timed_out"] is True
    assert receipt["host_budget_seconds"] == 1
