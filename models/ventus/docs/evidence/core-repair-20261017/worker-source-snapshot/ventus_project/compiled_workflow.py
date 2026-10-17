"""Compile one concrete source and run its frozen cases on the configured target.

The controller compiles with explicit local tools. Native timing and RTL honor
the execution profile. Remote snapshots are fresh; resume observes/fetches them
without resubmitting a job. No search or acceptance is performed.
"""

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from uuid import uuid4

from codesign.ventus.compiled_check import prepare, recheck_rtl, run_rtl, sha, validate_names
from codesign.ventus.compiler import compile_kernel
from codesign.ventus.config import Hardware
from codesign.ventus.driver_adapter import PROFILE, PROFILES
from codesign.ventus.process import run_bounded

from .config import ROOT, execution_mode, load_profile, remote_settings


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def event(root, kind, **facts):
    with (root / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": kind, **facts}) + "\n")
    print(json.dumps({"event": kind, **facts}), flush=True)


def profile_hash(remote, observer):
    keys = ("host", "run_root", "python", "project_root", "spdlog_root", "environment_script")
    data = {k: remote.get(k) for k in keys}
    data["observer"] = str(observer) if observer else None
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def build(manifest, compiler, llvm, out, *, compiler_budget=None):
    raw = json.loads(manifest.read_text())
    allowed = {
        "schema",
        "source",
        "source_sha256",
        "kernel",
        "optimization",
        "workitem_profile",
        "cases",
        "binding",
        "driver_profile",
    }
    if raw.get("schema") != "ventus-compiled-workflow-v1" or set(raw) - allowed:
        raise ValueError("expected ventus-compiled-workflow-v1 with known fields")
    driver_profile = raw.get("driver_profile", PROFILE)
    if driver_profile not in PROFILES:
        raise ValueError("unknown compiled workflow driver profile")
    validate_names(raw["cases"])
    validate_names([{"name": raw["kernel"]}])
    source = (manifest.parent / raw["source"]).resolve()
    if sha(source) != raw["source_sha256"]:
        raise ValueError("source differs from the concrete workflow manifest")
    supported = {"name", "arguments", "global_size", "local_size", "inputs", "outputs"}
    if any(set(c) != supported for c in raw["cases"]):
        raise ValueError(
            "workflow case has missing or unsupported execution fields; kernel/ELF are inherited"
        )
    binding = None
    if "binding" in raw:
        from codesign.ventus.gemm_lowering import verify_binding

        binding = verify_binding(raw, source.read_text())
    out.mkdir(parents=True, exist_ok=False)
    work = out / "work"
    work.mkdir()
    (work / "source").mkdir()
    shutil.copyfile(source, work / "source/kernel.cl")
    frozen = {**raw, "source": "source/kernel.cl", "driver_profile": driver_profile}
    write(work / "spec.json", frozen)
    if binding is not None:
        write(work / "binding.json", binding)
    event(out, "compilation_started", source_sha256=raw["source_sha256"])
    compile_kernel(
        work / "source/kernel.cl",
        raw["kernel"],
        compiler,
        llvm,
        work / "compile",
        optimization=raw.get("optimization", "O1"),
        workitem_profile=raw.get("workitem_profile", "rtl-r1"),
        host_budget=compiler_budget,
    )
    elf = work / "compile/program.elf"
    check = {
        "schema": "ventus-compiled-check-v1",
        "cases": [
            {**c, "elf": "compile/program.elf", "elf_sha256": sha(elf), "kernel": raw["kernel"]}
            for c in raw["cases"]
        ],
    }
    write(work / "cases.json", check)
    prepare(work / "cases.json", work / "input", driver_profile=driver_profile)
    event(out, "inputs_frozen", elf_sha256=sha(elf), cases=len(check["cases"]))
    return work


def code_snapshot(destination):
    files = list((ROOT / "codesign").glob("*.py"))
    files.extend(
        p
        for p in (ROOT / "codesign/ventus").rglob("*")
        if p.is_file()
        and "target" not in p.parts
        and "__pycache__" not in p.parts
        and p.suffix in {".py", ".rs", ".toml", ".lock", ".S", ".json", ".cpp"}
    )
    files.extend(
        ROOT / ("ventus_project/" + name)
        for name in ("__init__.py", "config.py", "compiled_workflow.py")
    )
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        dest = destination / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        hashes[relative.as_posix()] = sha(dest)
    write(destination / "code-hashes.json", hashes)


def worker(root, job_id, rtl_root, spdlog, observer, budget, timeout, phase="all"):
    if phase not in {"native", "all"}:
        raise ValueError("unknown compiled execution phase")
    hashes = json.loads((root / "code-hashes.json").read_text())
    for name, digest in hashes.items():
        if sha(root / name) != digest:
            raise ValueError("frozen worker source differs")
    work = root / "work"
    execution = work / "execution"
    execution.mkdir()
    write(root / "worker.json", {"pid": os.getpid(), "job_id": job_id})
    raw = json.loads((work / "cases.json").read_text())
    driver_profile = json.loads((work / "spec.json").read_text()).get(
        "driver_profile", "upstream-r1"
    )
    validate_names(raw["cases"])
    began = time.monotonic()
    native = []
    try:
        (execution / "native-input").mkdir()
        for case in raw["cases"]:
            initial = {
                str(b["address"] + 4 * i): v
                for b in case["inputs"]
                for i, v in enumerate(b["words"])
            }
            for b in case["outputs"]:
                initial.update(
                    {str(b["address"] + 4 * i): 0x7FC00000 for i in range(len(b["expected_words"]))}
                )
            program = {
                "schema": "ventus-dispatches-v1",
                "hardware": Hardware().to_dict(),
                "dispatches": [
                    {
                        **{
                            k: case[k]
                            for k in (
                                "name",
                                "kernel",
                                "elf_sha256",
                                "arguments",
                                "global_size",
                                "local_size",
                            )
                        },
                        "elf": "../../compile/program.elf",
                        "functional_words": initial,
                        "private_profile": "r1-workgroup"
                        if driver_profile == PROFILE
                        else "r1-upstream",
                    }
                ],
            }
            input_path = execution / "native-input" / (case["name"] + ".json")
            write(input_path, program)
            dest = execution / ("native-" + case["name"])
            remaining = budget - (time.monotonic() - began)
            if remaining <= 0:
                native.append(
                    {"name": case["name"], "completed": False, "reason": "workflow_budget"}
                )
                continue
            process = run_bounded(
                [
                    sys.executable,
                    "-m",
                    "codesign.ventus",
                    "instructions",
                    "--input",
                    str(input_path),
                    "--out",
                    str(dest),
                    "--budget",
                    str(min(60, remaining)),
                ],
                cwd=root,
                stdout=execution / (case["name"] + ".stdout"),
                stderr=execution / (case["name"] + ".stderr"),
                timeout=min(timeout, remaining),
            )
            result_path = dest / "experiment.json"
            result = json.loads(result_path.read_text()) if result_path.is_file() else {}
            ok = (
                process["returncode"] == 0
                and not process["timed_out"]
                and not process["output_exhausted"]
                and result.get("completed") is True
            )
            native.append(
                {
                    "name": case["name"],
                    "completed": ok,
                    "process": process,
                    "cycles": result.get("cycles"),
                    "instructions": result.get("instructions"),
                    "receipt_sha256": sha(result_path) if result_path.exists() else None,
                }
            )
            event(root, "native_finished", name=case["name"], completed=ok)
        reference = (
            run_rtl(
                work / "input/cases.json",
                execution / "rtl",
                rtl_root,
                spdlog,
                observer,
                timeout=timeout,
                host_budget=budget - (time.monotonic() - began),
            )
            if phase == "all"
            else {"status": "not_run", "reason": "Native-only planning; no RTL numerical check"}
        )
        result = {
            "schema": "ventus-compiled-workflow-result-v1",
            "job_id": job_id,
            "phase": phase,
            "native": native,
            "rtl": reference,
            "completed": all(c["completed"] for c in native)
            and (phase == "native" or reference["all_functional_ok"]),
            "code_manifest_sha256": sha(root / "code-hashes.json"),
            "case_manifest_sha256": sha(work / "cases.json"),
            "python": sys.version.split()[0],
            "architecture": platform.machine(),
            "scope": "Completion covers the requested phase only; native-only is not numerical RTL validation, accuracy, optimality or acceptance",
        }
    except BaseException as error:
        write(root / "worker-failure.json", {"job_id": job_id, "type": type(error).__name__})
        raise
    write(root / "worker-result.json", result)
    event(root, "worker_finished", completed=result["completed"])
    return result


def ssh(remote, command, **kwargs):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", remote["host"], command], **kwargs
    )


def fetch(out, remote):
    submission = json.loads((out / "submission.json").read_text())
    root = remote["run_root"] + "/" + submission["remote_relative"]
    if (out / "reference").exists():
        return verify_reference(out, out / "reference", submission)
    # Observation never launches a replacement. A stale PID is not completion.
    script = """import json,os,subprocess
from pathlib import Path
r=Path('.')
if (r/'worker-result.json').exists(): print('terminal')
elif (r/'worker-failure.json').exists(): print('failed')
elif (r/'worker.json').exists():
 p=json.loads((r/'worker.json').read_text());s=subprocess.run(['ps','-p',str(p['pid']),'-o','args='],capture_output=True,text=True)
 print('running' if p['job_id'] in s.stdout else 'interrupted')
else: print('unconfirmed')
"""
    command = (
        "cd "
        + shlex.quote(root)
        + " && "
        + shlex.quote(remote["run_root"] + "/.venv/bin/python")
        + " -c "
        + shlex.quote(script)
    )
    state = ssh(remote, command, capture_output=True, text=True, timeout=30)
    observed = state.stdout.strip()
    if state.returncode or observed not in {"terminal", "failed"}:
        event(
            out,
            "reference_observed",
            state=state.stdout.strip() if state.returncode == 0 else "observation_failed",
        )
        return None
    archive = out / ("reference-download-" + uuid4().hex[:8] + ".tar")
    status_file = "worker-result.json" if observed == "terminal" else "worker-failure.json"
    with archive.open("xb") as stream:
        transferred = ssh(
            remote,
            "tar -cf - -C "
            + shlex.quote(root)
            + " "
            + status_file
            + " worker.json code-hashes.json work/execution",
            stdout=stream,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    if transferred.returncode:
        event(out, "retrieval_failed")
        return None
    dest = out / ("reference-staging-" + uuid4().hex[:8])
    dest.mkdir()
    with tarfile.open(archive) as bundle:
        bundle.extractall(dest, filter="data")
    result = verify_reference(out, dest, submission)
    dest.rename(out / "reference")
    event(out, "reference_fetched", completed=result["completed"])
    return result


def verify_reference(out, dest, submission):
    if not (dest / "worker-result.json").exists() and (dest / "worker-failure.json").exists():
        failure = json.loads((dest / "worker-failure.json").read_text())
        if failure["job_id"] != submission["job_id"]:
            raise ValueError("worker failure belongs to another job")
        return {"completed": False, "status": "worker_failed", "failure": failure}
    result = json.loads((dest / "worker-result.json").read_text())
    if (
        result["job_id"] != submission["job_id"]
        or result["code_manifest_sha256"] != submission["code_manifest_sha256"]
        or result["case_manifest_sha256"] != sha(out / "work/cases.json")
        or result.get("phase", "all") != submission.get("phase", "all")
        or result.get("phase", "all") not in {"native", "all"}
    ):
        raise ValueError("downloaded reference identity differs")
    if sha(dest / "code-hashes.json") != submission["code_manifest_sha256"]:
        raise ValueError("downloaded code provenance differs")
    if result.get("phase", "all") == "native" and result["rtl"].get("status") != "not_run":
        raise ValueError("native-only execution must keep RTL explicitly not run")
    expected = json.loads((out / "work/input/cases.json").read_text())["cases"]
    if [c["name"] for c in result["native"]] != [c["name"] for c in expected]:
        raise ValueError("native case coverage differs")
    checked = []
    for case, native in zip(expected, result["native"], strict=True):
        receipt = dest / "work/execution" / ("native-" + case["name"]) / "experiment.json"
        if native.get("completed") and not native.get("receipt_sha256"):
            raise ValueError("native completion has no execution receipt")
        if native.get("receipt_sha256"):
            if sha(receipt) != native["receipt_sha256"]:
                raise ValueError("native execution receipt differs")
            if result.get("phase", "all") == "native" and native["completed"]:
                recorded = json.loads(receipt.read_text())
                if (
                    recorded.get("completed") is not True
                    or recorded["cycles"] != native["cycles"]
                    or recorded["hardware"] != Hardware().to_dict()
                    or len(recorded["stages"]) != 1
                    or recorded["stages"][0]["program"]["elf_sha256"] != case["elf_sha256"]
                ):
                    raise ValueError("native planning receipt differs from the frozen case")
        if result.get("phase", "all") == "all":
            checked.append(
                recheck_rtl(case, dest / "work/execution/rtl" / case["name"])["functional_ok"]
            )
    if result["completed"] != (all(c["completed"] for c in result["native"]) and all(checked)):
        raise ValueError("workflow completion differs from raw references")
    return result


def run(
    manifest,
    compiler,
    llvm,
    out,
    *,
    mode=None,
    rtl_root=None,
    spdlog=None,
    observer=None,
    budget=900,
    timeout=180,
    phase="all",
):
    build(manifest, compiler, llvm, out)
    return execute_prepared(
        out,
        mode=mode,
        rtl_root=rtl_root,
        spdlog=spdlog,
        observer=observer,
        budget=budget,
        timeout=timeout,
        phase=phase,
    )


def validate_prepared(out):
    """Check a transported compiled work tree before any target connection."""
    work = out / "work"
    spec = json.loads((work / "spec.json").read_text())
    source = work / "source/kernel.cl"
    receipt = json.loads((work / "compile/build.json").read_text())
    if (
        spec["source"] != "source/kernel.cl"
        or sha(source) != spec["source_sha256"]
        or not receipt["completed"]
        or receipt["source_sha256"]["kernel.cl"] != sha(source)
        or receipt["elf_sha256"] != sha(work / "compile/program.elf")
    ):
        raise ValueError("prepared compile source/ELF identity differs")
    if "binding" in spec:
        from codesign.ventus.gemm_lowering import verify_binding

        if verify_binding(spec, source.read_text()) != json.loads(
            (work / "binding.json").read_text()
        ):
            raise ValueError("prepared realization binding differs")
    expected = [
        {
            **case,
            "elf": "compile/program.elf",
            "elf_sha256": receipt["elf_sha256"],
            "kernel": spec["kernel"],
        }
        for case in spec["cases"]
    ]
    cases = json.loads((work / "cases.json").read_text())
    if cases != {"schema": "ventus-compiled-check-v1", "cases": expected}:
        raise ValueError("prepared cases differ from source/compile selection")
    profile = spec.get("driver_profile", PROFILE)
    regenerated = prepare(work / "cases.json", out / "admission-recheck", driver_profile=profile)
    if regenerated != json.loads((work / "input/cases.json").read_text()):
        raise ValueError("prepared launch/input receipt differs")
    for case in regenerated["cases"]:
        for name, digest in case["input_sha256"].items():
            if sha(work / "input" / case["name"] / name) != digest:
                raise ValueError("prepared launch/data bytes differ")
    return spec


def execute_prepared(
    out,
    *,
    mode=None,
    rtl_root=None,
    spdlog=None,
    observer=None,
    budget=900,
    timeout=180,
    phase="all",
):
    """Execute a fresh frozen work tree without repeating compilation or submission."""
    if (out / "submission.json").exists() or (out / "snapshot").exists():
        raise ValueError("prepared execution already submitted; observe the existing job")
    if phase not in {"native", "all"} or (phase == "native" and observer is not None):
        raise ValueError("unknown execution phase or observer requested without RTL")
    if (
        type(budget) is not int
        or not 1 <= budget <= 86400
        or type(timeout) is not int
        or not 1 <= timeout <= 600
    ):
        raise ValueError("invalid prepared execution time budget")
    validate_prepared(out)
    profile = load_profile()
    mode = execution_mode(profile, mode)
    remote = remote_settings(profile) if mode == "remote" else None
    if mode == "remote" and phase == "all":
        if any(
            not remote.get(k) or not str(remote[k]).startswith("/")
            for k in ("project_root", "spdlog_root")
        ):
            raise ValueError("configured RTL and spdlog roots must be absolute paths")
        rtl_root, spdlog = Path(remote["project_root"]), Path(remote["spdlog_root"])
    if phase == "all" and (rtl_root is None or spdlog is None):
        raise ValueError("local RTL execution needs explicit --rtl-root and --spdlog")
    snapshot = out / "snapshot"
    snapshot.mkdir()
    code_snapshot(snapshot)
    job_id = uuid4().hex
    identity = {
        "job_id": job_id,
        "mode": mode,
        "phase": phase,
        "code_manifest_sha256": sha(snapshot / "code-hashes.json"),
    }
    if mode == "local":
        shutil.copytree(out / "work", snapshot / "work")
        write(out / "submission.json", identity)
        command = [
            sys.executable,
            "-m",
            "ventus_project.compiled_workflow",
            "worker",
            "--root",
            str(snapshot),
            "--job-id",
            job_id,
            "--phase",
            phase,
            "--budget",
            str(budget),
            "--timeout",
            str(timeout),
        ]
        if phase == "all":
            command.extend(
                ["--rtl-root", str(rtl_root.resolve()), "--spdlog", str(spdlog.resolve())]
            )
        if observer:
            command.extend(["--observer", str(observer.resolve())])
        run_bounded(
            command,
            cwd=snapshot,
            stdout=out / "transport.stdout",
            stderr=out / "transport.stderr",
            timeout=budget + 60,
        )
        if (
            not (snapshot / "worker-result.json").exists()
            and not (snapshot / "worker-failure.json").exists()
        ):
            event(out, "local_worker_interrupted")
            return None
        return verify_reference(out, snapshot, identity)
    relative = "results/compiled-workflow-" + job_id
    identity.update(
        remote_relative=relative,
        profile_sha256=profile_hash(remote, observer),
        observer=str(observer) if observer else None,
    )
    # Absolute configured observer paths stay in the private transport receipt,
    # never in shared source or exported scientific result identities.
    write(out / "submission.json", identity)
    archive = out / "submission.tar"
    with tarfile.open(archive, "x") as bundle:
        for p in sorted(snapshot.rglob("*")):
            if p.is_file():
                bundle.add(p, arcname=p.relative_to(snapshot))
        bundle.add(out / "work", arcname="work")
    root = remote["run_root"] + "/" + relative
    with archive.open("rb") as stream:
        copied = ssh(
            remote,
            "mkdir " + shlex.quote(root) + " && tar -xf - -C " + shlex.quote(root),
            stdin=stream,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    if copied.returncode:
        event(out, "submission_failed")
        return None
    event(out, "reference_submitted", job_id=job_id)
    command = [
        remote["run_root"] + "/.venv/bin/python",
        "-m",
        "ventus_project.compiled_workflow",
        "worker",
        "--root",
        root,
        "--job-id",
        job_id,
        "--phase",
        phase,
        "--budget",
        str(budget),
        "--timeout",
        str(timeout),
    ]
    if phase == "all":
        command.extend(["--rtl-root", str(rtl_root), "--spdlog", str(spdlog)])
    if observer:
        command.extend(["--observer", str(observer)])
    setup = (
        ". " + shlex.quote(remote["environment_script"]) + " && "
        if remote.get("environment_script")
        else ""
    )
    shell = "cd " + shlex.quote(root) + " && " + setup + "exec " + shlex.join(command)
    run_bounded(
        ["ssh", "-o", "BatchMode=yes", remote["host"], shell],
        cwd=out,
        stdout=out / "transport.stdout",
        stderr=out / "transport.stderr",
        timeout=budget + 60,
    )
    return fetch(out, remote)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("--manifest", type=Path, required=True)
    start.add_argument("--compiler-root", type=Path, required=True)
    start.add_argument("--llvm-source", type=Path, required=True)
    start.add_argument("--out", type=Path, required=True)
    start.add_argument("--execution", choices=["local", "remote"])
    resume = sub.add_parser("resume")
    resume.add_argument("--out", type=Path, required=True)
    child = sub.add_parser("worker")
    child.add_argument("--root", type=Path, required=True)
    child.add_argument("--job-id", required=True)
    for p in (start, child):
        p.add_argument("--phase", choices=("native", "all"), default="all")
        p.add_argument("--rtl-root", type=Path)
        p.add_argument("--spdlog", type=Path)
        p.add_argument("--observer", type=Path)
        p.add_argument("--budget", type=int, default=900)
        p.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.command == "start":
        if not 1 <= args.budget <= 86400 or not 1 <= args.timeout <= 600:
            parser.error("workflow budget must be 1..86400s; per-case timeout 1..600s")
        result = run(
            args.manifest.resolve(),
            args.compiler_root.resolve(),
            args.llvm_source.resolve(),
            args.out.resolve(),
            mode=args.execution,
            rtl_root=args.rtl_root,
            spdlog=args.spdlog,
            observer=args.observer,
            budget=args.budget,
            timeout=args.timeout,
            phase=args.phase,
        )
    elif args.command == "worker":
        result = worker(
            args.root.resolve(),
            args.job_id,
            args.rtl_root,
            args.spdlog,
            args.observer,
            args.budget,
            args.timeout,
            args.phase,
        )
    else:
        submission = json.loads((args.out / "submission.json").read_text())
        if submission["mode"] != "remote":
            parser.error("resume observes remote jobs only")
        remote = remote_settings(load_profile())
        if profile_hash(remote, submission["observer"]) != submission["profile_sha256"]:
            parser.error("configured remote identity changed; no connection attempted")
        result = fetch(args.out.resolve(), remote)
    print(
        json.dumps(
            {
                "completed": result.get("completed") if result else False,
                "scope": "integration only; no search/acceptance",
            }
        )
    )
    if result is None or not result["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
