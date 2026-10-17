"""Load actual compiled ELF and declared inputs into the pinned RTL driver ABI.

Numerical outputs and named-kernel completion are independent of modeled timing.
The explicit boot stub supplies the register context assumed by the ELF provider;
its bytes/addresses are recorded and are not included in the compiled ELF identity.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from time import perf_counter

from .elf import ELFProgram
from .process import run_bounded
from .rtl_diagnostics import classify_rtl_logs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def word(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("expected an unsigned RV32 word")
    return value


def validate_names(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("expected nonempty case list")
    names = [c["name"] for c in cases]
    if any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
        raise ValueError("case names must be unique safe identifiers")
    if len(set(names)) != len(names):
        raise ValueError("case names must be unique safe identifiers")


def bootstrap(elf, launch):
    """Initialize scalar ABI registers; call the ELF entry; ENDPRG on return."""
    base = (max(elf.base + 4 * len(elf.words), max(elf.memory, default=0) + 4) + 127) // 128 * 128
    code = [0x37]  # LUI x0,0; physical x0 is explicitly initialized too.

    def li(rd, value):
        value = word(value)
        code.extend(
            [
                (((value + 0x800) >> 12) << 12 & 0xFFFFF000) | rd << 7 | 0x37,
                (value & 0xFFF) << 20 | rd << 15 | rd << 7 | 0x13,
            ]
        )

    def csr(rd, index):
        code.append(index << 20 | 2 << 12 | rd << 7 | 0x73)

    def add(rd, a, b):
        code.append(b << 20 | a << 15 | rd << 7 | 0x33)

    n = launch[19]
    stride, static = launch[20 + 2 * n : 22 + 2 * n]
    csr(2, 0x806)  # Work-group LDS base.
    li(5, static)
    add(8, 2, 5)
    csr(6, 0x805)  # Warp ID within this work-group.
    li(7, stride)
    code.append(1 << 25 | 7 << 20 | 6 << 15 | 6 << 7 | 0x33)  # MUL x6,x6,x7.
    add(2, 8, 6)
    return_li = len(code)
    li(1, 0)  # Patched below after the complete stub length is known.
    li(3, launch[17])
    li(4, 0)
    li(10, launch[9])
    li(5, launch[1])
    code.append(5 << 15 | 0x67)  # JALR x0,x5,0.
    return_pc = base + 4 * len(code)
    code.append(0x400B)
    value = return_pc
    code[return_li : return_li + 2] = [
        (((value + 0x800) >> 12) << 12 & 0xFFFFF000) | 1 << 7 | 0x37,
        (value & 0xFFF) << 20 | 1 << 15 | 1 << 7 | 0x13,
    ]
    return base, code


def prepare(manifest, out):
    raw = json.loads(manifest.read_text())
    if raw.get("schema") != "ventus-compiled-check-v1" or not raw.get("cases"):
        raise ValueError("expected nonempty ventus-compiled-check-v1")
    if out.exists():
        raise FileExistsError(out)
    validate_names(raw["cases"])
    out.mkdir(parents=True)
    cases = []
    for case in raw["cases"]:
        elf_path = (manifest.parent / case["elf"]).resolve()
        elf = ELFProgram.read(elf_path)
        if elf.sha256 != case["elf_sha256"]:
            raise ValueError("compiled ELF identity differs")
        inputs = {}
        for buffer in case.get("inputs", []):
            address = word(buffer["address"])
            if address % 4 or not buffer["words"]:
                raise ValueError("input must be a nonempty aligned word buffer")
            for i, value in enumerate(buffer["words"]):
                at = word(address + 4 * i)
                if at in inputs:
                    raise ValueError("input buffers overlap")
                inputs[at] = word(value)
        expected = {}
        for output in case["outputs"]:
            address = word(output["address"])
            if address % 4 or not output["expected_words"]:
                raise ValueError("output needs aligned reference words")
            for i, value in enumerate(output["expected_words"]):
                at = word(address + 4 * i)
                if at in inputs or at in expected:
                    raise ValueError(
                        "overlapping input/output buffers are unsupported by this checker"
                    )
                expected[at] = word(value)
        if not expected or len(inputs) + len(expected) > 262144:
            raise ValueError("checker needs outputs and at most 1 MiB of payload words")
        # Non-finite poison avoids passing unwritten finite reference outputs.
        if any(v == 0x7FC00000 for v in expected.values()):
            raise ValueError("reference collides with output poison word")
        initial = {**inputs, **{a: 0x7FC00000 for a in expected}}
        launch = elf.launch(
            case["kernel"],
            case["arguments"],
            global_size=case["global_size"],
            local_size=case["local_size"],
            functional_words=initial,
        )
        boot_pc, boot = bootstrap(elf, launch)
        memory = {elf.base + 4 * i: v for i, v in enumerate(elf.words)}
        for i in range(launch[19]):
            at, value = launch[20 + 2 * i : 22 + 2 * i]
            if at in memory:
                raise ValueError("ELF code and initial data overlap")
            memory[at] = value
        for i, value in enumerate(boot):
            at = boot_pc + 4 * i
            if at in memory:
                raise ValueError("boot stub overlaps ELF/data")
            memory[at] = value
        # The driver consumes caller-supplied backing segments. Compiled vector
        # spills use interleaved private storage; a partial cache-line store may
        # read its backing line even when the source never reads uninitialized
        # private data. Reserve the complete declared grid footprint explicitly.
        private_bytes = launch[3] * launch[4] * launch[5] * launch[-1]
        private_bytes = (private_bytes + 127) // 128 * 128
        if private_bytes > 1024 * 1024:
            raise ValueError("compiled checker private backing exceeds 1 MiB")
        for at in range(0xA0000000, 0xA0000000 + private_bytes, 4):
            if at in memory:
                raise ValueError("private backing overlaps program or user buffers")
            memory[at] = 0
        segments = []
        for at, value in sorted(memory.items()):
            if not segments or at != segments[-1][0] + 4 * len(segments[-1][1]):
                segments.append((at, []))
            segments[-1][1].append(value)
        global_size, local_size = case["global_size"], case["local_size"]
        header = [
            boot_pc,
            0,
            *[
                (g + extent - 1) // extent
                for g, extent in zip(global_size, local_size, strict=True)
            ],
            32,
            launch[4],
            launch[18],
            launch[8],
            launch[-1],
            launch[7],
            launch[6],
            0xA0000000,
            len(segments),
        ]
        sizes = [4 * len(data) for _, data in segments]
        metadata = [*header, *[at for at, _ in segments], *sizes, *sizes]
        dest = out / case["name"]
        dest.mkdir()
        (dest / "input.metadata").write_text(
            "".join(f"{v & 0xFFFFFFFF:08x}\n{v >> 32:08x}\n" for v in metadata)
        )
        (dest / "input.data").write_text(
            "".join(f"{v:08x}\n" for _, data in segments for v in data)
        )
        record = {
            "name": case["name"],
            "kernel": case["kernel"],
            "elf_sha256": elf.sha256,
            "expected_words": {str(a): v for a, v in expected.items()},
            "input_sha256": {name: sha(dest / name) for name in ("input.metadata", "input.data")},
            "launch_resources": {
                "warps": launch[4],
                "vgpr": launch[6],
                "sgpr": launch[7],
                "lds": launch[8],
                "rf_banks": 4,
                "compiler_usage": elf.resources[case["kernel"]],
            },
            "boot": {
                "base": boot_pc,
                "words": boot,
                "role": "explicit ABI context; excluded from compiled-kernel timing",
            },
            "private_backing": {
                "base": 0xA0000000,
                "bytes": private_bytes,
                "initialization": "zeroed RTL backing; does not certify initialized source-level private reads",
            },
        }
        cases.append(record)
    receipt = {
        "schema": "ventus-compiled-rtl-input-v1",
        "manifest_sha256": sha(manifest),
        "cases": cases,
        "scope": "Actual ELF words plus explicit boot context; numerical exact-word check, no timing calibration.",
    }
    (out / "cases.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def compare_outputs(stdout, expected):
    found, duplicates = {}, []
    for at, value in re.findall(
        r"dump-mem: mem\[0x([0-9a-f]+) \+: 4\] = 0x([0-9a-f]+)", stdout, re.I
    ):
        at, value = int(at, 16), int(value, 16)
        if at in found:
            duplicates.append(at)
        found[at] = value
    expected = {int(a): word(v) for a, v in expected.items()}
    mismatches = [
        {"address": a, "expected": v, "actual": found.get(a)}
        for a, v in expected.items()
        if found.get(a) != v
    ]
    return {
        "outputs_match": not mismatches and not duplicates,
        "expected_words": len(expected),
        "observed_words": len(found),
        "duplicate_addresses": duplicates,
        "mismatches": mismatches,
    }


def recheck_rtl(case, directory):
    """Recompute finite correctness from hash-bound logs; never trust stored pass flags."""
    summary = json.loads((directory / "summary.json").read_text())
    if (
        summary["name"] != case["name"]
        or summary["elf_sha256"] != case["elf_sha256"]
        or summary["case_input_sha256"] != case["input_sha256"]
    ):
        raise ValueError("RTL reference does not bind this ELF and initialized launch")
    contents = {}
    for name in ("stdout.log", "stderr.log"):
        path = directory / name
        if sha(path) != summary["logs_sha256"][name]:
            raise ValueError("RTL log identity differs")
        contents[name] = path.read_text(errors="replace")
    completion = classify_rtl_logs(
        contents["stdout.log"],
        contents["stderr.log"],
        kernel_name=case["name"],
        returncode=summary["returncode"],
        timed_out=summary["timed_out"],
    )
    numeric = compare_outputs(contents["stdout.log"], case["expected_words"])
    return {
        "functional_ok": completion["kernel_completed"] and numeric["outputs_match"],
        "completion": completion,
        "numerical": numeric,
        "reference_sha256": sha(directory / "summary.json"),
        "rtl_binary_sha256": summary["rtl_binary_sha256"],
        "rtl_library_sha256": summary["rtl_library_sha256"],
        "scope": "Finite outputs and named completion for this frozen program/input/launch on the recorded RTL.",
    }


def run_rtl(inputs, out, rtl_root, spdlog, observer=None, *, timeout=120, host_budget=None):
    if type(timeout) is not int or not 1 <= timeout <= 600:
        raise ValueError("RTL timeout must be 1..600 seconds per case")
    rtl_root, spdlog = rtl_root.resolve(), spdlog.resolve()
    observer = observer.resolve() if observer else None
    rtl = rtl_root / "sim-verilator/build/driver_example/debug/sim-VentusRTL"
    library = rtl_root / "sim-verilator/build/libVentusRTL/debug/libVentusRTL.so"
    identities = {"rtl_binary_sha256": sha(rtl), "rtl_library_sha256": sha(library)}
    if observer:
        identities["observer_sha256"] = sha(observer)
    payload = json.loads(inputs.read_text())
    if payload.get("schema") != "ventus-compiled-rtl-input-v1":
        raise ValueError("unexpected compiled RTL input schema")
    validate_names(payload["cases"])
    for case in payload["cases"]:
        if set(case["input_sha256"]) != {"input.metadata", "input.data"}:
            raise ValueError("expected exact metadata/data identities")
        if not case["expected_words"]:
            raise ValueError("expected nonempty output reference")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    rows = []
    start_all = perf_counter()
    for case in payload["cases"]:
        source = (inputs.parent / case["name"]).resolve()
        for name, digest in case["input_sha256"].items():
            if sha(source / name) != digest:
                raise ValueError("frozen RTL input hash differs")
        dest = (out / case["name"]).resolve()
        dest.mkdir()
        command = [
            str(rtl),
            "--kernel",
            f"name={case['name']},metafile={source / 'input.metadata'},datafile={source / 'input.data'}",
            "--sim-time-max",
            "2000000",
            "--snapshot",
            "20000000",
        ]
        addresses = sorted(map(int, case["expected_words"]))
        ranges = []
        for a in addresses:
            if not ranges or a != ranges[-1][1] + 4:
                ranges.append([a, a])
            else:
                ranges[-1][1] = a
        for first, last in ranges:
            command.extend(["--dump-mem", f"0x{first:x},0x{last:x}"])
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (
            str(spdlog / "lib") + ":" + str(library.parent) + ":" + env.get("LD_LIBRARY_PATH", "")
        )
        if observer:
            env.update(
                LD_PRELOAD=str(observer),
                VENTUS_TRACE_PATH=str(dest / "trace.csv"),
                VENTUS_LAUNCH_NOT_BEFORE="512",
            )
        begin = perf_counter()
        remaining = timeout if host_budget is None else host_budget - (perf_counter() - start_all)
        if remaining <= 0:
            for name in ("stdout.log", "stderr.log"):
                (dest / name).write_text("")
            process = {"returncode": -1, "timed_out": True, "output_exhausted": False}
        else:
            process = run_bounded(
                command,
                cwd=dest,
                env=env,
                stdout=dest / "stdout.log",
                stderr=dest / "stderr.log",
                timeout=min(timeout, remaining),
            )
        code, timed_out = process["returncode"], process["timed_out"]
        if process["output_exhausted"]:
            code = -1
        stdout, stderr = [
            (dest / name).read_text(errors="replace") for name in ("stdout.log", "stderr.log")
        ]
        completion = classify_rtl_logs(
            stdout, stderr, kernel_name=case["name"], returncode=code, timed_out=timed_out
        )
        numerical = compare_outputs(stdout, case["expected_words"])
        unsupported = re.findall(r"[^\n]*UNDEFINED INSTRUCTION[^\n]*", stdout + "\n" + stderr)
        status = (
            "timeout"
            if timed_out
            else "unsupported_isa"
            if unsupported
            else "execution_failed"
            if not completion["kernel_completed"]
            else "passed"
            if numerical["outputs_match"]
            else "output_mismatch"
        )
        row = {
            "name": case["name"],
            "elf_sha256": case["elf_sha256"],
            **identities,
            "inputs_sha256": sha(inputs),
            "case_input_sha256": case["input_sha256"],
            "completed": completion["kernel_completed"],
            "functional_ok": completion["kernel_completed"] and numerical["outputs_match"],
            "status": status,
            "unsupported_instruction_diagnostics": unsupported,
            "completion": completion,
            "numerical": numerical,
            "returncode": code,
            "timed_out": timed_out,
            "output_exhausted": process["output_exhausted"],
            "wall_seconds": perf_counter() - begin,
            "logs_sha256": {name: sha(dest / name) for name in ("stdout.log", "stderr.log")},
            "scope": "Concrete finite input/output check on RTL; no universal proof or timing accuracy claim.",
        }
        (dest / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("name", "completed", "functional_ok")}), flush=True)
    result = {
        "schema": "ventus-compiled-rtl-check-v1",
        "cases": rows,
        "all_functional_ok": all(r["functional_ok"] for r in rows),
    }
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("rtl")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rtl-root", type=Path)
    p.add_argument("--spdlog", type=Path)
    p.add_argument("--observer", type=Path)
    p.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.out)
    else:
        from ventus_project.config import load_profile

        profile = load_profile().get("remote", {})
        result = run_rtl(
            args.input,
            args.out,
            args.rtl_root or Path(profile["project_root"]),
            args.spdlog or Path(profile["spdlog_root"]),
            args.observer,
            timeout=args.timeout,
        )
        if not result["all_functional_ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
