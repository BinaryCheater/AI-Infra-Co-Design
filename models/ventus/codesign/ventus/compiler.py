"""Compile real OpenCL kernels with a configured official Ventus LLVM checkout."""

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

from .elf import ELFProgram
from .process import run_bounded


def compile_kernel(
    source,
    kernel,
    compiler_root,
    llvm_source,
    out,
    *,
    optimization="O1",
    workitem_profile="upstream",
    timeout=120,
    host_budget=None,
):
    if optimization not in {"O1", "O2", "O3"}:
        raise ValueError("unsupported optimization level")
    if type(timeout) is not int or not 1 <= timeout <= 3600:
        raise ValueError("compiler timeout must be 1..3600 seconds per command")
    if host_budget is not None and (not math.isfinite(host_budget) or host_budget <= 0):
        raise ValueError("compiler host budget must be finite and positive")
    compiler_root, llvm_source = compiler_root.resolve(), llvm_source.resolve()
    clang, objdump = compiler_root / "bin/clang", compiler_root / "bin/llvm-objdump"
    workitem = llvm_source / "libclc/riscv32/lib/workitem"
    if workitem_profile not in {"upstream", "rtl-r1"}:
        raise ValueError("unknown workitem library profile")
    assembly_source = workitem / "workitem.S"
    runtime_identity = {"profile": workitem_profile}
    if workitem_profile == "rtl-r1":
        runtime = Path(__file__).parent / "runtime"
        assembly_source = runtime / "rtl-r1-workitem.S"
        runtime_identity.update(json.loads((runtime / "rtl-r1-workitem.json").read_text()))
        if (
            hashlib.sha256(assembly_source.read_bytes()).hexdigest()
            != runtime_identity["assembly_sha256"]
        ):
            raise ValueError("R1 compatibility library differs from its source receipt")
    helpers = sorted(workitem.glob("*.cl"))
    if not clang.is_file() or not objdump.is_file() or not helpers:
        raise ValueError("official Ventus compiler or workitem sources absent")
    out.mkdir()
    source = source.resolve()
    begin = perf_counter()
    target = [str(clang), "-target", "riscv32", "-mcpu=ventus-gpgpu"]
    assembly = out / "workitem.o"
    commands = [
        [
            *target,
            "-I" + str(workitem.parent),
            "-c",
            str(assembly_source),
            "-o",
            str(assembly),
        ],
        [
            *target,
            "-cl-std=CL2.0",
            "-" + optimization,
            "-I" + str(llvm_source / "libclc/generic/include"),
            "-nodefaultlibs",
            "-fuse-ld=lld",
            "--ld-path=" + str(compiler_root / "bin/ld.lld"),
            "-Wl,-T," + str(llvm_source / "utils/ldscripts/ventus/elf32lriscv.ld"),
            "-Wl,-e," + kernel,
            str(source),
            *map(str, helpers),
            str(assembly),
            "-o",
            str(out / "program.elf"),
        ],
        [str(objdump), "-d", "--mattr=+v,+zfinx", str(out / "program.elf")],
    ]
    logs = []
    status = 0
    process_records = []
    for index, command in enumerate(commands):
        stdout, stderr = out / f"command-{index}.stdout", out / f"command-{index}.stderr"
        remaining = timeout if host_budget is None else host_budget - (perf_counter() - begin)
        if remaining <= 0:
            stdout.write_text("")
            stderr.write_text("compiler host budget exhausted\n")
            result = {
                "returncode": -1,
                "timed_out": True,
                "output_exhausted": False,
                "wall_seconds": 0,
            }
        else:
            result = run_bounded(
                command, cwd=None, stdout=stdout, stderr=stderr, timeout=min(timeout, remaining)
            )
        process_records.append(result)
        logs.append(stderr.read_text(errors="replace"))
        status = result["returncode"]
        if result["timed_out"] or result["output_exhausted"]:
            status = 1
        if status:
            break
        if command[0] == str(objdump):
            (out / "program.dump").write_bytes(stdout.read_bytes())
    # This output is an ignored build directory. Portable receipts bind bytes
    # and use symbolic paths; diagnostics remain local to the build machine.
    (out / "build.log").write_text("\n".join(logs))
    sources = [source, *helpers, assembly_source, workitem.parent / "ventus.h"]
    receipt = {
        "completed": status == 0,
        "exit_code": status,
        "kernel": kernel,
        "target": "riscv32/ventus-gpgpu",
        "optimization": optimization,
        "command_timeout_seconds": timeout,
        "host_budget_seconds": host_budget,
        "processes": process_records,
        "support_sha256": {
            "linker_script": hashlib.sha256(
                (llvm_source / "utils/ldscripts/ventus/elf32lriscv.ld").read_bytes()
            ).hexdigest(),
            "generic_headers": tree_hash(llvm_source / "libclc/generic/include"),
        },
        "workitem_library": runtime_identity,
        "host_seconds": perf_counter() - begin,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "compiler_sha256": hashlib.sha256(clang.read_bytes()).hexdigest(),
        "linker_sha256": hashlib.sha256((compiler_root / "bin/ld.lld").read_bytes()).hexdigest(),
        "objdump_sha256": hashlib.sha256(objdump.read_bytes()).hexdigest(),
    }
    if status == 0:
        program = ELFProgram.read(out / "program.elf")
        receipt.update({"elf_sha256": program.sha256, "resources": program.resources})
        if kernel not in program.resources or kernel not in program.symbols:
            status = 1
            receipt.update(
                {"completed": False, "exit_code": 1, "error": "requested kernel entry absent"}
            )
    (out / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if status:
        raise ValueError("official kernel compilation failed; see the new result's build.log")
    print(json.dumps({"out": str(out), "completed": True, "elf_sha256": receipt["elf_sha256"]}))


def tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
