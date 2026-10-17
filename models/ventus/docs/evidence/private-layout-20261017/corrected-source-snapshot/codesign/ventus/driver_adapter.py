"""Build an explicit, hash-bound R1 host-input adapter; never edit the RTL tree."""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from .process import run_bounded

PROFILE = "workgroup-private-v1"
PROFILES = ("upstream-r1", PROFILE)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verilator_headers(text, configured=None):
    candidates = (
        [configured]
        if configured
        else re.findall(r"^\s*VERILATOR_ROOT\s*=\s*(\S+)\s*$", text, re.M)
    )
    for value in candidates:
        root = Path(value)
        if root.is_absolute() and (root / "include/verilated.h").is_file():
            return root
    raise ValueError("configured Verilator headers could not be located")


def build(rtl_root, out):
    out.mkdir()
    source = Path(__file__).with_name("runtime") / "r1-private-driver.cpp"
    headers = rtl_root / "sim-verilator/build/libVentusRTL/debug"
    verilator = shutil.which("verilator")
    compiler = shutil.which("g++")
    if not verilator or not compiler:
        raise ValueError(
            "private driver adapter needs the configured target's Verilator and C++ compiler"
        )
    version = run_bounded(
        [verilator, "-V"],
        cwd=out,
        stdout=out / "verilator.stdout",
        stderr=out / "verilator.stderr",
        timeout=30,
    )
    if version["returncode"] or version["timed_out"] or version["output_exhausted"]:
        raise ValueError("Verilator toolchain discovery failed; see adapter logs")
    tool_root = verilator_headers(
        (out / "verilator.stdout").read_text(), os.environ.get("VERILATOR_ROOT")
    )
    binary = out / "r1-private-driver.so"
    process = run_bounded(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-shared",
            str(source),
            "-I" + str(headers),
            "-I" + str(tool_root / "include"),
            "-I" + str(tool_root / "include/vltstd"),
            "-ldl",
            "-o",
            str(binary),
        ],
        cwd=out,
        stdout=out / "compile.stdout",
        stderr=out / "compile.stderr",
        timeout=120,
    )
    receipt = {
        "profile": PROFILE,
        "process": process,
        "source_sha256": sha(source),
        "target_header_sha256": sha(headers / "Vdut.h"),
        "target_library_sha256": sha(headers / "libVentusRTL.so"),
        "compiler_sha256": sha(Path(compiler)),
        "verilator_header_sha256": sha(tool_root / "include/verilated.h"),
        "completed": process["returncode"] == 0
        and not process["timed_out"]
        and not process["output_exhausted"],
    }
    if receipt["completed"]:
        receipt["binary_sha256"] = sha(binary)
    (out / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["completed"]:
        raise ValueError("private driver adapter compilation failed; see retained adapter logs")
    return binary, receipt
