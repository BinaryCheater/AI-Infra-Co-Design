"""Read-only integrity and frozen evidence replay for the standalone export."""

import argparse
import hashlib
import json
import subprocess
import sys

from .config import ROOT


def verify_export(root=ROOT, *, archive_only=False):
    manifest = json.loads((root / "archive/export-manifest.json").read_text())
    entries = [
        row for row in manifest["files"] if not archive_only or row["path"].startswith("archive/")
    ]
    if not entries:
        raise ValueError("Export manifest contains no selected artifacts")
    for row in entries:
        path = root / row["path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Export manifest path escapes project")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["export_sha256"]:
            raise ValueError(f"Export artifact changed: {row['path']}")
    result = {"export_integrity": True, "files": len(entries)}
    if archive_only:
        result["scope"] = "archive only; current executable code excluded"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrity-only", action="store_true")
    parser.add_argument(
        "--archive-only",
        action="store_true",
        help="With --integrity-only, verify historical artifacts after current code evolves",
    )
    parser.add_argument(
        "--budget", type=float, default=300, help="Host seconds for frozen Transformer replay"
    )
    args = parser.parse_args()
    if args.archive_only and not args.integrity_only:
        parser.error("--archive-only requires --integrity-only; it cannot authorize frozen replay")
    print(json.dumps(verify_export(archive_only=args.archive_only)), flush=True)
    if args.integrity_only:
        return
    flow = "analysis/ventus_flow_20261006"
    commands = [
        [
            f"{flow}/cost_model/scripts/build_table.py",
            f"{flow}/cost_model/raw",
            "codesign/ventus_costs/table_v1.json",
            "--verify",
        ],
        [f"{flow}/quick_complete_20261006/check.py", "--verify"],
        [f"{flow}/performance-v7/check.py", "--verify"],
        [
            "-m",
            "codesign.ventus",
            "verify-transformer",
            "--out",
            f"{flow}/performance-v7/small-s16-final",
            "--budget",
            str(args.budget),
        ],
    ]
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
    print("All selected frozen evidence replays passed; no RTL/synthesis rerun or result writes.")


if __name__ == "__main__":
    main()
