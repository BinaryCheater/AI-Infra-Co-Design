"""Experimental source translation of SlowDown's downstream ready network.

Not the default v3 model. This changes packet admission, not cache timing.
Sources: ibuffer.scala SlowDown/ibuffer2issue, pipe.scala scoreboard gating,
operandCollector.scala instDemux, and Chisel LockingRRArbiter.grant.
"""

from collections import defaultdict

from .frontend import FrontendEngine

MODEL_VERSION = "ventus-source-ready-experimental-v1"


def rr_ready(valid, last_grant, downstream_ready):
    """Chisel grant includes invalid inputs before the selected valid producer.

    Ready cannot be computed as chosen==index: empty SlowDown registers still
    use that combinational signal to admit the next packet.
    """
    if not valid or not 0 <= last_grant < len(valid):
        raise ValueError("RR needs a nonempty vector and an in-range last grant")
    masked = [request and i > last_grant for i, request in enumerate(valid)]
    requests = masked + list(valid)
    prefix, grants = False, []
    for request in requests:
        grants.append(not prefix)
        prefix |= request
    return tuple(
        bool(downstream_ready and ((grants[i] and i > last_grant) or grants[i + len(valid)]))
        for i in range(len(valid))
    )


class SourceReadyFrontend(FrontendEngine):
    """Retain stale control bits, gate packet loads with pre-edge RR ready.

    Limited to the pinned multi-warp collector geometry. The legacy fetch hit,
    RF and cache/LSU approximations remain. Empty mask selects control slot zero,
    even when its instruction was already consumed; the next instruction's
    hazards must not stand in for those physical output bits.
    """

    def __init__(self, builder, workload):
        super().__init__(builder, workload)
        if not self.source_demux:
            raise ValueError("source ready requires collectors == warps_per_sm > 1")
        self.control_head = {}
        self.operations_by_name = {op.name: op for op in workload.operations}
        # Same zero-initialized Verilator profile as the RF arbiters.
        self.rr_collect = defaultdict(lambda: 1)

    def packet_loaded(self, key, packet):
        self.control_head[key] = self.operations_by_name[packet[0]]

    def _scoreboard_ready(self, key, op):
        # Inspect without ready_edges' dependency-cache side effects: stale
        # control bits are not the next program instruction.
        if key in self.warp_busy:
            return False
        if key in self.last_issue and self.b.times[self.last_issue[key]] >= self.clock:
            return False
        if op is None:  # Reset control register contains no reads/writes.
            return True
        for reg in (*op.sources, *([op.destination] if op.destination else [])):
            writer = self.registers.get((key, reg)) if reg != "x0" else None
            if writer and (
                writer not in self.completion or self.b.times[self.completion[writer]] >= self.clock
            ):
                return False
        return True

    def unpack_ready(self):
        valid = defaultdict(lambda: [False] * self.hw.warps_per_sm)
        states = {}
        for key in self.streams:
            if key[0] not in self.active:
                continue
            sm, wids, _ = self.active[key[0]]
            wid = wids[key[1]]
            op = (
                self.operations_by_name[self.unpack[key][0]]
                if self.unpack[key]
                else self.control_head.get(key)
            )
            path = self.path(op) if op else "scalar"
            ready = self._scoreboard_ready(key, op)
            states[key] = sm, wid, path, ready
            valid[sm, path][wid] = bool(self.unpack[key]) and ready
        downstream = {}
        for sm in range(self.hw.sms):
            free = sum(
                i not in self.collectors[sm] and self.collector_ready_cycle[sm, i] <= self.clock
                for i in self.collector_ids
            )
            for path, required in (("vector", 1), ("scalar", 2)):
                last = (self.rr_collect[sm, path] - 1) % self.hw.warps_per_sm
                downstream[sm, path] = rr_ready(valid[sm, path], last, free >= required)
        return {
            key
            for key, (sm, wid, path, ready) in states.items()
            if ready and downstream[sm, path][wid]
        }


def main(*, candidate_predict=None, model_version=MODEL_VERSION):
    """Replay existing traces only; never launch RTL or overwrite evidence."""
    import argparse
    import csv
    import hashlib
    import json
    from pathlib import Path

    from .cache_acceptance import AcceptanceBuilder
    from .frontend_parity import frozen_cases, python_reference
    from .graph import execute
    from .trace_diff import compare_trace

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--banks", type=int, choices=(4, 8), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    raw_cases = json.loads(args.cases.read_text())
    cases = list(frozen_cases(args.cases, args.banks))
    reports = []
    for raw, case in zip(raw_cases, cases, strict=True):
        report = {"name": raw["name"], "hardware": case["hardware"].to_dict()}
        try:
            trace = args.reference_root / raw["name"] / "trace.csv"
            with trace.open() as stream:
                rows = list(csv.reader(stream))
            report["trace_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
            workload, hardware = case["workload"], case["hardware"]
            legacy = python_reference(workload, hardware, endpoint_names=case["endpoint_names"])
            if candidate_predict is None:
                graph = SourceReadyFrontend(AcceptanceBuilder(hardware), workload).run()
                execution = execute(graph)
                times = dict(zip(graph.names, execution["times"], strict=True))
                events = {
                    op.name: {
                        event: times[op.name + "." + event]
                        for event in ("collect", "writeback")
                        if op.name + "." + event in times
                    }
                    for op in workload.operations
                }
                candidate = {
                    "instruction_events": events,
                    "first_collect": min(e["collect"] for e in events.values()),
                }
            else:
                candidate = candidate_predict(
                    workload, hardware, endpoint_names=case["endpoint_names"]
                )
                report["candidate_counters"] = candidate.get("counters", {})
                report["candidate_limitations"] = candidate.get("limitations", [])
                report["candidate_native_identity"] = {
                    k: candidate[k]
                    for k in ("native_version", "native_source_sha256", "native_runtime_sha256")
                    if k in candidate
                }
            report["paths"] = {}
            for label, prediction in (("v3", legacy), (model_version, candidate)):
                diff = compare_trace(workload, raw["words"], prediction, rows)
                report["paths"][label] = diff
                if diff["status"] not in {"different", "match"}:
                    continue
                end = max(
                    prediction["instruction_events"][n]["writeback"] for n in case["endpoint_names"]
                )
                rtl_ends = [
                    int(r[0])
                    for r in rows
                    if r[1] == "writeback" and int(r[3], 16) == raw["final_tensor_pc"]
                ]
                reference = max(rtl_ends) - diff["alignment"]["rtl_first_collect"]
                cycles = end - prediction["first_collect"]
                diff.update(
                    prediction_cycles=cycles,
                    reference_cycles=reference,
                    absolute_relative_error=abs(cycles - reference) / reference,
                )
            report["status"] = (
                "compared"
                if all("absolute_relative_error" in p for p in report["paths"].values())
                else "incomplete"
            )
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            report.update(status="open", error=str(error))
        reports.append(report)
        print(
            json.dumps(
                {
                    "name": report["name"],
                    "status": report["status"],
                    "errors": {
                        k: v.get("absolute_relative_error")
                        for k, v in report.get("paths", {}).items()
                    },
                }
            ),
            flush=True,
        )
    result = {
        "schema": "ventus.source-ready.replay.v1"
        if candidate_predict is None
        else "ventus.frontend.candidate-replay.v1",
        "model_version": model_version,
        "evidence_role": "post-reference development replay; no new independent RTL",
        "hardware_identity_status": "caller declared; verify reference realization separately",
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "sources": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "cases": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
