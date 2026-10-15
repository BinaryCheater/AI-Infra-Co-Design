"""agentic: run, inspect, replay, serve.

  agentic run     --ledger runs/demo.db --run gemm-001 [--mode observe|step|drive]
  agentic inspect --ledger runs/demo.db [--run gemm-001] [--template tmpl_...]
  agentic replay  --ledger runs/demo.db --run gemm-001 --as gemm-001-replay   (byte-for-byte check)
  agentic serve   --ledger runs/demo.db [--port 8765] [--start gemm-002 --mode step]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading

from ..agents.formulator.scripted import ScriptedFormulator
from ..agents.orchestrator import Controller, Orchestrator
from ..ledger import Ledger, rebuild, replay_signature, template_view, timeline


def _formulator(a):
    from ..chips import load_chip, scope_chip
    chip = scope_chip(load_chip(a.chip)) if getattr(a, "chip", None) else None
    return ScriptedFormulator(chip=chip, extrapolation=getattr(a, "extrapolation", False))


def cmd_run(a):
    L = Ledger(a.ledger)
    ctl = Controller(a.mode)
    o = Orchestrator(L, a.run, _formulator(a), ctl, evaluator_backend=a.evaluator)
    if a.mode == "observe":
        o.run_loop()
    else:
        t = threading.Thread(target=o.run_loop, daemon=True)
        t.start()
        print("commands: enter = continue, m <mode>, p = pause, r <checkpoint seq> = rewind, q = quit")
        while t.is_alive():
            w = ctl.snapshot()["waiting"]
            if w:
                print(f"waiting at {w['checkpoint']} (checkpoint seq {w['seq']}, iteration {w['iteration']}, "
                      f"pending {w['pending']})")
                try:
                    line = input("> ").strip()
                except EOFError:
                    line = "q"
                if line == "q":
                    ctl.stop(); break
                elif line.startswith("m "):
                    ctl.set_mode(line[2:].strip()); ctl.resume()
                elif line == "p":
                    ctl.pause()
                elif line.startswith("r "):
                    ctl.rewind(int(line[2:])); ctl.resume()
                else:
                    ctl.resume()
            else:
                t.join(0.2)
    _print_timeline(L, a.run)
    if a.export:
        export_run(L, a.run, a.export)


def export_run(L, run, path):
    """A JSON summary of one run (story, measurements, kernels by id, optimism history) for records kept in git."""
    from ..ledger.story import story
    ev = L.events(run)
    meas = [L.get(e["artifact"]) for e in ev if e["kind"] == "measured"]
    out = {"run": run, "story": story(L, run), "events": [(e["kind"], e["iteration"], e["artifact"]) for e in ev],
           "measurements": [{k: m.get(k) for k in ("id", "contract", "assignment", "kernel", "predicted_cost", "measured_cost",
                                                   "discrepancy", "utilizations", "dominant_resource", "simulator", "reference_agrees", "ci95", "clocks", "host")}
                            for m in meas],
           "kernels": {m["kernel"]: {"sha256": L.get(m["kernel"])["sha256"], "arch": L.get(m["kernel"]).get("arch")} for m in meas if m.get("kernel")},
           "optimism": [e["payload"] for e in ev if e["kind"] == "optimism"],
           "extrapolating": [e["payload"] for e in ev if e["kind"] == "extrapolating"],
           "finished": next((e["payload"] for e in ev if e["kind"] == "run_finished"), None)}
    json.dump(out, open(path, "w"), indent=1, default=str)
    print("exported", path)


def _print_timeline(L, run):
    for r in timeline(L, run):
        gate = f"{r['family_verdict'] or '-'}/{r['instance_verdict'] or '-'}"
        obj = f"{r['objective']:.2f}" if r["objective"] is not None else "-"
        print(f"{r['iteration']:>3} L{r['level']} {r['workload'] or '-':<24} {r['refines'] or '-':<52} "
              f"{r['structure'] or '-'}({r['bucket'] or '-'}) {r['regime_assumed'] or '-'}"
              f"{'->' + r['regime_actual'] if r['flip'] else ''} | {obj} gap {r['gap']} | {gate} | "
              f"{r['strength']}->{r['strength_final']} | {r['outcome']}")


def cmd_inspect(a):
    L = Ledger(a.ledger)
    run = a.run or (L.runs()[-1] if L.runs() else None)
    if a.template:
        print(json.dumps(template_view(L, a.template, run), indent=2, default=str))
        return
    print("runs:", L.runs())
    if run:
        _print_timeline(L, run)
        v = rebuild(L, run)
        print(f"population {len(v['population'])}  proofs {len(v['proofs'])}  flips {len(v['flips'])}  "
              f"overrides {len(v['overrides'])}")
        for k, i in v["incumbents"].items():
            print(f"  incumbent {k}: {i['cost']:.2f} us  {i['design']}  (iteration {i['iteration']})")


def cmd_replay(a):
    L = Ledger(a.ledger)
    o = Orchestrator(L, a.as_run, ScriptedFormulator(), Controller("observe"))
    o.run_loop()
    sig_a = replay_signature([e for e in L.events(a.run) if e["kind"] != "checkpoint"])
    sig_b = replay_signature([e for e in L.events(a.as_run) if e["kind"] != "checkpoint"])
    same = sig_a == sig_b
    print(f"replay {'IDENTICAL' if same else 'DIFFERS'}: {len(sig_a)} vs {len(sig_b)} events")
    if not same:
        for i, (x, y) in enumerate(zip(sig_a, sig_b)):
            if x != y:
                print("first difference at event", i); print(" ", x); print(" ", y); break
    sys.exit(0 if same else 1)


def cmd_serve(a):
    from ..ui.server import main as serve_main
    serve_main(a.ledger, a.run, a.host, a.port, a.start, a.mode)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agentic", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--ledger", required=True); r.add_argument("--run", required=True)
    r.add_argument("--mode", default="observe", choices=["observe", "step", "drive"])
    r.add_argument("--chip", default=None, help="hardware contract name, e.g. spark-6951-gb10 (default: illustrative)")
    r.add_argument("--evaluator", default="roofline", help="roofline, simulator (the GB10 model, needs --chip of a measured contract; every number labeled by its coverage row) or cuda:<ssh-alias>")
    r.add_argument("--extrapolation", action="store_true", help="append a step that changes a fixed chip field")
    r.add_argument("--export", default=None, help="write a JSON summary of the run here"); r.set_defaults(fn=cmd_run)
    i = sub.add_parser("inspect"); i.add_argument("--ledger", required=True); i.add_argument("--run")
    i.add_argument("--template"); i.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("replay"); p.add_argument("--ledger", required=True); p.add_argument("--run", required=True)
    p.add_argument("--as", dest="as_run", required=True); p.set_defaults(fn=cmd_replay)
    s = sub.add_parser("serve"); s.add_argument("--ledger", required=True); s.add_argument("--run")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8765)
    s.add_argument("--start", help="start a new run with this name on launch")
    s.add_argument("--mode", default="step", choices=["observe", "step", "drive"]); s.set_defaults(fn=cmd_serve)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
