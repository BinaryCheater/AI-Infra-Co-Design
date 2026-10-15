"""codegen(T, x) -> CUDA source for a solved tiling design (notes/hardware_calibration_design.md §3.1).

The kernel template `kernel.cu.j2` is the template's program with Tm, Tn, Tk, S, P and beta as
compile-time constants; grid rounding and the tail guard follow `scope.structure`, so a floor-grid
design compiles to a kernel that really drops the tail. The rendered source is content-addressed by
the caller (a `kernel` artifact) so every measurement points at the exact code that produced it.
Rendering uses jinja2 when available and otherwise the `{{NAME}}` subset the template needs.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent


def render(template_text: str, values: dict) -> str:
    try:
        import jinja2
        return jinja2.Environment(undefined=jinja2.StrictUndefined).from_string(template_text).render(**values)
    except ImportError:
        out = template_text
        for k, v in values.items():
            out = out.replace("{{" + k + "}}", str(v)).replace("{{ " + k + " }}", str(v))
        if "{{" in out:
            raise KeyError("unrendered placeholder in kernel template")
        return out


def codegen(T: dict, x: dict) -> str:
    vals = x["values"]
    s = T["scope"]["structure"]
    values = {
        "CONTRACT_ID": T.get("id", "?"), "ASSIGNMENT_ID": x.get("id", "?"),
        "TM": int(vals["Tm"]), "TN": int(vals["Tn"]), "TK": int(vals["Tk"]), "S": int(vals["S"]), "P": int(vals["P"]),
        "BETA": 1 if vals.get("beta") else 0,
        "GRID_CEIL": 1 if s.get("grid_rounding", "ceil") == "ceil" else 0,
        "GUARDED": 1 if s.get("guard", "bounded") == "bounded" else 0,
        # the copy path is chosen at generation time (the workload is known): cp.async when every row stride is 16 B aligned
        "USE_ASYNC": 1 if (int(T["scope"]["workload"]["k"]) % 4 == 0 and int(T["scope"]["workload"]["n"]) % 4 == 0 and int(vals["Tk"]) % 4 == 0 and int(vals["Tn"]) % 4 == 0) else 0,
    }
    if values["TM"] % 16 or values["TN"] % 16:
        raise ValueError("the FP32 kernel needs Tm and Tn to be multiples of 16")
    if not s.get("leading_barrier", True):
        raise ValueError("the kernel always has a leading barrier; a template without one is a different program")
    return render((HERE / "kernel.cu.j2").read_text(), values)
