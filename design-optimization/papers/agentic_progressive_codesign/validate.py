"""Validate the manuscript's numerical transcriptions and LaTeX references."""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows(name):
    with (ROOT / "evidence" / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    sources = {
        n: json.loads((ROOT / "evidence" / f"ventus-issue-{n}.json").read_text()) for n in (3, 4, 6)
    }
    for row in rows("controlled.csv"):
        for key in ("prefill", "decode"):
            assert row[key] in sources[4]["body"], (row, key)
    for row in rows("fusion.csv"):
        for key in ("fused", "separate"):
            assert row[key] in sources[4]["body"], (row, key)
    trajectory_source = sources[3]["body"] + "\n".join(
        comment["body"] for comment in sources[3]["comments"]
    )
    for row in rows("trajectory.csv"):
        for key in ("prefill", "area"):
            assert row[key] in trajectory_source, (row, key)
    methods = {row["method"]: row for row in rows("controlled.csv")}
    for scenario in ("prefill", "decode"):
        joint = float(methods["Joint"][scenario])
        sequential = float(methods["SW then HW"][scenario])
        print(
            f"{scenario}: baseline reduction {100 * (1 - joint):.4f}%; "
            f"sequential reduction {100 * (1 - joint / sequential):.4f}%"
        )
    fused = rows("fusion.csv")
    print(
        "H0 fusion reduction:",
        f"{100 * (1 - float(fused[0]['fused']) / float(fused[0]['separate'])):.4f}%",
    )
    print(
        "U3 separation reduction:",
        f"{100 * (1 - float(fused[1]['separate']) / float(fused[1]['fused'])):.4f}%",
    )
    text = "\n".join(path.read_text() for path in ROOT.rglob("*.tex"))
    citations = {
        key for group in re.findall(r"\\cite\{([^}]+)\}", text) for key in group.split(",")
    }
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    assert citations <= bibitems, citations - bibitems
    refs = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", text))
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    assert refs <= set(labels), refs - set(labels)
    assert len(labels) == len(set(labels)), "Duplicate labels"
    for name in re.findall(r"\\input\{([^}]+)\}", text):
        assert (ROOT / f"{name}.tex").is_file(), name
    assert not re.search(r"[\u4e00-\u9fff]", text), "Non-English manuscript text"
    assert not re.search(r"\b(?:TODO|TBD|FIXME)\b", text), "Unresolved placeholder"
    assert "evidence/controlled.csv" in text
    assert "evidence/fusion.csv" in text
    assert "evidence/trajectory.csv" in text
    print(f"Validated {len(citations)} citations, {len(labels)} labels, three data tables.")


if __name__ == "__main__":
    main()
