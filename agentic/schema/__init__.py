"""JSON schemas for every artifact the agents exchange, and a validator.

Artifacts are the messages. Free text lives only in `rationale`, which is stored and shown and never
parsed. Every artifact has an `id`; ids are content hashes so a replayed run produces identical ids.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import jsonschema

HERE = Path(__file__).resolve().parent
KINDS = ("contract", "assignment", "verdict", "measurement", "diagnosis", "event", "chip", "kernel")


@lru_cache(maxsize=None)
def schema(kind: str) -> dict:
    if kind not in KINDS:
        raise KeyError(kind)
    return json.loads((HERE / f"{kind}.json").read_text())


def validate(kind: str, artifact: dict) -> dict:
    """Raise jsonschema.ValidationError if the artifact does not conform; return it otherwise."""
    jsonschema.validate(artifact, schema(kind), cls=jsonschema.Draft202012Validator)
    return artifact


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


# Recorded but never part of an identity or of a replay comparison: wall-clock fields, and the SMT
# query hash, because Z3's SMT-LIB text carries fresh internal names that differ between calls.
TIMING_KEYS = frozenset({"checker_time", "solver_time", "formulator_time", "evaluator_time", "seconds", "ts",
                         "query_sha256"})


def strip_timing(obj):
    """Drop the volatile fields (see TIMING_KEYS), recursively."""
    if isinstance(obj, dict):
        return {k: strip_timing(v) for k, v in obj.items() if k not in TIMING_KEYS}
    if isinstance(obj, list):
        return [strip_timing(v) for v in obj]
    return obj


def content_id(kind: str, body: dict, prefix: str | None = None) -> str:
    """Deterministic id: a prefix plus the sha256 of the body without its `id` and timing fields."""
    body = strip_timing({k: v for k, v in body.items() if k != "id"})
    h = hashlib.sha256(canonical(body).encode()).hexdigest()[:12]
    return f"{prefix or kind[:4]}_{h}"


def with_id(kind: str, body: dict, prefix: str | None = None) -> dict:
    out = dict(body)
    out["id"] = content_id(kind, body, prefix)
    return validate(kind, out)


def validate_chip(chip: dict) -> dict:
    """A hardware contract: the illustrative one, a measured one from the calibration suite, or ventus."""
    return validate("chip", chip)
