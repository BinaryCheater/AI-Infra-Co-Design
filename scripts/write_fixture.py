"""Regenerate agentic/examples/gemm_tiling/contract.json (the S0 fixture) from the builder."""
import json
from pathlib import Path

from agentic.examples.gemm_tiling import fixture
from agentic.schema import with_id

out = Path(__file__).resolve().parents[1] / "agentic/examples/gemm_tiling/contract.json"
out.write_text(json.dumps(with_id("contract", fixture(), "tmpl"), indent=2) + "\n")
print("wrote", out)
