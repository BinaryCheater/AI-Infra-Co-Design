"""F4, the parts that need no GPU: a scripted run on the measured spark-6951 contract with the roofline
backend, plus the extrapolation step: the template that changes smem_bytes_per_block is marked
`extrapolating`, measured by roofline, its strength capped at observed, and the story says so."""
from pathlib import Path

import pytest

from agentic.agents.formulator.scripted import ScriptedFormulator
from agentic.agents.orchestrator import Controller, Orchestrator
from agentic.chips import catalog, extrapolation, hardware_contract, load_chip, scope_chip
from agentic.ledger import Ledger, rebuild
from agentic.ledger.story import story

CHIP = "spark-6951-gb10"


def test_chip_loader():
    if CHIP not in catalog():
        pytest.skip("no measured contract")
    c = load_chip(CHIP)
    hw = hardware_contract(scope_chip(c))
    assert hw.sms == 48 and hw.smem_bytes_per_block == 101376 and 0 < hw.barrier_s < 1e-6
    assert extrapolation(scope_chip(c)) == {}
    assert extrapolation({**scope_chip(c), "smem_bytes_per_block": 131072}) == {"smem_bytes_per_block": {"measured": 101376, "template": 131072}}
    assert extrapolation({"name": "illustrative-fp32-gpu", "kind": "illustrative"}) == {}


def test_f4_extrapolation_headless(tmp_path):
    if CHIP not in catalog():
        pytest.skip("no measured contract")
    L = Ledger(tmp_path / "x.db")
    f = ScriptedFormulator(chip=scope_chip(load_chip(CHIP)), extrapolation=True)
    Orchestrator(L, "x", f, Controller("observe"), evaluator_backend="roofline").run_loop()
    ex = L.events("x", kinds=["extrapolating"])
    assert len(ex) == 1 and "smem_bytes_per_block" in ex[0]["payload"]["fields"]
    it = ex[0]["iteration"]
    adm = [e for e in L.events("x", kinds=["admitted"]) if e["iteration"] == it]
    assert adm and adm[0]["payload"]["extrapolating"] and adm[0]["payload"]["evaluator"] == "roofline"
    assert adm[0]["payload"]["strength"] == "observed"
    s = {e["iteration"]: e for e in story(L, "x")}
    assert "changes the chip" in s[it]["sentence"] and "only assumed" in s[it]["sentence"]
    # the measured contract is in every template's scope, and the solver respected its capacity
    T = L.get(adm[0]["artifact"])
    assert T["scope"]["chip"]["name"] == CHIP and T["scope"]["chip"]["kind"] == "measured"
    v = rebuild(L, "x")
    assert len(v["population"]) == 6                 # five accepted scripted templates + the extrapolating one
