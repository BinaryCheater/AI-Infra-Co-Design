"""Explicit scope of the compiled causal model; no numerical/accuracy certificate."""


def contract(hardware):
    return {
        "version": "ventus-compiled-mechanisms-v1",
        "integer_units": {
            "multiply": "Independent vector issue path; two multiplier stages and one result queue; initiation interval one",
            "divide": "Shared serialized SFU; IntDivMod input-dependent idle/pre/compute/recovery/finish; masked groups of max(1,threads/4) lanes",
            "unknown_division_operands": "Rejected; supply functional input words rather than a constant timing guess",
        },
        "dynamic_memory": {
            "controller": "Current-edge InputFIFO/address/instruction-MSHR/L1 primary+target/response state",
            "conservation": "Exactly-once group, instruction and external responses; final queues and writes drained",
            "l2": "Bounded directory/MSHR/SourceD stages and writeback; full inclusive probes/SRAM-port arbitration remain approximate",
        },
        "external_target": {
            "kind": "pinned Mem_SimWrapper"
            if hardware.memory_channels == 1
            and hardware.memory_bytes_per_cycle == hardware.line_bytes
            and hardware.memory_latency == 2
            else "explicit parametric extension of wrapper; not the fixed RTL realization",
            "response_slots_per_channel": 5,
            "request_ready": "Old slot occupancy, no same-edge full-slot reuse",
            "response_selection": "Lowest eligible slot; consumer backpressure retains occupancy",
            "delay": "LDS-address range delay zero; other addresses use memory_latency; registered response eligibility",
            "dram_bank_row_controller": "not modeled",
        },
        "instruction_supply": {
            "compiled_elf": "Dynamic decoded stream; instruction fetch, cold I-cache and redirect timing excluded",
            "probe_frontend": "Finite hit-path fetch/packet/replay model; same causal data-memory core",
        },
        "coverage_limits": [
            "Finite RTL references are required for numerical claims; timing completion alone is insufficient",
            "Packed byte/halfword global stores have an R1 numerical counterexample; not certified",
            "Workgroup barriers retain conservative block timing fences",
            "Subgroup/atomic/device-system fence and extended precision lack baseline RTL validation",
            "Per-case timing accuracy targets are not established by this implementation",
        ],
    }
