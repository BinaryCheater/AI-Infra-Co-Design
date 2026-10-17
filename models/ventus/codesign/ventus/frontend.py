"""Fixed-priority straight-line instruction supply, separate from the v7 body model.

This is a bounded *hit-path* abstraction of warp_schedule.scala,
InstrBufferV2/SlowDown and ICache.scala. It models packet capacity and replay,
not I-cache misses, redirects, CTA dispatch or complete Chisel ready propagation.
All instructions that can overlap the measured endpoint must be supplied,
including other warps' epilogues. No RTL observations are inputs to this model.
"""

import re
from collections import defaultdict, deque

from .lsu_control import LSUControlMixin
from .online import OnlineEngine, Token

FRONTEND_VERSION = "ventus-source-frontend-v3"


def validate_straight_line(workload):
    """Require a concrete contiguous instruction stream; reject template IR."""
    last_pc = {}
    for op in workload.operations:
        match = re.search(r"(?:^|\.)pc([0-9a-fA-F]{8})$", op.name)
        if match is None or op.kind == "barrier":
            raise ValueError("frontend needs contiguous PC-named straight-line instructions")
        pc = int(match[1], 16)
        key = op.block, op.warp
        if pc % 4 or (key in last_pc and pc != last_pc[key] + 4):
            raise ValueError("frontend does not model branches, omitted instructions or regext")
        if key not in last_pc and pc % 8:
            raise ValueError("frontend currently requires an aligned two-word initial packet")
        last_pc[key] = pc


class FrontendEngine(LSUControlMixin, OnlineEngine):
    """Two-word fetch, two packet slots and one unpack register per warp.

    The two-cycle I-cache hit pipe is not backpressured. A response arriving
    at a full packet FIFO replays its PC and squashes younger requests for that
    warp. Fetch priority is lowest physical warp ID among non-full FIFOs.
    SlowDown is approximated using scoreboard/collector-token readiness; the
    exact downstream combinational RR-ready network is not reproduced.
    """

    def __init__(self, builder, workload):
        validate_straight_line(workload)
        super().__init__(builder, workload)
        self.collector_reuse_delay = 1
        self.rf_waiting = {}
        # Chisel lastGrant is uninitialized. This profile matches the observed
        # Verilator zero initialization; the first search begins after slot 0.
        self.rf_pivot = defaultdict(lambda: 1)
        self.strict_issue_order = True  # RTL RR selects before downstream ready.
        self.idle_jumps = False  # Frontend progresses independently of FU completion.
        self.packets = defaultdict(deque)
        self.unpack = defaultdict(deque)
        self.fetch_index = defaultdict(int)
        self.fetch_pipe = []
        self.instructions = {key: tuple(stream) for key, stream in self.streams.items()}

    def ready_edges(self, key, op):
        if not self.unpack[key] or self.unpack[key][0] != op.name:
            return None
        return super().ready_edges(key, op)

    def schedule_operands(self, op, sm, wid, collector, accept):
        if not op.sources:
            return super().schedule_operands(op, sm, wid, collector, accept)
        slots = op.source_slots if op.source_slots is not None else tuple(range(len(op.sources)))
        token = Token(op, sm, wid, accept)
        self.rf_waiting[(sm, collector)] = {
            "token": token,
            "sources": [(slot, reg) for slot, reg in zip(slots, op.sources, strict=True)],
            "grants": {},
        }
        return token

    def service_operands(self):
        requests = defaultdict(list)
        completed = []
        for (sm, collector), state in self.rf_waiting.items():
            token, grants = state["token"], state["grants"]
            for slot, reg in state["sources"]:
                # Request valid persists through the synchronous return edge.
                # Repeated grants use bank bandwidth and advance RR but the first
                # return supplies the operand.
                if slot not in grants or self.clock < self.b.times[grants[slot]] + 2:
                    bank = (token.wid + int(reg[1:])) % self.hw.rf_banks
                    requests[(sm, reg[0], bank)].append((collector * 4 + slot, collector, slot))
            if len(grants) == len(state["sources"]) and all(
                self.clock >= self.b.times[node] + 2 for node in grants.values()
            ):
                completed.append((sm, collector))
        for key in completed:
            self.rf_waiting.pop(key)
        for key, pending in requests.items():
            sm = key[0]
            for _ in range(min(self.hw.rf_read_ports, len(pending))):
                choice = min(
                    pending, key=lambda r: (r[0] - self.rf_pivot[key]) % (4 * self.hw.collectors)
                )
                requester, collector, slot = choice
                pending.remove(choice)
                self.rf_pivot[key] = (requester + 1) % (4 * self.hw.collectors)
                state = self.rf_waiting[(sm, collector)]
                if slot not in state["grants"]:
                    state["grants"][slot] = self.b.node(
                        state["token"].operation.name + f".operand_slot{slot}",
                        [(state["token"].node, 0), (self.now(), 0)],
                    )
                else:
                    self.b.graph.count("rf_duplicate_grants")

        for (sm, collector), state in self.rf_waiting.items():
            if len(state["grants"]) == len(state["sources"]) and not state.get("scheduled"):
                token = state["token"]
                operands = self.b.node(
                    token.operation.name + ".operands", [(n, 3) for n in state["grants"].values()]
                )
                ready = Token(token.operation, token.sm, token.wid, operands)
                self.at(
                    self.b.times[operands],
                    lambda t=ready, c=collector: self.issue_ready[
                        (t.sm, self.path(t.operation))
                    ].append((c, t)),
                )
                state["scheduled"] = True

    def unpack_ready(self):
        return {
            key
            for key, stream in self.streams.items()
            if stream
            and key[0] in self.active
            and super(FrontendEngine, self).ready_edges(key, stream[0]) is not None
        }

    def packet_loaded(self, key, packet):
        """Extension hook for retaining the physical SlowDown control register."""

    def collect(self):
        # Sample eligibility before this edge's FIFO dequeue/response. Queue has
        # neither flow nor pipe, so a simultaneous dequeue cannot admit a full FIFO.
        choices = {}
        for key in self.streams:
            if (
                key[0] in self.active
                and len(self.packets[key]) < 2
                and self.fetch_index[key] < len(self.instructions[key])
            ):
                sm, wids, _ = self.active[key[0]]
                wid = wids[key[1]]
                if sm not in choices or wid < choices[sm][0]:
                    choices[sm] = wid, key
        before = {key: len(stream) for key, stream in self.streams.items()}
        can_unpack = self.unpack_ready()
        super().collect()
        self.service_operands()
        for key, stream in self.streams.items():
            if len(stream) < before[key]:
                self.unpack[key].popleft()
        for key in can_unpack:
            if not self.unpack[key] and self.packets[key]:
                packet = self.packets[key].popleft()
                self.unpack[key].extend(packet)
                self.packet_loaded(key, packet)
        for entry in tuple(self.fetch_pipe):
            when, key, index = entry
            if when == self.clock:
                self.fetch_pipe.remove(entry)
                if len(self.packets[key]) < 2:
                    self.packets[key].append(
                        [op.name for op in self.instructions[key][index : index + 2]]
                    )
                    self.b.graph.count("frontend_packets")
                else:
                    self.fetch_index[key] = index
                    self.fetch_pipe = [item for item in self.fetch_pipe if item[1] != key]
                    self.b.graph.count("frontend_replays")
        for _, key in choices.values():
            self.fetch_pipe.append((self.clock + 2, key, self.fetch_index[key]))
            self.fetch_index[key] += 2


def predict_span(workload, hardware, *, endpoint_names):
    """Evaluate selected endpoints without retaining a full event DAG.

    This uses the same service recurrence as ``build_graph(frontend=True)``.
    It is intended for search over small concrete programs; neither path is a
    closed-form predictor or a new native Rust execution mode.
    """
    from .cache_acceptance import AcceptanceMixin
    from .session import SummaryBuilder

    workload.validate(hardware)
    endpoints = frozenset(endpoint_names)
    if not endpoints or any(
        name not in {op.name for op in workload.operations} for name in endpoints
    ):
        raise ValueError("endpoint_names must name operations in the supplied workload")
    wanted = {name + ".writeback" for name in endpoints}

    class SpanBuilder(AcceptanceMixin, SummaryBuilder):
        def __init__(self):
            super().__init__(hardware)
            self.first_collect = None
            self.endpoint_times = {}

        def node(self, name, edges=()):
            node = super().node(name, edges)
            cycle = self.times[node]
            if name.endswith(".collect") and self.first_collect is None:
                self.first_collect = cycle
            if name in wanted:
                self.endpoint_times[name.removesuffix(".writeback")] = cycle
            return node

    builder = SpanBuilder()
    graph = FrontendEngine(builder, workload).run()
    if builder.endpoint_times.keys() != endpoints or builder.first_collect is None:
        raise ValueError("requested endpoints must produce writebacks")
    return {
        "model_version": FRONTEND_VERSION,
        "prediction_cycles": max(builder.endpoint_times.values()) - builder.first_collect,
        "first_collect": builder.first_collect,
        "endpoint_times": builder.endpoint_times,
        "counters": graph.counters,
        "events": len(builder.times),
        "full_dag_retained": False,
    }
