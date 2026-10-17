"""Experimental causal L1/LSU timing for concrete straight-line FP32 programs.

Requests and return targets change only at the current service edge. Lower L2
and memory timing still use Builder's explicit transaction abstraction. This is
not a translation of every DCache.scala control signal or an RTL accuracy claim.
"""

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from math import ceil

from .frontend import FrontendEngine
from .online import Token
from .timing import Line

MODEL_VERSION = "ventus-causal-memory-experimental-v1"


@dataclass(frozen=True)
class CacheRequest:
    token: Token
    entry: int
    line: int
    group: int
    node: int


@dataclass
class Primary:
    line: int
    targets: dict = field(default_factory=dict)
    release_started: bool = False


class CausalDataCache:
    """One core request slot, two lower returns, pipelined target drain.

    Source anchors: DCache.scala coreReq_Q/memRsp_Q/coreRsp_st2/coreRsp_Q;
    L1MSHR.scala missRspOut_st1 and releasing_stall. Lower returns hold their
    primary ID until all targets drain and the empty-entry acknowledgement fires.
    LRU tags, lower-level arbitration and dirty-eviction timing remain abstractions.
    """

    def __init__(self, builder, sm):
        self.b, self.hw, self.sm = builder, builder.hw, sm
        self.clock = -1
        self.core_request = None
        self.core_response = deque()
        self.response_stage = None
        self.target_response = None
        self.lower_returns = deque()
        self.lower_pending = []
        self.write_pending = []
        self.primary = {}
        self.accepted_this_cycle = False
        self.accept_blocked = False
        self.miss_line = None
        self.events = []

    def node(self, name, *edges):
        return self.b.node(f"sm{self.sm}.causal.{name}", [*edges, (0, self.clock)])

    def record(self, kind, **fields):
        self.events.append({"cycle": self.clock, "kind": kind, **fields})

    def matching(self, line):
        return next((i for i, p in self.primary.items() if p.line == line), None)

    def capacity_available(self, line):
        entry = self.matching(line)
        return (
            len(self.primary[entry].targets) < self.hw.l1_subentries
            if entry is not None
            else len(self.primary) < self.hw.l1_mshrs
        )

    def accept(self, request):
        """Perform the current coreReq handshake, never a future reservation."""
        if (
            self.accepted_this_cycle
            or self.accept_blocked
            or self.core_request is not None
            or self.miss_line == request.line
            or not self.capacity_available(request.line)
        ):
            self.b.graph.count("causal_cache_admission_stalls")
            return False
        node = self.node("core.accept", (request.node, 0))
        self.core_request = CacheRequest(
            request.token, request.entry, request.line, request.group, node
        )
        self.accepted_this_cycle = True
        self.record("core_accept", name=request.token.operation.name, group=request.group)
        return True

    def install(self, primary, node):
        cache = self.b.l1[self.sm, (primary.line // self.hw.line_bytes) % self.hw.l1_sets]
        if primary.line not in cache and len(cache) >= self.hw.l1_ways:
            victim, old = cache.popitem(last=False)
            if old.dirty:
                self.b.memory_transfer(node, victim, write=True)
        cache[primary.line] = Line(node)

    def step(self, cycle, *, core_ready):
        if cycle != self.clock + 1:
            raise ValueError("causal cache must advance exactly one edge")
        self.clock = cycle
        self.accepted_this_cycle = False
        self.miss_line = None
        self.write_pending[:] = [n for n in self.write_pending if self.b.times[n] > cycle]
        # All decisions use the old queue occupancies. coreRsp_Q has no pipe;
        # the two single-entry response queues do have pipe, but no flow.
        core_space = len(self.core_response) < self.hw.threads
        delivered = self.core_response.popleft() if core_ready and self.core_response else None
        stage_move = (
            self.response_stage is not None
            and self.b.times[self.response_stage[1]] < cycle
            and core_space
        )
        stage_space = self.response_stage is None or stage_move
        old_target_valid = self.target_response is not None
        target_move = old_target_valid and stage_space
        target_space = self.target_response is None or target_move
        returning = self.lower_returns[0] if self.lower_returns else None
        releasing = any(p.release_started for p in self.primary.values())
        # A refill occupies the tag write port before releasing_stall registers.
        self.accept_blocked = releasing or returning is not None
        if stage_move:
            request, node = self.response_stage
            node = self.node("core.response.enqueue", (node, 1))
            self.core_response.append((request, node))
            self.response_stage = None
        if target_move:
            request, node = self.target_response
            self.response_stage = request, self.node("response.stage", (node, 1))
            self.target_response = None
        if delivered is not None:
            request, node = delivered
            node = self.node("core.response.accept", (node, 1))
            if self.b.times[node] != cycle:
                raise RuntimeError("cache consumed a response before its causal edge")
            delivered = request, node
            self.record("core_response", name=request.token.operation.name, group=request.group)

        # memRspIn.valid is gated on coreRsp_Q.enq.ready. Its ready remains low
        # while targets exist, although each target may enter missRspOut_st1.
        if returning is not None and core_space:
            entry, lower = returning
            primary = self.primary[entry]
            if not primary.release_started:
                self.install(primary, self.node("refill", (lower, 1)))
            primary.release_started = True
            if primary.targets and target_space:
                slot = min(primary.targets)
                request = primary.targets.pop(slot)
                node = self.node("target.enqueue", (lower, 1), (request.node, 0))
                self.target_response = request, node
                self.record(
                    "target_drain",
                    primary=entry,
                    slot=slot,
                    name=request.token.operation.name,
                    group=request.group,
                )
            elif not primary.targets:
                self.lower_returns.popleft()
                del self.primary[entry]
                self.record("primary_release", primary=entry)
        elif self.core_request is not None and not old_target_valid:
            # This stage runs only on an old request. A newly accepted request
            # becomes visible on the following cycle through the non-flow queue.
            request = self.core_request
            cache = self.b.l1[self.sm, (request.line // self.hw.line_bytes) % self.hw.l1_sets]
            write = request.token.operation.kind == "store"
            entry = self.matching(request.line)
            hit = request.line in cache and entry is None
            if not hit and not write:
                self.miss_line = request.line
            if hit and stage_space and self.response_stage is None:
                record = cache.pop(request.line)
                cache[request.line] = record
                record.dirty |= write
                self.b.graph.count("l1_hit")
                self.response_stage = request, self.node("hit.response", (request.node, 1))
                self.core_request = None
            elif write and entry is None and len(self.write_pending) < self.hw.l1_write_entries:
                # Retain the declared legacy write-miss lower transaction. The
                # local acknowledgement is separate from physical visibility.
                if stage_space and self.response_stage is None:
                    self.b.graph.count("l1_write_miss")
                    queued = self.node("write.mem_req", (request.node, 2))
                    lower = self.b.lower_read(
                        self.b.node("write.cluster", [(queued, 1)]), request.line, allocate=False
                    )
                    self.write_pending.append(
                        self.b.memory_transfer(lower, request.line, write=True)
                    )
                    self.response_stage = request, queued
                    self.core_request = None
            elif not write and self.capacity_available(request.line):
                if entry is None:
                    entry = min(set(range(self.hw.l1_mshrs)) - self.primary.keys())
                    self.primary[entry] = Primary(request.line)
                    start = self.node("primary.allocate", (request.node, 1))
                    queued = self.b.node("causal.mem_req.queues", [(start, 2)])
                    lower = self.b.lower_read(queued, request.line)
                    self.lower_pending.append((entry, lower))
                    self.b.graph.count("l1_read_miss")
                else:
                    self.b.graph.count("l1_merge")
                primary = self.primary[entry]
                slot = min(set(range(self.hw.l1_subentries)) - primary.targets.keys())
                primary.targets[slot] = request
                self.core_request = None
                self.record(
                    "target_allocate",
                    primary=entry,
                    slot=slot,
                    name=request.token.operation.name,
                    group=request.group,
                )

        # memRsp_Q has two non-flow, non-pipe slots. A full queue cannot accept
        # on its dequeue edge. Due lower transactions remain held, never dropped.
        pre_lower_size = len(self.lower_returns) + (
            1 if returning and returning not in self.lower_returns else 0
        )
        due = sorted(
            (item for item in self.lower_pending if self.b.times[item[1]] <= cycle),
            key=lambda item: (self.b.times[item[1]], item[0]),
        )
        if due and pre_lower_size < 2:
            item = due[0]
            self.lower_pending.remove(item)
            self.lower_returns.append(item)
            self.record("lower_response_enqueue", primary=item[0])
        return delivered


class CausalFrontend(FrontendEngine):
    """InputFIFO/AddrCalculate states and finite per-target return queues.

    Uses the v3 hit-path instruction frontend to isolate memory changes. Branch,
    Icache, CTA and shared L2 arbitration gaps remain explicit.
    """

    def __init__(self, builder, workload):
        workload.validate(builder.hw)
        super().__init__(builder, workload)
        self.cache = [CausalDataCache(builder, sm) for sm in range(self.hw.sms)]
        self.address_state = [None] * self.hw.sms
        self.address_clock = [-1] * self.hw.sms
        self.remaining = {}
        self.partial_return = [False] * self.hw.sms
        self.lds_returns = [[] for _ in range(self.hw.sms)]
        self.lds_busy_until = [0] * self.hw.sms

    def writebacks(self):
        self.partial_return = [False] * self.hw.sms
        for sm, cache in enumerate(self.cache):
            ready = self.mshr_next[sm] == "idle"
            response = cache.step(self.clock, core_ready=ready)
            if response is None and ready:
                due = [(r, n) for r, n in self.lds_returns[sm] if self.b.times[n] <= self.clock]
                if due:
                    response = min(due, key=lambda r: self.b.times[r[1]])
                    self.lds_returns[sm].remove(response)
            if response is not None:
                request, node = response
                self.remaining[request.token.operation.name] -= 1
                last = self.remaining[request.token.operation.name] == 0
                self.partial_return[sm] = not last
                if last:
                    result = Token(request.token.operation, sm, request.token.wid, node)
                    self.mshr_arrivals[sm].append((request.entry, result))
                else:
                    self.b.graph.count("causal_partial_returns")
        consumed = super().writebacks()
        for sm, partial in enumerate(self.partial_return):
            if partial:
                self.mshr_accepted_return[sm] = True
                self.mshr_next[sm] = "idle"
        return consumed

    def address(self):
        for sm in range(self.hw.sms):
            if self.address_clock[sm] == self.clock:
                continue
            self.address_clock[sm] = self.clock
            state = self.address_state[sm]
            if state is None:
                token = self.lsu_buffer[sm]
                if token is not None and self.b.times[token.node] < self.clock:
                    self.lsu_buffer[sm] = None
                    node = self.b.node(
                        token.operation.name + ".lsu.dequeue", [(token.node, 1), (self.now(), 0)]
                    )
                    groups = OrderedDict()
                    for address in token.operation.addresses:
                        line = address // self.hw.line_bytes * self.hw.line_bytes
                        groups.setdefault(line, []).append(address)
                    self.address_state[sm] = {
                        "phase": "save",
                        "token": token,
                        "node": node,
                        "groups": deque(groups.items()),
                    }
                continue
            token = state["token"]
            if state["phase"] == "save":
                if self.mshr_state[sm] != "idle" or len(self.mshr_used[sm]) >= self.hw.lsu_entries:
                    continue
                entry = min(set(range(self.hw.lsu_entries)) - self.mshr_used[sm].keys())
                self.mshr_used[sm][entry] = token.operation.name
                self.mshr_tags[sm][entry] = token
                self.mshr_allocated_cycle[sm] = self.clock
                self.remaining[token.operation.name] = len(state["groups"])
                state.update(phase="requests", entry=entry, group=0)
                state["node"] = self.b.node(
                    token.operation.name + ".lsu.save", [(state["node"], 1), (self.now(), 0)]
                )
                if self.mshr_accepted_return[sm]:
                    self.mshr_next[sm] = "add"
                    self.b.graph.count("lsu_mshr_deferred_allocations")
                continue
            line, addresses = state["groups"][0]
            request = CacheRequest(token, state["entry"], line, state["group"], state["node"])
            if 0x70000000 <= line < 0x70000000 + self.hw.lds_bytes:
                if self.lds_busy_until[sm] > self.clock:
                    continue
                counts = Counter((a // 4) % self.hw.lds_banks for a in addresses)
                rounds = max(ceil(n / self.hw.lds_ports) for n in counts.values())
                self.b.graph.count("lds_rounds", rounds)
                node = self.b.node(
                    token.operation.name + f".lds.group{request.group}",
                    [(state["node"], 1), (self.now(), 0)],
                )
                self.lds_returns[sm].append(
                    (request, self.b.node("causal.lds.response", [(node, rounds + 2)]))
                )
                self.lds_busy_until[sm] = self.clock + rounds + (token.operation.kind == "store")
                accepted = True
            else:
                accepted = self.cache[sm].accept(request)
            if accepted:
                state["groups"].popleft()
                state["group"] += 1
                if not state["groups"]:
                    self.address_state[sm] = None


def predict(workload, hardware, *, endpoint_names, trace=False):
    """Run fresh causal state; graph replay checks its event-time accounting."""
    from .graph import execute
    from .timing import Builder

    workload.validate(hardware)
    endpoints = frozenset(endpoint_names)
    by_name = {op.name: op for op in workload.operations}
    if not endpoints or any(
        name not in by_name or not by_name[name].destination for name in endpoints
    ):
        raise ValueError("endpoint_names must identify writeback-producing operations")
    builder = Builder(hardware)
    engine = CausalFrontend(builder, workload)
    graph = engine.run()
    if any(engine.remaining.values()) or any(engine.lds_returns):
        raise RuntimeError("causal LSU completed with unreturned instruction groups")
    for cache in engine.cache:
        if (
            cache.core_request is not None
            or cache.primary
            or cache.lower_pending
            or cache.lower_returns
            or cache.target_response is not None
            or cache.response_stage is not None
            or cache.core_response
        ):
            raise RuntimeError("causal cache completed with undrained requests")
        accepted = [(e["name"], e["group"]) for e in cache.events if e["kind"] == "core_accept"]
        returned = [(e["name"], e["group"]) for e in cache.events if e["kind"] == "core_response"]
        if len(set(accepted)) != len(accepted) or sorted(accepted) != sorted(returned):
            raise RuntimeError("causal cache lost or duplicated a transaction")
    result = execute(graph)
    if result["times"] != builder.times:
        raise RuntimeError("causal event graph replay differs")
    times = dict(zip(graph.names, result["times"], strict=True))
    events = {
        op.name: {
            event: times[op.name + "." + event]
            for event in ("collect", "operands", "issue", "writeback", "endprg")
            if op.name + "." + event in times
        }
        for op in workload.operations
    }
    first = min(e["collect"] for e in events.values())
    endpoints = {name: events[name]["writeback"] for name in endpoints}
    answer = {
        "model_version": MODEL_VERSION,
        "prediction_cycles": max(endpoints.values()) - first,
        "first_collect": first,
        "endpoint_times": endpoints,
        "instruction_events": events,
        "counters": result["counters"],
        "limitations": [
            "Experimental causal L1/LSU; legacy lower L2/memory calendar, LRU L1, hit-only instruction frontend, no branches/CTA/launch/flush fidelity."
        ],
    }
    if trace:
        answer["cache_events"] = [cache.events for cache in engine.cache]
    return answer


def main():
    from .source_ready import main as replay

    replay(candidate_predict=predict, model_version=MODEL_VERSION)


if __name__ == "__main__":
    main()
