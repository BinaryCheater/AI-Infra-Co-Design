"""Per-cycle instruction-MSHR service for the straight-line frontend model.

Pinned pipeline/MSHR.scala has mutually exclusive idle/add/out states. It accepts
cache responses only in idle, defers a simultaneous allocation to add, and emits
complete entries only in out. The source's out transition clears valid_entry
(the *free* index), not output_entry; retaining that behavior matters to service.
Partial-lane/multiple-line returns are still aggregated by the memory builder.
"""

from collections import deque

from .online import Token


class LSUControlMixin:
    def __init__(self, builder, workload):
        super().__init__(builder, workload)
        self.mshr_state = ["idle"] * self.hw.sms
        self.mshr_next = list(self.mshr_state)
        self.mshr_used = [{} for _ in self.mshr_state]
        self.mshr_complete = [{} for _ in self.mshr_state]
        self.mshr_tags = [{} for _ in self.mshr_state]
        self.mshr_arrivals = [deque() for _ in self.mshr_state]
        self.mshr_accepted_return = [False] * self.hw.sms
        self.mshr_allocated_cycle = [-1] * self.hw.sms
        self.address_pending = [None] * self.hw.sms

    def mshr_output_ready(self, sm):
        entry = min(self.mshr_complete[sm]) if self.mshr_complete[sm] else 0
        tag = self.mshr_tags[sm].get(entry)
        if tag is None or tag.operation.kind == "store":
            return True
        path = tag.operation.destination[0] if tag.operation.destination else "v"
        blockers = set()
        for key, pipe in self.pipelines.items():
            if key[0] != sm or key[1] in {"fadd", "fma"} or pipe.output is None:
                continue
            op = pipe.output.operation
            destination_path = op.destination[0] if op.destination else "v"
            if destination_path == path and op.kind in {
                "scalar",
                "vector",
                "fadd",
                "fma",
                "fmax",
                "ftoi",
                "itof",
            }:
                blockers.add("alu" if op.kind in {"scalar", "vector"} else "fpu")
        return len(blockers) < self.hw.writeback_ports

    def writebacks(self):
        self.mshr_state = list(self.mshr_next)
        self.mshr_accepted_return = [False] * self.hw.sms
        pre_complete = [set(entries) for entries in self.mshr_complete]
        output_ready = [self.mshr_output_ready(sm) for sm in range(self.hw.sms)]
        pre_free = [
            min(set(range(self.hw.lsu_entries)) - entries.keys())
            if len(entries) < self.hw.lsu_entries
            else 0
            for entries in self.mshr_used
        ]
        # A blocked output is retained in MSHR data/tag storage, not in a hidden
        # decoupled output FIFO. The next state's valid must be recomputed.
        for queue in self.lsu_output:
            queue.clear()
        for sm, state in enumerate(self.mshr_state):
            if state == "out" and self.mshr_complete[sm]:
                entry = min(self.mshr_complete[sm])
                self.lsu_output[sm].append(self.mshr_complete[sm][entry])
            if state == "idle" and self.mshr_arrivals[sm]:
                entry, token = self.mshr_arrivals[sm].popleft()
                self.mshr_complete[sm][entry] = token
                self.mshr_accepted_return[sm] = True
                self.b.graph.count("lsu_mshr_returns")
            if self.lsu_output[sm] and self.lsu_output[sm][0].operation.kind == "store":
                token = self.lsu_output[sm].popleft()
                done = self.b.node(
                    token.operation.name + ".store.complete", [(token.node, 0), (self.now(), 0)]
                )
                self.retire(token, done)
        consumed = super().writebacks()
        for sm, state in enumerate(self.mshr_state):
            if state == "out":
                fired = len(self.mshr_complete[sm]) < len(pre_complete[sm])
                # The implemented RTL uses valid_entry here, even though an
                # output_entry-based test would remove the empty-out bubble.
                self.mshr_next[sm] = (
                    "out" if fired and pre_complete[sm] - {pre_free[sm]} else "idle"
                )
            elif state == "add":
                self.mshr_next[sm] = "out" if output_ready[sm] and pre_complete[sm] else "idle"
            else:
                self.mshr_next[sm] = (
                    "out" if output_ready[sm] and self.mshr_complete[sm] else "idle"
                )
            self.b.graph.count("lsu_mshr_" + state + "_cycles")
        return consumed

    def retire(self, token, done):
        sm = token.sm
        for entry, name in tuple(self.mshr_used[sm].items()):
            if name == token.operation.name:
                del self.mshr_used[sm][entry]
                self.mshr_complete[sm].pop(entry, None)
        super().retire(token, done)

    def address(self):
        for sm in range(self.hw.sms):
            if (
                self.address_pending[sm] is None
                and self.lsu_buffer[sm] is not None
                and self.address_until[sm] <= self.clock
            ):
                token = self.lsu_buffer[sm]
                self.lsu_buffer[sm] = None
                first = self.b.node(
                    token.operation.name + ".lsu.dequeue", [(token.node, 0), (self.now(), 0)]
                )
                self.address_pending[sm] = token, first
            if self.address_pending[sm] is None:
                continue
            token, first = self.address_pending[sm]
            # InputFIFO/dequeue and AddrCalculate.s_save precede from_addr.fire.
            if (
                self.b.times[first] + 2 > self.clock
                or self.mshr_state[sm] != "idle"
                or self.mshr_allocated_cycle[sm] == self.clock
                or len(self.mshr_used[sm]) >= self.hw.lsu_entries
            ):
                continue
            start = self.b.node(
                token.operation.name + ".lsu.save.admitted", [(first, 0), (0, self.clock - 2)]
            )
            entry = min(set(range(self.hw.lsu_entries)) - self.mshr_used[sm].keys())
            self.mshr_used[sm][entry] = token.operation.name
            self.mshr_tags[sm][entry] = token
            self.mshr_allocated_cycle[sm] = self.clock
            if self.mshr_accepted_return[sm]:
                self.mshr_next[sm] = "add"
                self.b.graph.count("lsu_mshr_deferred_allocations")
            done = self.b.memory_instruction(start, token.operation, sm, token.wid)
            release = self.b.resources[("lsu.address.release", sm)][-1]
            self.address_until[sm] = self.b.times[release]
            self.address_pending[sm] = None
            result = Token(token.operation, sm, token.wid, done)
            # Base memory.done includes the output edge. Feed its preceding
            # response-arrival edge into the actual idle/add/out controller.
            self.at(
                self.b.times[done] - 1,
                lambda s=sm, i=entry, t=result: self.mshr_arrivals[s].append((i, t)),
            )
