"""Bounded Dcache miss-drain admission for the opt-in frontend model.

This is paired with per-cycle LSU control; enabling it alone changed queue phasing
and regressed some cases. It remains a partial transaction-level abstraction.

Source: pinned L1MSHR.scala 319–364 and DCache.scala 218–219, 616–623.
A returning miss drains one target per cycle, then clears the primary entry;
releasing_stall blocks all new core requests. Target-full and primary-full
requests retain the upstream address path. This is still an abstraction: output
queue backpressure, exact simultaneous allocation/return and tag-port bypass
need separate checks. A later target can extend a return window after earlier
future admissions have been booked; this remains a known approximation. No
measured kernel coefficient enters this model.
"""

from collections import defaultdict

from .timing import Builder


class AcceptanceMixin:
    """Mixin before Builder/SummaryBuilder; uses normal graph event identities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_pending = defaultdict(list)
        self.cache_request_cursor = {}
        self.cache_instruction_accepts = None

    def _cache_gate(self, ready, sm, line):
        gate = ready
        while True:
            now = self.times[gate]
            blocking = []
            active = [p for p in self.cache_pending[sm] if self.times[p["release"]] > now]
            # Tag allocation conflicts at the first returning beat; registered
            # releasing_stall then holds through target drain and primary clear.
            for pending in active:
                if self.times[pending["return"]] <= now:
                    blocking.append(pending["release"])
            same = next((p for p in active if p["line"] == line), None)
            if same and same["targets"] >= self.hw.l1_subentries:
                blocking.append(same["release"])
            if not same and len(active) >= self.hw.l1_mshrs:
                blocking.append(min(active, key=lambda p: self.times[p["release"]])["release"])
            if not blocking:
                return gate, same
            gate = self.node("l1.core_request.wait", [(gate, 0), *((n, 0) for n in blocking)])

    def lower_read(self, *args, **kwargs):
        result = super().lower_read(*args, **kwargs)
        self.cache_lower_return = result
        return result

    def global_line(self, ready, sm, line, write):
        cursor = self.cache_request_cursor.get(sm)
        if cursor is not None:
            ready = self.node("l1.core_request.order", [(ready, 0), (cursor, 1)])
        accepted, same = self._cache_gate(ready, sm, line)
        self.cache_request_cursor[sm] = accepted
        if self.cache_instruction_accepts is not None:
            self.cache_instruction_accepts.append(accepted)
        before = len(self.resources[("l1.mshr", sm)])
        done = super().global_line(accepted, sm, line, write)
        if len(self.resources[("l1.mshr", sm)]) > before:
            # Base fill response = lower + 4: memRsp_Q, missRspOut queue,
            # coreRsp staging and coreRsp_Q. MSHR target drain starts at lower+1.
            lower = self.cache_lower_return
            returning = self.node("l1.mshr.return_begin", [(lower, 1)])
            released = self.node("l1.mshr.primary_release", [(returning, 2)])
            self.cache_pending[sm].append(
                {"line": line, "targets": 1, "return": returning, "release": released}
            )
        elif same is not None and self.times[accepted] < self.times[same["return"]]:
            same["targets"] += 1
            same["release"] = self.node(
                "l1.mshr.primary_release", [(same["return"], same["targets"] + 1)]
            )
            # A secondary response follows the primary's target service. Keep
            # existing response constraints, rather than subtracting their delay.
            done = self.node(
                "l1.secondary.serial_response", [(done, 0), (same["return"], same["targets"] + 2)]
            )
        return done

    def memory_instruction(self, issue, op, sm, wid):
        self.cache_instruction_accepts = []
        try:
            done = super().memory_instruction(issue, op, sm, wid)
            accepts = self.cache_instruction_accepts
            if accepts:
                old = self.resources[("lsu.address.release", sm)][-1]
                released = self.node(
                    op.name + ".dcache.accepted.release",
                    [(old, 0), *((accepted, 1) for accepted in accepts)],
                )
                self.resources[("lsu.address.release", sm)][-1] = released
            return done
        finally:
            self.cache_instruction_accepts = None


class AcceptanceBuilder(AcceptanceMixin, Builder):
    pass
