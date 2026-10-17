"""Pin source MSHRv2 transitions independently of aggregate GEMM accuracy."""

from collections import deque
from types import SimpleNamespace

from codesign.ventus.lsu_control import LSUControlMixin


class Base:
    def writebacks(self):
        for sm, queue in enumerate(self.lsu_output):
            if queue and self.ready:
                token = queue.popleft()
                self.mshr_complete[sm].pop(token.entry)
                self.mshr_used[sm].pop(token.entry)
        return set()


class Controller(LSUControlMixin, Base):
    def __init__(self, state, complete=(), ready=True):
        self.hw = SimpleNamespace(sms=1, lsu_entries=4)
        self.mshr_next = [state]
        self.mshr_complete = [
            {i: SimpleNamespace(entry=i, operation=SimpleNamespace(kind="load")) for i in complete}
        ]
        self.mshr_used = [{i: f"load{i}" for i in complete}]
        self.mshr_arrivals = [deque()]
        self.lsu_output = [deque()]
        self.ready = ready
        self.b = SimpleNamespace(graph=SimpleNamespace(count=lambda *args: None))

    def mshr_output_ready(self, sm):
        return self.ready


def test_last_output_retains_source_empty_out_bubble():
    controller = Controller("out", [0])
    controller.writebacks()
    assert controller.mshr_complete == [{}]
    assert controller.mshr_next == ["out"]
    controller.writebacks()
    assert controller.mshr_next == ["idle"]


def test_blocked_output_returns_idle_without_hidden_output_fifo():
    controller = Controller("out", [0], ready=False)
    controller.writebacks()
    assert controller.mshr_next == ["idle"]
    assert 0 in controller.mshr_complete[0]
    controller.writebacks()
    assert not controller.lsu_output[0]
    assert controller.mshr_next == ["idle"]
    controller.ready = True
    controller.writebacks()
    assert controller.mshr_next == ["out"]
    assert 0 in controller.mshr_complete[0]


def test_add_waits_for_preedge_writeback_readiness_and_defers_returns():
    controller = Controller("add", [0], ready=False)
    controller.mshr_arrivals[0].append((1, object()))
    controller.writebacks()
    assert controller.mshr_next == ["idle"]
    assert len(controller.mshr_arrivals[0]) == 1
    controller.writebacks()
    assert not controller.mshr_arrivals[0]
    assert controller.mshr_next == ["idle"]
