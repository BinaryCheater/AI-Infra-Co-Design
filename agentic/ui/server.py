"""The architect's UI server: static pages, a JSON API over the ledger, and a WebSocket that streams
ledger events and accepts the command set of design §5 (pause, resume, step, mode, edit, inject,
override, rewind, replay, start).

One process: the loop runs in a thread; the server is asyncio. The UI reads only the ledger (through
the API) and the controller snapshot; nothing on screen comes from anywhere else.
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import threading
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import websockets
from websockets.asyncio.server import serve

from ..agents.formulator import catalog
from ..agents.formulator.scripted import ScriptedFormulator
from ..agents.orchestrator import Controller, Orchestrator
from ..ledger import Ledger, rebuild, template_view, timeline

STATIC = Path(__file__).resolve().parent / "static"


class Session:
    """One ledger, at most one live run, any number of browser clients."""

    def __init__(self, ledger: Ledger, run: str | None = None):
        self.ledger = ledger
        self.run = run or (ledger.runs()[-1] if ledger.runs() else None)
        self.ctl = Controller("step")
        self.orch: Orchestrator | None = None
        self.thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set = set()
        ledger.subscribe(self._on_event)

    # ---------------------------------------------------------------- run control
    def live(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, run: str, mode: str = "step", formulator=None):
        if self.live():
            raise RuntimeError("a run is already live")
        self.ctl = Controller(mode)
        self.run = run
        self.orch = Orchestrator(self.ledger, run, formulator or ScriptedFormulator(), self.ctl)
        self.thread = threading.Thread(target=self._guarded_run, name=f"run:{run}", daemon=True)
        self.thread.start()

    def _guarded_run(self):
        try:
            self.orch.run_loop()
        except Exception:
            pass                                   # already recorded in the ledger by the orchestrator
        self._broadcast({"type": "controller", "controller": self.state()["controller"]})

    def state(self) -> dict:
        ctl = self.ctl.snapshot()
        ctl["live"] = self.live()
        ctl["finished"] = bool(self.orch and self.orch.finished)
        ctl["error"] = self.orch.error if self.orch else None
        return {"run": self.run, "runs": self.ledger.runs(), "controller": ctl,
                "timeline": timeline(self.ledger, self.run) if self.run else [],
                "summary": self._summary()}

    def _summary(self) -> dict:
        if not self.run:
            return {}
        v = rebuild(self.ledger, self.run)
        return {"population": [{"id": k, "iteration": e["iteration"],
                                "refines": e["contract"]["provenance"].get("refines"),
                                "workload": e["contract"]["scope"]["workload"].get("name"),
                                "signature": e["contract"].get("signature")} for k, e in v["population"].items()],
                "incumbents": v["incumbents"], "proofs": len(v["proofs"]), "flips": v["flips"],
                "overrides": v["overrides"], "events": self.ledger.last_seq(self.run)}

    # ---------------------------------------------------------------- commands
    def command(self, msg: dict) -> dict:
        op = msg.get("op")
        if op == "start":
            self.start(msg.get("run") or f"run-{len(self.ledger.runs()) + 1:03d}", msg.get("mode", "step"))
        elif op == "select":
            self.run = msg["run"]
        elif op == "mode":
            self.ctl.set_mode(msg["mode"])
        elif op == "pause":
            self.ctl.pause()
        elif op in ("resume", "step"):
            self.ctl.resume()
        elif op == "stop":
            self.ctl.stop()
        elif op == "edit":
            self.ctl.edit(msg["contract"], msg.get("who", "architect"), msg.get("reason", ""))
        elif op == "inject":
            self.ctl.inject(msg["template"], msg.get("who", "architect"), msg.get("reason", ""))
        elif op == "override":
            self.ctl.override(msg["target"], msg["decision"], msg.get("reason", ""), msg.get("who", "architect"))
        elif op == "rewind":
            self.ctl.rewind(int(msg["to"]), bool(msg.get("fresh")), msg.get("who", "architect"))
            self.ctl.resume()
        elif op == "replay":
            cps = self.ledger.checkpoints(self.run)
            if cps:
                self.ctl.rewind(cps[0]["seq"], bool(msg.get("fresh")), msg.get("who", "architect"))
                self.ctl.resume()
        else:
            return {"type": "error", "error": f"unknown op {op!r}"}
        snap = self.state()["controller"]
        self._broadcast({"type": "controller", "controller": snap})
        return {"type": "ok", "op": op, "controller": snap}

    # ---------------------------------------------------------------- streaming
    def _on_event(self, ev: dict):
        self._broadcast({"type": "event", "event": ev})
        if ev["kind"] == "checkpoint":
            self._broadcast({"type": "controller", "controller": self.state()["controller"]})

    def _broadcast(self, msg: dict):
        if self.loop is None or not self.clients:
            return
        data = json.dumps(msg, default=str)
        self.loop.call_soon_threadsafe(self._send_all, data)

    def _send_all(self, data: str):
        for ws in list(self.clients):
            asyncio.ensure_future(self._send(ws, data))

    async def _send(self, ws, data: str):
        try:
            await ws.send(data)
        except Exception:
            self.clients.discard(ws)


# -------------------------------------------------------------------- HTTP API

def _response(connection, status, data: bytes, content_type: str):
    resp = connection.respond(status, "")
    del resp.headers["Content-Length"]          # respond() sets one for the empty body
    del resp.headers["Content-Type"]
    resp.headers["Content-Type"] = content_type
    resp.headers["Content-Length"] = str(len(data))
    resp.headers["Cache-Control"] = "no-store"
    resp.body = data
    return resp


def _json(connection, body, status=HTTPStatus.OK):
    data = json.dumps(body, default=str).encode()
    return _response(connection, status, data, "application/json")


def _static(connection, name: str):
    path = (STATIC / name).resolve()
    if not path.is_file() or STATIC not in path.parents:
        return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
    return _response(connection, HTTPStatus.OK, path.read_bytes(),
                     mimetypes.guess_type(str(path))[0] or "application/octet-stream")


def make_process_request(session: Session):
    def process_request(connection, request):
        url = urlparse(request.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        p = url.path
        if p == "/ws":
            return None                                   # upgrade to WebSocket
        if p == "/" or p == "/index.html":
            return _static(connection, "index.html")
        if p.startswith("/api/"):
            try:
                return _json(connection, api(session, p[5:], q))
            except KeyError as e:
                return _json(connection, {"error": f"not found: {e}"}, HTTPStatus.NOT_FOUND)
        return _static(connection, p.lstrip("/"))
    return process_request


def api(session: Session, route: str, q: dict):
    L = session.ledger
    run = q.get("run") or session.run
    if route == "state":
        return session.state()
    if route == "timeline":
        return timeline(L, run)
    if route == "events":
        return L.events(run, since=int(q.get("since", 0)))
    if route == "catalog":
        return catalog()
    if route == "population":
        return session._summary()
    if route.startswith("template/"):
        return template_view(L, route[9:], run)
    if route.startswith("artifact/"):
        return L.get(route[9:])
    if route == "checkpoints":
        return L.checkpoints(run)
    raise KeyError(route)


async def ws_handler(session: Session, ws):
    session.clients.add(ws)
    try:
        await ws.send(json.dumps({"type": "hello", "state": session.state()}, default=str))
        async for raw in ws:
            try:
                msg = json.loads(raw)
                reply = session.command(msg)
            except Exception as e:
                reply = {"type": "error", "error": f"{type(e).__name__}: {e}"}
            await ws.send(json.dumps(reply, default=str))
    finally:
        session.clients.discard(ws)


async def serve_forever(session: Session, host: str = "127.0.0.1", port: int = 8765):
    session.loop = asyncio.get_running_loop()
    async with serve(lambda ws: ws_handler(session, ws), host, port,
                     process_request=make_process_request(session), max_size=16 * 1024 * 1024) as server:
        print(f"architect UI at http://{host}:{port}/   (ledger {session.ledger.path}, websockets {websockets.__version__})")
        await server.serve_forever()


def main(ledger_path: str, run: str | None = None, host: str = "127.0.0.1", port: int = 8765,
         start: str | None = None, mode: str = "step"):
    session = Session(Ledger(ledger_path), run)
    if start:
        session.start(start, mode)
    asyncio.run(serve_forever(session, host, port))
