"""The investigator's tools. Every graph/vector call goes through MCPGraph and is logged.

Each tool returns plain Python data; the Trace records name, arguments, latency,
transport and a one-line result digest so the case file and the UI can show
exactly what the agent looked at.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from casegraph.agent.mcp_client import MCPGraph

FMT = "%Y-%m-%d %H:%M:%S"


def dt(s: str) -> datetime:
    return datetime.strptime(str(s)[:19], FMT)


def ds(d: datetime) -> str:
    return d.strftime(FMT)


def _attrs(v: dict) -> dict:
    """Vertex-set rows: strip 'R.' style prefixes from printed attribute names."""
    a = v.get("attributes", {})
    return {k.split(".", 1)[-1]: val for k, val in a.items()} | {"v_id": v.get("v_id")}


@dataclass
class ToolCall:
    step: int
    tool: str
    args: dict
    latency_s: float
    transport: str
    digest: str
    why: str = ""


@dataclass
class Trace:
    calls: list[ToolCall] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.calls)


class Tools:
    """Graph + retrieval tools. `ref` strings are what the case evidence cites."""

    def __init__(self, g: MCPGraph, trace: Trace):
        self.g = g
        self.trace = trace
        self.step = 0

    def _q(self, name: str, params: dict, digest_fn, why: str = "") -> Any:
        t0 = time.time()
        out = self.g.run_query(name, params)
        dig = digest_fn(out) if out is not None else "no result"
        self.trace.calls.append(ToolCall(self.step, name, params, round(time.time() - t0, 3), self.g.transport, dig, why))
        return out

    # ---------------------------------------------------------------- context
    def txn_context(self, txn_id: str, why: str = "") -> dict:
        out = self._q("txn_context", {"txn_id": txn_id},
                      lambda o: f"txn {txn_id}", why)
        t = _attrs(out[0]["txn"][0]) if out and out[0].get("txn") else {}
        meta = out[1] if len(out) > 1 else {}
        dev = (meta.get("device") or [""])[0]
        reg = (meta.get("region") or [""])[0]
        return {"txn": t, "card": (meta.get("card") or [""])[0], "customer": (meta.get("customer") or [""])[0],
                "holder": (meta.get("holder") or [""])[0], "holder_txns": sum((meta.get("holder_txns") or {}).values()),
                "device": dev, "device_cards_total": (meta.get("device_cards_total") or {}).get(dev, 0),
                "device_txns_total": (meta.get("device_txns_total") or {}).get(dev, 0),
                "region": reg, "region_cards_total": (meta.get("region_cards_total") or {}).get(reg, 0),
                "email_domains": meta.get("email_domains") or []}

    def card_window(self, card_id: str, t0: datetime, t1: datetime, why: str = "") -> list[dict]:
        out = self._q("card_window", {"card_id": card_id, "t0": ds(t0), "t1": ds(t1)},
                      lambda o: f"{o[0]['n_in_window']} txns on {card_id}", why)
        rows = out[0]["txns"] if out else []
        return sorted(rows, key=lambda r: (r["ts"], r["id"]))

    def behaviour_profile(self, card_id, holder_id, before, device, region, p_email, product, amount, why="") -> dict:
        out = self._q("behaviour_profile", {"card_id": card_id, "holder_id": holder_id, "before_ts": ds(before),
                                            "device_id": device or "", "region": region or "", "p_email": p_email or "",
                                            "product": product or "", "amount": float(amount)},
                      lambda o: f"card n={o[0]['card_n']} holder n={o[1]['holder_n']}", why)
        return {**out[0], **out[1]}

    def prior_cases(self, card_id, holder_id, device, before, why="") -> dict:
        out = self._q("prior_cases", {"card_id": card_id, "holder_id": holder_id or "", "device_id": device or "",
                                      "before_ts": ds(before)},
                      lambda o: f"{len(o[0]['closed_cases'])} closed, {len(o[0]['agent_cases'])} agent cases", why)
        return {"closed": out[0]["closed_cases"], "agent": out[0]["agent_cases"]}

    def region_context(self, card_id, region, at, window_h=72, why="") -> dict:
        out = self._q("region_context", {"card_id": card_id, "region": region, "at_ts": ds(at), "window_h": window_h},
                      lambda o: f"prior in region={o[0]['card_prior_txns_in_region']}", why)
        return out[0]

    def device_neighbors(self, device, t0, t1, why="") -> dict:
        out = self._q("device_neighbors", {"device_id": device, "t0": ds(t0), "t1": ds(t1)},
                      lambda o: f"{o[0]['n_cards_in_window']} cards on device in window", why)
        res = dict(out[0])
        txns = {}
        if len(out) > 1:
            for v in out[1].get("CARDS", []):
                a = v["attributes"]
                txns[v["v_id"]] = a.get("txns") or a.get("CARDS.@txns") or a.get("@txns") or []
        res["card_txns"] = txns
        return res

    def device_ring(self, t0, t1, seed_device="", seed_card="", why="") -> dict:
        out = self._q("device_ring", {"seed_device": seed_device, "seed_card": seed_card, "t0": ds(t0), "t1": ds(t1)},
                      lambda o: f"{len(o[0]['components'])} component(s)", why)
        return out[0]

    def device_community(self, device, before, why="") -> dict:
        out = self._q("device_community", {"device_id": device, "before_ts": ds(before)},
                      lambda o: f"community {o[0]['community']}: {o[0]['n_holders']} holders, "
                                f"{len(o[0]['closed_case_outcome'])} earlier closed cases", why)
        return out[0]

    def recurring_check(self, holder, card, region, product, amount, before, why="") -> dict:
        out = self._q("recurring_check", {"holder_id": holder, "card_id": card, "region": region or "", "product": product,
                                          "amount": float(amount), "before_ts": ds(before)},
                      lambda o: f"{len(o[0]['holder_repeats'])} holder / {len(o[0]['card_region_repeats'])} card repeats", why)
        return out[0]

    # ---------------------------------------------------------------- vectors
    def similar_cases_fp(self, fp: list[float], k=12, why="") -> list[dict]:
        out = self._q("similar_cases_fp", {"qv": fp, "k": k}, lambda o: f"{len(o[0]['cases'])} by fingerprint", why)
        dist = out[1].get("distance", {}) if len(out) > 1 else {}
        return sorted([_attrs(v) | {"distance": dist.get(v["v_id"])} for v in out[0]["cases"]],
                      key=lambda r: r["distance"] if r["distance"] is not None else 9)

    def similar_cases_text(self, qv: list[float], k=8, why="") -> list[dict]:
        out = self._q("similar_cases_text", {"qv": qv, "k": k}, lambda o: f"{len(o[0]['cases'])} by narrative", why)
        dist = out[1].get("distance", {}) if len(out) > 1 else {}
        return sorted([_attrs(v) | {"distance": dist.get(v["v_id"])} for v in out[0]["cases"]],
                      key=lambda r: r["distance"] if r["distance"] is not None else 9)

    def similar_agent_cases(self, fp: list[float], k=5, why="") -> list[dict]:
        out = self._q("similar_agent_cases", {"qv": fp, "k": k}, lambda o: f"{len(o[0]['cases'])} agent cases", why)
        dist = out[1].get("distance", {}) if len(out) > 1 else {}
        return [_attrs(v) | {"distance": dist.get(v["v_id"])} for v in out[0]["cases"]]

    def search_knowledge(self, qv: list[float], k=5, why="") -> list[dict]:
        out = self._q("search_knowledge", {"qv": qv, "k": k}, lambda o: f"{len(o[0]['chunks'])} chunks", why)
        dist = out[-1].get("distance", {}) if out else {}
        return sorted([_attrs(v) | {"distance": dist.get(v["v_id"])} for v in out[0]["chunks"]],
                      key=lambda r: r["distance"] if r["distance"] is not None else 9)


def around(ts: datetime, before_h: float, after_h: float) -> tuple[datetime, datetime]:
    return ts - timedelta(hours=before_h), ts + timedelta(hours=after_h)
