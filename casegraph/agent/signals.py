"""Signal analyzers: turn what the tools returned into weighted evidence.

Each finding is an Evidence item with
  * claim / source / ref / entity_ids: exactly what goes into the case file,
  * weight: a log-odds contribution (positive = towards fraud),
  * group: which independent line of evidence it belongs to. The stopping rule
    in the policy needs "at least two independent pieces of evidence", so two
    findings from the same group (say, two device facts) count once.

Nothing here decides an action. That is policy.py's job.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from casegraph.agent.tools import dt
from casegraph.rag.embed import near_threshold

GROUPS = ("model", "sequence", "device", "network", "region", "history", "memory", "customer", "amount")


@dataclass
class Evidence:
    claim: str
    source: str            # graph | document | customer | external
    ref: str
    entity_ids: list[str]
    weight: float = 0.0
    group: str = "history"
    signal: str = ""

    def as_answer(self) -> dict:
        return {"claim": self.claim, "source": self.source, "ref": self.ref, "entity_ids": self.entity_ids}


@dataclass
class Findings:
    evidence: list[Evidence] = field(default_factory=list)
    patterns: dict[str, float] = field(default_factory=dict)   # pattern -> support
    episode: dict[str, dict] = field(default_factory=dict)      # txn_id -> row, candidate affected txns
    connected_cards: set[str] = field(default_factory=set)
    connected_devices: set[str] = field(default_factory=set)
    shared_element: str = ""
    flags: dict[str, bool] = field(default_factory=dict)
    notes: dict[str, object] = field(default_factory=dict)

    def add(self, ev: Evidence) -> None:
        self.evidence.append(ev)

    def support(self, pattern: str, w: float) -> None:
        self.patterns[pattern] = self.patterns.get(pattern, 0.0) + w


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def money(x: float) -> str:
    return f"${x:,.2f}"


# --------------------------------------------------------------------------- detectors
def card_testing(window: list[dict], flagged: dict) -> list[dict] | None:
    """>=3 tiny online authorisations within an hour, then a larger purchase (policy R5)."""
    online = [r for r in window if r["channel"] == "online"]
    for i, r in enumerate(online):
        if r["amount"] >= 10:
            continue
        t0 = dt(r["ts"])
        small = [x for x in online[i:] if x["amount"] < 10 and dt(x["ts"]) - t0 <= timedelta(hours=1)]
        if len(small) < 3:
            continue
        last = dt(small[-1]["ts"])
        bigger = [x for x in online if last <= dt(x["ts"]) <= last + timedelta(hours=24)
                  and x["amount"] >= max(20.0, 5 * max(s["amount"] for s in small))]
        if bigger and any(x["id"] == flagged["id"] for x in small + bigger):
            return small + bigger[:2]
    return None


def structuring(window: list[dict], flagged: dict) -> list[dict] | None:
    """Several online purchases in a short burst, each just under a round
    authorisation limit (the undocumented pattern in the closed cases)."""
    ft = dt(flagged["ts"])
    near = [r for r in window if r["channel"] == "online" and abs((dt(r["ts"]) - ft).total_seconds()) <= 5400]
    under = [r for r in near if near_threshold(r["amount"])]
    if len(under) >= 3 and any(r["id"] == flagged["id"] for r in under):
        span = (dt(under[-1]["ts"]) - dt(under[0]["ts"])).total_seconds() / 60
        if span <= 90:
            return under
    return None


def recurring(rec: dict) -> dict | None:
    """Same amount, same product, same place, repeating on a regular rhythm (policy R7)."""
    best = None
    for key in ("holder_repeats", "card_region_repeats"):
        if key == "card_region_repeats" and not rec.get("_region"):
            continue   # without a billing region a card-level repeat mixes many cardholders
        hits = sorted(rec.get(key) or [], key=lambda h: h["ts"])
        # collapse same-day duplicates
        days = []
        for h in hits:
            d = dt(h["ts"]).date()
            if not days or days[-1][0] != d:
                days.append((d, h))
        if len(days) < 3:
            continue
        gaps = [(days[i + 1][0] - days[i][0]).days for i in range(len(days) - 1)]
        med = statistics.median(gaps)
        regular = sum(1 for g in gaps if abs(g - med) <= max(2, 0.35 * med)) / len(gaps)
        if 5 <= med <= 35 and regular >= 0.5:
            cand = {"source": key, "n": len(days), "median_gap_days": med, "regularity": round(regular, 2),
                    "ids": [h["id"] for _, h in days][-6:], "last": days[-1][1]["ts"]}
            if not best or cand["n"] > best["n"]:
                best = cand
    return best
