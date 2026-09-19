"""Autonomous monitor: find what nobody raised an alert on.

    python -m casegraph.monitor            # scan Nov-Dec, investigate what it finds -> monitor/*.json

The 20 exam cases start from a trigger. This goes the other way: it slides a
14-day window across November and December, runs the device_ring
connected-components query on each window, and keeps groups that look like
coordinated abuse (few device profiles, several cards, and either a high average case-memory
score or mostly New-to-account devices behind proxies). Each new
group becomes an investigation with the same agent, triggered as an internal
monitor alert, and is written to the graph like any other case. Groups already
covered by the exam cases are skipped (the agent finds them through case memory).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from casegraph import config as C
from casegraph.agent.investigator import Investigator
from casegraph.agent.tools import Tools, Trace, ds, dt

OUT = C.ROOT / "monitor"


def scan(inv: Investigator, start="2016-11-01", end="2016-12-31", days=14, step=7) -> list[dict]:
    t = datetime.fromisoformat(start)
    stop = datetime.fromisoformat(end)
    found: dict[str, dict] = {}
    while t < stop:
        t1 = t + timedelta(days=days)
        res = inv.g.run_query("device_ring", {"t0": ds(t), "t1": ds(t1), "min_cards": 3, "top": 40})[0]
        for comp in res["components"]:
            devs = sorted(res["component_devices"].get(comp["label"], []))
            cards = sorted(res["component_cards"].get(comp["label"], []))
            n = max(comp["n_txns"], 1)
            new, proxy = comp["n_new_device"] / n, comp["n_proxy"] / n
            if not (1 <= len(devs) <= 3 and 3 <= comp["n_cards"] <= 60):
                continue
            if not (comp["avg_model"] >= 0.4 or (new >= 0.5 and proxy >= 0.5)):
                continue
            key = "|".join(devs)
            prev = found.get(key)
            if not prev or comp["n_cards"] > prev["n_cards"]:
                found[key] = {**comp, "devices": devs, "cards": cards, "new_share": round(new, 2),
                              "proxy_share": round(proxy, 2), "window": [ds(t), ds(t1)]}
        t += timedelta(days=step)
    return sorted(found.values(), key=lambda c: -c["n_cards"])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    inv = Investigator()
    covered = set()
    for f in sorted(C.CASES_DIR.glob("HHG-*.json")):
        covered |= set(json.loads(f.read_text())["case"]["connected_device_profiles"])
    groups = scan(inv)
    tools = Tools(inv.g, Trace())
    print(f"{len(groups)} candidate group(s)")
    log = []
    n = 0
    for gp in groups:
        if set(gp["devices"]) & covered:
            log.append({**gp, "action": "skipped: already covered by an exam case"})
            continue
        dn = tools.device_neighbors(gp["devices"][0], dt(gp["window"][0]), dt(gp["window"][1]),
                                        why="monitor: pick the seed transaction")
        best = None
        for card, txns in dn["card_txns"].items():
            for tid in txns:
                ctx = tools.txn_context(tid, why="monitor: score candidate seed")
                if not best or ctx["txn"]["model_prob"] > best[1]["txn"]["model_prob"]:
                    best = (card, ctx)
            if best and best[1]["txn"]["model_prob"] > 0.5:
                break
        card, ctx = best
        n += 1
        f = ctx["txn"]
        case = {"case_id": f"MON-{n:03d}", "opened_at": ds(dt(f["ts"]) + timedelta(hours=6)),
                "trigger_type": "analyst_request",
                "trigger_text": (f"Autonomous monitor: {gp['n_cards']} cards share {len(gp['devices'])} unusual device "
                                 f"profile(s) between {gp['window'][0][:10]} and {gp['window'][1][:10]} "
                                 f"({gp['new_share']:.0%} New-to-account, {gp['proxy_share']:.0%} behind a proxy). "
                                 f"Review transaction {f['id']} on card {card}."),
                "flagged_txn_id": f["id"], "card_id": card, "customer_id": card.split("-")[0], "risk_score": ""}
        ans = inv.investigate(case)
        (OUT / f"{case['case_id']}.json").write_text(json.dumps({"trigger": case, **ans}, indent=2))
        k = ans["case"]
        print(f"{case['case_id']} {gp['devices'][0][:40]:40s} cards={gp['n_cards']:>3} -> {k['verdict']} "
              f"p={k['fraud_probability']} {k['pattern']} exp=${k['exposure_usd']:,.2f} sar={ans['sar']['file']}")
        log.append({**gp, "action": f"investigated as {case['case_id']}"})
        if n >= 6:
            break
    (OUT / "scan.json").write_text(json.dumps(log, indent=2, default=str))
    inv.g.close()


if __name__ == "__main__":
    main()
