"""Write an investigation into the graph so the next investigation can find it.

InvestigationCase  -- the case record (status, verdict, probability, pattern, actions, summary)
  -HAS_EVENT->     CaseEvent  one per step: trigger, gather, plan, assess, decide, request, response, memory
  -CASE_TXN->      Transaction (role flagged / affected)
  -CASE_CARD->     Card        (role subject / connected)
  -CASE_DEVICE->   DeviceProfile
  -CASE_PATTERN->  Pattern
  -CASE_CITES->    ClosedCase  (the prior cases used as memory)
vectors            fp  = behavioural fingerprint, emb = embedding of the summary

prior_cases() and similar_agent_cases() read these back, so a device or card
named in one case becomes evidence in the next.
"""
from __future__ import annotations

import json
from datetime import datetime

from casegraph.rag.embed import embed_one


def checkpoint(inv, status: str, p: float | None, pattern: str, actions: list[dict] | None) -> None:
    """Upsert the case vertex mid-investigation so its progression is visible in the graph."""
    attrs = {"case_ref": inv.case["case_id"], "opened_at": inv.case["opened_at"], "trigger_type": inv.case["trigger_type"],
             "status": status, "pattern": pattern, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    if p is not None:
        attrs["fraud_probability"] = round(p, 2)
        attrs["verdict"] = "uncertain"
    if actions is not None:
        attrs["initial_actions"] = json.dumps(actions)
    inv.g.upsert_vertex("InvestigationCase", inv.graph_case_id, attrs)


def persist_case(inv, answer: dict, F, episode: list[dict], similar: dict) -> None:
    g, gid, case = inv.g, inv.graph_case_id, answer["case"]
    nba = answer["next_best_actions"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    g.upsert_vertex("InvestigationCase", gid, {
        "case_ref": answer["case_id"], "opened_at": inv.case["opened_at"], "trigger_type": inv.case["trigger_type"],
        "status": case["status"], "verdict": case["verdict"], "fraud_probability": case["fraud_probability"],
        "pattern": case["pattern"], "exposure_usd": case["exposure_usd"], "sar_filed": answer["sar"]["file"],
        "initial_actions": json.dumps(nba["initial"]), "final_actions": json.dumps(nba["final"]),
        "summary": case["summary"], "updated_at": now})
    events = inv.events + [{"step": len(inv.events) + 1, "kind": "memory", "title": "case written to the graph",
                            "detail": gid}]
    for e in events:
        g.upsert_vertex("CaseEvent", f"{gid}-E{e['step']:02d}",
                        {"step": e["step"], "kind": e["kind"], "title": e["title"][:300], "detail": str(e["detail"])[:2000]})
    g.upsert_edges("HAS_EVENT", "InvestigationCase", "CaseEvent", [(gid, f"{gid}-E{e['step']:02d}", {}) for e in events])
    fid = inv.case["flagged_txn_id"]
    txn_edges = [(gid, fid, {"role": "flagged"})] + [(gid, r["id"], {"role": "affected"}) for r in episode if r["id"] != fid]
    g.upsert_edges("CASE_TXN", "InvestigationCase", "Transaction", txn_edges)
    card_edges = [(gid, inv.ctx["card"], {"role": "subject"})] + [(gid, c, {"role": "connected"})
                                                                   for c in case["connected_card_ids"]]
    g.upsert_edges("CASE_CARD", "InvestigationCase", "Card", card_edges)
    devs = set(case["connected_device_profiles"]) | ({inv.ctx["device"]} if inv.ctx["device"] else set())
    g.upsert_edges("CASE_DEVICE", "InvestigationCase", "DeviceProfile", [(gid, d, {}) for d in devs])
    g.upsert_edges("CASE_PATTERN", "InvestigationCase", "Pattern", [(gid, case["pattern"], {})])
    dist = {c["v_id"]: c.get("distance") for c in similar["by_fp"] + similar["by_text"]}
    g.upsert_edges("CASE_CITES", "InvestigationCase", "ClosedCase",
                   [(gid, c, {"similarity": round(1 - float(dist.get(c) or 0), 4)}) for c in case["similar_prior_cases"]])
    g.upsert_vectors("InvestigationCase", "fp", [(gid, similar["fp"])])
    g.upsert_vectors("InvestigationCase", "emb", [(gid, embed_one(case["summary"]))])
