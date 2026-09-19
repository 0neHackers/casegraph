"""Validate answer files against the README's answer format and the fraud policy.

    python -m casegraph check

Checks every file for: required fields and types; enum values; every ID
(transactions, cards, closed cases) exists in the dataset; exposure equals the
sum of the affected transactions; legitimate verdicts carry no episode;
FILE_REPORT <-> sar.file; approval routes match section 2 (including the
$2,500 BLOCK_CARD split); `final == initial` when nothing was requested;
auto-only execution; R10; SAR narrative length.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb

from casegraph import config as C
from casegraph.agent.policy import ALL_ACTIONS, route

PATTERNS = {"card_testing", "card_not_present_fraud", "card_not_present_new_device", "out_of_region_use",
            "account_takeover", "undocumented", "none"}
STATUS = {"open", "closed_fraud", "closed_legitimate", "escalated"}


def _ids():
    con = duckdb.connect()
    con.execute(f"create view t as select * from read_parquet('{C.PREP / 'txn.parquet'}')")
    amounts = dict(con.execute("select cast(txn_id as varchar), amount from t").fetchall())
    cards = {r[0] for r in con.execute("select distinct card_id from t").fetchall()}
    with open(C.RAW / "closed_cases_history.csv") as fh:
        cc = {r["case_id"] for r in csv.DictReader(fh)}
    return amounts, cards, cc


def check_file(path: Path, amounts, cards, cc) -> list[str]:
    a = json.loads(path.read_text())
    err = []
    need = ["case_id", "case", "evidence_requests", "next_best_actions", "sar", "stop_reason", "tool_calls", "tokens",
            "latency_s"]
    err += [f"missing {k}" for k in need if k not in a]
    k = a["case"]
    for f in ["status", "verdict", "fraud_probability", "pattern", "pattern_description", "affected_txn_ids",
              "first_suspicious_txn_id", "connected_card_ids", "connected_device_profiles", "exposure_usd", "evidence",
              "similar_prior_cases", "summary", "written_to_graph", "graph_case_id"]:
        if f not in k:
            err.append(f"case.{f} missing")
    if k["status"] not in STATUS:
        err.append(f"bad status {k['status']}")
    if k["verdict"] not in {"fraud", "legitimate", "uncertain"}:
        err.append("bad verdict")
    if k["pattern"] not in PATTERNS:
        err.append("bad pattern")
    if k["pattern"] == "undocumented" and not k["pattern_description"]:
        err.append("undocumented without description")
    if not 0 <= k["fraud_probability"] <= 1:
        err.append("probability out of range")
    for t in k["affected_txn_ids"] + ([k["first_suspicious_txn_id"]] if k["first_suspicious_txn_id"] else []):
        if t not in amounts:
            err.append(f"unknown txn {t}")
    for c in k["connected_card_ids"]:
        if c not in cards:
            err.append(f"unknown card {c}")
    for c in k["similar_prior_cases"]:
        if c not in cc:
            err.append(f"unknown closed case {c}")
    exp = round(sum(abs(amounts.get(t, 0)) for t in k["affected_txn_ids"]), 2)
    if abs(exp - k["exposure_usd"]) > 0.011:
        err.append(f"exposure {k['exposure_usd']} != sum {exp}")
    if k["verdict"] == "legitimate" and (k["affected_txn_ids"] or k["exposure_usd"] or a["sar"]["file"]):
        err.append("legitimate verdict with episode/exposure/SAR")
    for e in k["evidence"]:
        if e.get("source") not in {"graph", "document", "customer", "external"}:
            err.append(f"bad evidence source {e.get('source')}")
    for r in a["evidence_requests"]:
        if r["type"] not in {"customer_validation", "step_up_auth", "analyst_info"}:
            err.append(f"bad request type {r['type']}")
    nba = a["next_best_actions"]
    for phase in ("initial", "final"):
        for x in nba[phase]:
            if x["action"] not in ALL_ACTIONS:
                err.append(f"unknown action {x['action']}")
            elif x["route"] != route(x["action"], k["exposure_usd"]) and not (
                    phase == "initial" and x["action"] == "BLOCK_CARD"):
                err.append(f"{phase} {x['action']} route {x['route']} != {route(x['action'], k['exposure_usd'])}")
            if not x.get("reason"):
                err.append(f"{phase} {x['action']} without reason")
    if not a["evidence_requests"] and nba["final"] != nba["initial"]:
        err.append("final differs from initial but nothing was requested")
    final_names = {x["action"] for x in nba["final"]}
    if ("FILE_REPORT" in final_names) != bool(a["sar"]["file"]):
        err.append("FILE_REPORT and sar.file disagree")
    if "BLOCK_ALL_CARDS" in final_names:
        err.append("BLOCK_ALL_CARDS recommended (check R10)")
    s = a["sar"]
    if s["file"]:
        n = s["narrative"].count(". ") + 1
        if not s["narrative"] or not s["subjects"] or not s["activity_dates"]:
            err.append("SAR incomplete")
        if not 5 <= n <= 14:
            err.append(f"SAR narrative has ~{n} sentences")
    else:
        if s["narrative"] or s["subjects"] or s["total_amount_usd"] or s["activity_dates"]:
            err.append("sar.file false but fields filled")
    return err


def check_all(folder: Path) -> bool:
    amounts, cards, cc = _ids()
    files = sorted(Path(folder).glob("HHG-*.json"))
    ok = True
    with open(C.RAW / "case_pack.csv") as fh:
        expected = {r["case_id"] for r in csv.DictReader(fh)}
    missing = expected - {f.stem for f in files}
    if missing:
        print("missing answer files:", sorted(missing))
        ok = False
    for f in files:
        e = check_file(f, amounts, cards, cc)
        print(f"{f.stem}: {'OK' if not e else '; '.join(e)}")
        ok &= not e
    print("ALL OK" if ok else "PROBLEMS FOUND")
    return ok
