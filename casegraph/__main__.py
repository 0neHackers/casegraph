"""CaseGraph command line.

    python -m casegraph run HHG-006            # investigate one case from the case pack
    python -m casegraph run --all              # all 20, writes cases/<case_id>.json
    python -m casegraph run HHG-002 --simulate deny   # force the assumed customer reply
    python -m casegraph check                  # validate every answer file against the format and policy
    python -m casegraph reset-memory           # forget the agent's own cases before a clean re-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time

from casegraph import config as C


def load_pack() -> list[dict]:
    with open(C.RAW / "case_pack.csv") as fh:
        return list(csv.DictReader(fh))


def cmd_run(a) -> None:
    from casegraph.agent.investigator import Investigator

    pack = load_pack()
    todo = sorted(pack, key=lambda c: c["opened_at"]) if a.all else [c for c in pack if c["case_id"] in a.cases]
    if not todo:
        sys.exit("no such case")
    inv = Investigator(persist=not a.no_persist, simulate=a.simulate)
    print(f"graph transport: {inv.g.transport} | llm: {'gemini ' + inv.llm.model if inv.llm.online else 'offline'}")
    out_dir = C.CASES_DIR if a.simulate == "evidence" else C.ROOT / "runs" / f"simulate-{a.simulate}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = C.RUNS_DIR / "traces"
    runs.mkdir(parents=True, exist_ok=True)
    for c in todo:
        t = time.time()
        ans = inv.investigate(c)
        (out_dir / f"{c['case_id']}.json").write_text(json.dumps(ans, indent=2))
        trace = {"case_id": c["case_id"], "events": inv.events, "facts": inv.facts,
                 "tool_calls": [vars(x) for x in inv.trace.calls], "fingerprint": inv.fp,
                 "evidence_weights": [{"claim": e.claim, "weight": e.weight, "group": e.group, "signal": e.signal}
                                      for e in inv.F.evidence]}
        (runs / f"{c['case_id']}.json").write_text(json.dumps(trace, indent=2, default=str))
        k = ans["case"]
        print(f"{c['case_id']}  {k['verdict']:10s} p={k['fraud_probability']:.2f} {k['pattern']:28s} "
              f"exp=${k['exposure_usd']:>9,.2f}  sar={ans['sar']['file']!s:5s} "
              f"final={[x['action'] for x in ans['next_best_actions']['final']]}  "
              f"calls={ans['tool_calls']} tok={ans['tokens']} {time.time() - t:.1f}s")
    inv.g.close()


def cmd_reset(a) -> None:
    """Forget the agent's own cases (InvestigationCase + CaseEvent). The bank's data is untouched."""
    from casegraph.graph.conn import conn

    c = conn()
    for vt in ("CaseEvent", "InvestigationCase"):
        print(vt, c.delVertices(vt))


def cmd_check(a) -> None:
    from casegraph.checks import check_all

    ok = check_all(C.CASES_DIR)
    sys.exit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser(prog="casegraph")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("cases", nargs="*")
    r.add_argument("--all", action="store_true")
    r.add_argument("--no-persist", action="store_true", help="do not write the case into the graph")
    r.add_argument("--simulate", default="evidence", choices=["evidence", "deny", "confirm", "none"])
    r.set_defaults(fn=cmd_run)
    z = sub.add_parser("reset-memory")
    z.set_defaults(fn=cmd_reset)
    k = sub.add_parser("check")
    k.set_defaults(fn=cmd_check)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
