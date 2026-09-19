"""Print the markdown results table for the README from cases/*.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
print("| Case | Trigger | Verdict | p | Pattern | Exposure | Initial → final action | SAR |")
print("|---|---|---|---:|---|---:|---|:-:|")
for f in sorted((ROOT / "cases").glob("HHG-*.json")):
    a = json.loads(f.read_text())
    k, n = a["case"], a["next_best_actions"]
    trig = {"risk_score": "risk score", "customer_report": "customer", "analyst_request": "analyst"}
    pack = {r.split(",")[0]: r.split(",")[2] for r in (ROOT / "data/raw/case_pack.csv").read_text().splitlines()[1:]}
    ini = ", ".join(x["action"] for x in n["initial"] if x["action"] not in ("CREATE_CASE",))
    fin = ", ".join(x["action"] for x in n["final"] if x["action"] not in ("CREATE_CASE",))
    arrow = f"`{ini}`" if ini == fin else f"`{ini}` → `{fin}`"
    print(f"| {a['case_id']} | {trig.get(pack.get(a['case_id']), '')} | {k['verdict']} | {k['fraud_probability']:.2f} | "
          f"{k['pattern'].replace('_', ' ')} | ${k['exposure_usd']:,.2f} | {arrow} | {'✅' if a['sar']['file'] else '—'} |")
