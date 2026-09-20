"""The action desk: where permissions are enforced.

The policy says the agent may recommend anything, but may only *execute* `auto`
actions; `L1` (team lead) and `L2` (fraud manager) actions wait for a human. This
module is that boundary. `auto` actions are sent to the (mock) bank systems that own
them and come back with a reference; everything else is put on an approval queue
with the approver's role. Nothing here can execute an L1/L2 action: there is no code
path for it.

The systems are simulated, as the task allows ("sending customer messages, freezing
accounts, blocking cards ... may be simulated, stubbed, or represented through mock
APIs"). The references are deterministic so re-runs produce the same log.
"""
from __future__ import annotations

import hashlib

APPROVER = {"L1": "team lead (L1)", "L2": "fraud manager (L2)"}

SYSTEMS = {
    "ALLOW_TRANSACTION": ("authorisation-service", "authorisation released"),
    "MONITOR_CARD": ("risk-engine", "card sensitivity raised for 72 hours"),
    "MONITOR_CONNECTED_CARDS": ("risk-engine", "connected cards placed under monitoring"),
    "WARN_CUSTOMER": ("notification-service", "informational message sent to the cardholder"),
    "VERIFY_WITH_CUSTOMER": ("notification-service", "verification request sent (app push + SMS)"),
    "STEP_UP_AUTH": ("auth-service", "one-time-passcode challenge armed on the card"),
    "GENERATE_REPORT": ("case-system", "internal investigation report generated"),
    "CREATE_CASE": ("case-system", "case opened and written to TigerGraph"),
    "ESCALATE_TO_ANALYST": ("analyst-queue", "case handed to the fraud-analyst queue with evidence"),
    "CLOSE_NO_FRAUD": ("case-system", "alert closed as legitimate"),
}


def _ref(prefix: str, case_id: str, stage: str, action: str) -> str:
    return prefix + "-" + hashlib.sha1(f"{case_id}|{stage}|{action}".encode()).hexdigest()[:8]


def dispatch(case_id: str, stage: str, actions: list[dict], extra: dict | None = None) -> list[dict]:
    """Execute the auto actions, queue the rest. Returns one log row per action."""
    log = []
    for a in actions:
        if a["route"] == "auto":
            system, what = SYSTEMS[a["action"]]
            if a["action"] == "MONITOR_CONNECTED_CARDS" and extra and extra.get("connected"):
                what = f"{len(extra['connected'])} connected card(s) placed under monitoring"
            log.append({"stage": stage, "action": a["action"], "route": "auto", "status": "executed",
                        "system": system, "detail": what, "ref": _ref(system.split("-")[0].upper(), case_id, stage, a["action"])})
        else:
            log.append({"stage": stage, "action": a["action"], "route": a["route"], "status": "pending_approval",
                        "system": "approval-queue", "detail": f"awaiting {APPROVER[a['route']]}",
                        "ref": _ref("APR", case_id, stage, a["action"])})
    return log
