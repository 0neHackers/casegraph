"""Fraud Policy v1.0 as code.

The LLM never picks action identifiers or approval routes on its own. It
reasons over evidence; this module turns the assessment into the policy's
actions, routes, report decision and stopping decision, and cites the rule
for each one. That keeps every recommendation inside the bank's permissions:
the agent may only *execute* `auto` actions, `L1`/`L2` actions are recommended
and wait for a human.
"""
from __future__ import annotations

from dataclasses import dataclass, field

AUTO = {"ALLOW_TRANSACTION", "MONITOR_CARD", "MONITOR_CONNECTED_CARDS", "WARN_CUSTOMER", "VERIFY_WITH_CUSTOMER",
        "STEP_UP_AUTH", "GENERATE_REPORT", "CREATE_CASE", "ESCALATE_TO_ANALYST", "CLOSE_NO_FRAUD"}
ALL_ACTIONS = AUTO | {"DECLINE_TRANSACTION", "BLOCK_CARD", "BLOCK_ALL_CARDS", "FILE_REPORT"}
ORDER = ["DECLINE_TRANSACTION", "STEP_UP_AUTH", "VERIFY_WITH_CUSTOMER", "BLOCK_CARD", "BLOCK_ALL_CARDS", "ALLOW_TRANSACTION",
         "CREATE_CASE", "MONITOR_CARD", "MONITOR_CONNECTED_CARDS", "WARN_CUSTOMER", "FILE_REPORT", "ESCALATE_TO_ANALYST",
         "GENERATE_REPORT", "CLOSE_NO_FRAUD"]

DECISIVE_HI, DECISIVE_LO = 0.85, 0.15      # policy section 6
CASE_THRESHOLD = 0.30                      # section 3a
R1_THRESHOLD = 0.70


def route(action: str, exposure: float) -> str:
    if action in AUTO:
        return "auto"
    if action == "DECLINE_TRANSACTION":
        return "L1"
    if action == "BLOCK_CARD":
        return "L1" if exposure <= 2500 else "L2"
    return "L2"   # BLOCK_ALL_CARDS, FILE_REPORT


@dataclass
class Situation:
    trigger: str                       # risk_score | customer_report | analyst_request
    p: float                           # current fraud probability
    groups: int                        # independent evidence groups behind the assessment
    pattern: str
    exposure: float
    channel: str
    single_signal: bool = False
    card_testing: bool = False
    testing_purchase_over_100: bool = False
    undocumented: bool = False
    shared_origin: bool = False        # shared device / region cluster / another customer's fraud
    shared_element: str = ""
    connected_cards: int = 0
    recurring: bool = False
    customer_denied: bool = False
    customer_confirmed: bool = False
    no_reply: bool = False
    conflicting: bool = False
    compromised_cards: int = 1
    risk_score: float = 0.0
    credentials_compromised: bool = False


@dataclass
class Plan:
    actions: list[dict] = field(default_factory=list)

    def add(self, action: str, reason: str, exposure: float) -> None:
        assert action in ALL_ACTIONS, action
        if any(a["action"] == action for a in self.actions):
            return
        self.actions.append({"action": action, "route": route(action, exposure), "reason": reason})

    def sorted(self) -> list[dict]:
        return sorted(self.actions, key=lambda a: ORDER.index(a["action"]))

    def names(self) -> set[str]:
        return {a["action"] for a in self.actions}


def sar_required(s: Situation, verdict: str) -> tuple[bool, str]:
    """Section 3a: fraud confirmed or strongly suspected AND one of the triggers."""
    if verdict != "fraud" and s.p < R1_THRESHOLD:
        return False, "No report: fraud is not confirmed or strongly suspected (section 3a)."
    why = []
    if s.exposure > 1000:
        why.append(f"exposure ${s.exposure:,.2f} exceeds $1,000")
    if s.shared_origin:
        why.append(f"the activity connects to a shared {s.shared_element or 'origin'} and other cards")
    if s.undocumented:
        why.append("the pattern is coordinated / undocumented (R9)")
    if not why:
        return False, (f"Case only, no report: exposure ${s.exposure:,.2f} is under $1,000, no shared device, region "
                       "cluster or link to another customer's fraud, and the pattern is documented (section 3a).")
    return True, "File: " + "; ".join(why) + " (section 3a" + (", R6" if s.shared_origin else "") + \
        (", R9" if s.undocumented else "") + ", R2)."


def fraud_actions(s: Situation, plan: Plan, first: str) -> None:
    e = s.exposure
    if s.card_testing and not s.testing_purchase_over_100 and not s.customer_denied:
        plan.add("DECLINE_TRANSACTION", "R5: card-testing sequence; decline the pending authorisation", e)
        plan.add("STEP_UP_AUTH", "R5: require step-up before any further activity", e)
    else:
        why = first
        if s.card_testing and s.testing_purchase_over_100:
            why = "R5: testing sequence and a purchase over $100 has already cleared"
        plan.add("BLOCK_CARD", f"{why}; exposure ${e:,.2f} {'<=' if e <= 2500 else '>'} $2,500 so route "
                               f"{route('BLOCK_CARD', e)}", e)
    if s.compromised_cards >= 2 or s.credentials_compromised:
        plan.add("BLOCK_ALL_CARDS", "R10: at least two of the customer's cards show confirmed fraud", e)
    plan.add("CREATE_CASE", "Section 3a / R2: fraud probability above 0.30; open the internal case and write it to the graph", e)
    file, why = sar_required(s, "fraud")
    if file:
        plan.add("FILE_REPORT", why, e)
    if s.connected_cards:
        plan.add("MONITOR_CONNECTED_CARDS", f"R6: {s.connected_cards} other card(s) share the {s.shared_element or 'origin'}", e)
    if s.undocumented:
        plan.add("ESCALATE_TO_ANALYST", "R9: coordinated / undocumented pattern; hand to an analyst with the evidence", e)


def legit_actions(s: Situation, plan: Plan, why: str) -> None:
    e = s.exposure
    if s.trigger == "customer_report":
        plan.add("CREATE_CASE", "Section 3a: the customer disputed a charge, so a case records the outcome", e)
    if s.recurring:
        plan.add("WARN_CUSTOMER", "R7: remind the customer of the recurring charge they set up", e)
    elif s.trigger != "customer_report":
        plan.add("ALLOW_TRANSACTION", why, e)
    plan.add("CLOSE_NO_FRAUD", why, e)


def uncertain_actions(s: Situation, plan: Plan) -> None:
    e = s.exposure
    plan.add("CREATE_CASE", "Section 3a: fraud probability reaches 0.30 / evidence was requested", e)
    if s.no_reply:
        plan.add("MONITOR_CARD", "R4: no reply within 24 hours; raise monitoring for 72 hours", e)
        plan.add("DECLINE_TRANSACTION", "R4: decline pending authorisations while unverified", e)
    else:
        plan.add("MONITOR_CARD", "R8: verdict uncertain; monitor while the case is open", e)
    if e > 500 or s.conflicting:
        plan.add("ESCALATE_TO_ANALYST", f"R8{' / R4' if s.no_reply else ''}: uncertain with exposure ${e:,.2f} "
                                        f"{'> $500' if e > 500 else ''}{' and conflicting evidence' if s.conflicting else ''}", e)


def initial_plan(s: Situation) -> tuple[Plan, str | None]:
    """What to recommend before any requested evidence returns.
    Returns the plan and the evidence request to make (None = decide now)."""
    plan, e = Plan(), s.exposure
    decisive_hi = s.p >= DECISIVE_HI and s.groups >= 2
    decisive_lo = s.p <= DECISIVE_LO and s.groups >= 2

    if s.recurring and s.trigger == "customer_report":
        plan.add("CREATE_CASE", "R7 / section 3a: the customer disputes a charge that matches their own recurring pattern", e)
        plan.add("VERIFY_WITH_CUSTOMER", "R7: confirm with the customer that this is their repeating payment; do not block", e)
        plan.add("WARN_CUSTOMER", "R7: send a recurring-charge reminder", e)
        return plan, "customer_validation"
    if decisive_hi and (s.trigger != "risk_score" or not s.single_signal):
        fraud_actions(s, plan, "Probability >= 0.85 on independent evidence (section 6)")
        return plan, None
    if decisive_lo:
        if s.trigger == "risk_score" and not (s.p <= 0.05 and s.risk_score < 0.70):
            # Case memory: every one of the 900 cleared alerts in the bank's history was closed only after the
            # cardholder confirmed it. A strong model alert is contradicted, not dismissed, so we verify first.
            plan.add("CREATE_CASE", "Section 3a: evidence is being requested, which opens a case", e)
            plan.add("VERIFY_WITH_CUSTOMER", f"R1: the alert rests on the risk score alone ({s.risk_score:.2f}) and the "
                                             f"graph evidence contradicts it (p={s.p:.2f}); confirm with the cardholder, "
                                             "as the bank did for every cleared alert in its history, then close", e)
            return plan, "customer_validation"
        if s.trigger == "customer_report":
            plan.add("CREATE_CASE", "Section 3a: every customer dispute opens a case", e)
            plan.add("VERIFY_WITH_CUSTOMER", "R1: the graph contradicts the dispute; check the details with the customer "
                                             "before closing", e)
            return plan, "customer_validation"
        legit_actions(s, plan, "Probability <= 0.15 on two independent pieces of evidence (section 6); "
                               "risk score alone is not a verdict (R1)")
        return plan, None

    # uncertain zone: gather evidence through policy-approved controls
    if s.card_testing:
        fraud_actions(s, plan, "R5")
        plan.add("VERIFY_WITH_CUSTOMER", "R5 / R1: confirm the testing sequence with the cardholder", e)
        return plan, "customer_validation"
    if s.p >= CASE_THRESHOLD or s.trigger == "customer_report":
        plan.add("CREATE_CASE", "Section 3a: fraud probability >= 0.30 or evidence requested", e)
    else:
        plan.add("CREATE_CASE", "Section 3a: evidence is being requested, which opens a case", e)
    if s.trigger == "customer_report":
        plan.add("MONITOR_CARD", "R1: keep the card active but watched while the dispute is verified", e)
        plan.add("VERIFY_WITH_CUSTOMER", "R2 precondition: confirm the unauthorised use (card in possession, no household "
                                         "use) before blocking", e)
        return plan, "customer_validation"
    if s.p >= R1_THRESHOLD and not s.single_signal:
        plan.add("DECLINE_TRANSACTION", f"Probability {s.p:.2f} on several signals: hold the authorisation until verified", e)
    plan.add("VERIFY_WITH_CUSTOMER", f"R1: probability {s.p:.2f}{' rests on a single signal' if s.single_signal else ''}; "
                                     "verify before any block", e)
    if s.channel == "online":
        plan.add("STEP_UP_AUTH", "R1: require a one-time passcode on further online activity", e)
    plan.add("MONITOR_CARD", "Keep the card active under raised monitoring while verification is pending", e)
    return plan, "customer_validation"


def final_plan(s: Situation, verdict: str) -> Plan:
    plan, e = Plan(), s.exposure
    if verdict == "fraud":
        first = "R2: customer denies the transaction" if s.customer_denied else "Fraud established on independent evidence"
        fraud_actions(s, plan, first)
    elif verdict == "legitimate":
        why = ("R3: customer confirmed the transaction" if s.customer_confirmed else
               "Probability <= 0.15 on independent evidence (section 6)")
        if s.recurring:
            why = "R7 then R3: the customer recognised their own recurring payment"
        legit_actions(s, plan, why)
    else:
        uncertain_actions(s, plan)
    return plan


def status_for(verdict: str, plan: Plan) -> str:
    if verdict == "fraud":
        return "closed_fraud"
    if verdict == "legitimate":
        return "closed_legitimate"
    return "escalated" if "ESCALATE_TO_ANALYST" in plan.names() else "open"
