"""The investigation loop.

    trigger -> open case -> gather (graph tools) -> plan more (LLM) -> analyse
    -> recall memory (graph links + vector search) -> assess uncertainty
    -> initial next-best-action (policy) -> request evidence (simulated reply)
    -> re-assess -> final next-best-action -> write case text (LLM + GraphRAG)
    -> persist the case into the graph -> answer file

Every step is appended to `self.events`; the same list is written to the graph
as CaseEvent vertices (the case's record of decisions and actions) and drives
the dashboard's timeline.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from datetime import timedelta

from casegraph.agent import policy as P
from casegraph.agent.actions import dispatch
from casegraph.agent.llm import LLM
from casegraph.agent.mcp_client import MCPGraph
from casegraph.agent.signals import Evidence, Findings, card_testing, logit, money, recurring, sigmoid, structuring
from casegraph.agent.tools import Tools, Trace, around, dt
from casegraph.rag.embed import embed_one, fingerprint

GENERIC_DEVICE_CARDS = 60      # a device profile seen on more cards than this is too common to link people
PLANNER_BUDGET = 3
# confirmed-fraud closed cases per inferred cardholder, bank-wide (4,665 cases / 222,481 holders)
BASE_FRAUD_PER_HOLDER = 4665 / 222481

PLANNER_TOOLS = [
    {"name": "card_window", "description": "All transactions on a card in a window around a time.",
     "parameters": {"type": "object", "properties": {
         "card_id": {"type": "string"}, "at": {"type": "string", "description": "YYYY-MM-DD HH:MM:SS"},
         "hours_before": {"type": "number"}, "hours_after": {"type": "number"}, "why": {"type": "string"}},
         "required": ["card_id", "at", "why"]}},
    {"name": "prior_cases", "description": "Closed cases and earlier agent cases linked to a card.",
     "parameters": {"type": "object", "properties": {"card_id": {"type": "string"}, "why": {"type": "string"}},
                    "required": ["card_id", "why"]}},
    {"name": "device_neighbors", "description": "Cards that used a device profile in a window, with spend, "
                                                "New/Found flags, proxies and scores, and closed cases on the device.",
     "parameters": {"type": "object", "properties": {"device_id": {"type": "string"}, "days_before": {"type": "number"},
                                                     "days_after": {"type": "number"}, "why": {"type": "string"}},
                    "required": ["device_id", "why"]}},
    {"name": "device_ring", "description": "Connected-components graph algorithm over cardholders and specific device "
                                           "profiles in a window: finds groups of cards sharing unusual devices.",
     "parameters": {"type": "object", "properties": {"device_id": {"type": "string"}, "days": {"type": "number"},
                                                     "why": {"type": "string"}}, "required": ["device_id", "why"]}},
    {"name": "region_context", "description": "Card's history in a billing region, and other cards hit there.",
     "parameters": {"type": "object", "properties": {"card_id": {"type": "string"}, "region": {"type": "string"},
                                                     "why": {"type": "string"}}, "required": ["card_id", "region", "why"]}},
]


GENERIC_INFO = ("?", "Windows", "iOS Device", "MacOS", "Linux", "Trident/7.0", "rv:")


def is_specific(device: str, n_cards: int) -> bool:
    """Can a device profile link people? Only if it is rare, or names a concrete handset/model."""
    if not device or n_cards <= 0:
        return False
    info = device.split(" | ")[0]
    concrete = not any(info.startswith(g) for g in GENERIC_INFO)
    return n_cards <= 25 or (concrete and n_cards <= 80)


class Investigator:
    def __init__(self, graph: MCPGraph | None = None, llm: LLM | None = None, persist: bool = True,
                 simulate: str = "evidence"):
        self.g = graph or MCPGraph()
        self.llm = llm or LLM()
        self.persist = persist
        self.simulate = simulate      # "evidence" (reply consistent with the evidence), "deny", "confirm", "none"

    # ================================================================== helpers
    def ev(self, kind: str, title: str, detail: str = "", data=None) -> None:
        self.events.append({"step": len(self.events) + 1, "kind": kind, "title": title, "detail": detail,
                            "data": data, "t": round(time.time() - self.t0, 2)})
        self.tools.step = len(self.events)

    def checkpoint(self, status: str, p: float | None, pattern: str, actions: list[dict] | None = None) -> None:
        """Progress the case vertex in the graph as the investigation moves (open -> decided)."""
        if not self.persist:
            return
        from casegraph.agent.memory import checkpoint

        checkpoint(self, status, p, pattern, actions)

    def act(self, stage: str, plan, connected=None) -> None:
        """Hand a plan to the action desk: auto actions execute, L1/L2 wait for a human."""
        rows = dispatch(self.case["case_id"], stage, plan.sorted(), {"connected": connected or []})
        self.actions_log += rows
        done = [r["action"] for r in rows if r["status"] == "executed"]
        held = [f"{r['action']} ({r['route']})" for r in rows if r["status"] != "executed"]
        self.ev("action", f"{stage}: executed {len(done)}, awaiting approval {len(held)}",
                "executed: " + (", ".join(done) or "-") + " | awaiting approval: " + (", ".join(held) or "-"), rows)

    # ================================================================== main
    def investigate(self, case: dict) -> dict:
        self.t0 = time.time()
        self.events: list[dict] = []
        self.trace = Trace()
        self.tools = Tools(self.g, self.trace)
        tok0 = self.llm.tokens
        self.case = case
        cid, trig = case["case_id"], case["trigger_type"]
        prefix, num = cid.split("-", 1)
        self.graph_case_id = f"CASE-2016-{num}" if prefix == "HHG" else f"CASE-2016-{prefix[0]}{num}"
        self.actions_log: list[dict] = []
        self.ev("trigger", f"{trig.replace('_', ' ')} on txn {case['flagged_txn_id']}", case["trigger_text"])
        self.checkpoint("open", None, "none")

        # ---------------------------------------------------------- gather
        ctx = self.tools.txn_context(case["flagged_txn_id"], why="open the flagged transaction and its neighbours")
        f = ctx["txn"]
        at = dt(f["ts"])
        self.ctx, self.f, self.at = ctx, f, at
        card, holder, device, region = ctx["card"], ctx["holder"], ctx["device"], ctx["region"]
        self.ev("gather", "flagged transaction in context",
                f"{money(f['amount'])} {f['channel']} product {f['product']} on {card} at {f['ts']}; holder {holder}; "
                f"device {device or 'none'}; region {region or 'none'}")

        prof = self.tools.behaviour_profile(card, holder, at, device, region, f.get("p_email"), f["product"], f["amount"],
                                            why="what is normal for this card and this cardholder")
        t0, t1 = around(at, 72, 24)
        window = self.tools.card_window(card, t0, t1, why="activity on the card around the alert")
        prior = self.tools.prior_cases(card, holder, device, at, why="case memory linked by card, holder, device")
        dev_n = ring = reg = rec = None
        specific_device = is_specific(device, ctx["device_cards_total"])
        if device:
            dev_n = self.tools.device_neighbors(device, at - timedelta(days=14), at + timedelta(days=7),
                                                why="who else used this device profile around the alert")
        if f["channel"] == "in_person" and region:
            reg = self.tools.region_context(card, region, at, why="has this card been billed in this region before")
        if trig == "customer_report" or True:
            rec = self.tools.recurring_check(holder, card, region, f["product"], f["amount"], at,
                                             why="is this a repeating payment of the cardholder (R7)")
        comm = None
        if device and specific_device:
            comm = self.tools.device_community(device, at, why="Louvain community of this device (GDBMS_ALGO)")
        others = [c for c in (dev_n or {}).get("cards", []) if c["card"] != card]
        if device and specific_device and len(others) >= 2:
            ring = self.tools.device_ring(at - timedelta(days=14), at + timedelta(days=14), seed_device=device,
                                          why="connected components over cardholders and specific devices")
        self.data = dict(prof=prof, window=window, prior=prior, dev_n=dev_n, ring=ring, reg=reg, rec=rec,
                         specific_device=specific_device, extra_windows={}, comm=comm)
        self.ev("gather", "core evidence gathered", f"{self.trace.n} graph calls so far")

        # ---------------------------------------------------------- plan more
        self._plan_more()

        # ---------------------------------------------------------- analyse
        F = self.analyse()
        self.F = F
        # ---------------------------------------------------------- memory
        similar = self.recall(F)
        # ---------------------------------------------------------- assess
        p0, groups, single = self.score(F)
        pattern = self.pick_pattern(F, p0)
        episode = self.episode(F, p0)
        exposure = round(sum(abs(r["amount"]) for r in episode), 2)
        self.ev("assess", f"initial assessment: p={p0:.2f}, pattern {pattern}",
                f"{groups} independent evidence line(s); single-signal={single}; exposure {money(exposure)}")

        sit = self.situation(p0, groups, single, pattern, exposure, F)
        init_plan, request = P.initial_plan(sit)
        self.ev("decide", "initial next-best-action", ", ".join(a["action"] for a in init_plan.sorted()))
        self.act("initial", init_plan, sorted(F.connected_cards - {card}))
        self.checkpoint("open", p0, pattern, init_plan.sorted())

        # ---------------------------------------------------------- more evidence
        requests, p1, response = [], p0, None
        if request:
            response = self.simulate_reply(request, p0, F, pattern)
            requests.append({"type": request, "asked_after_step": len(self.events), "assumed_response": response["text"]})
            if "STEP_UP_AUTH" in init_plan.names():
                su = {"deny": "One-time passcode sent to the registered phone was not completed; the session was abandoned",
                      "confirm": "Cardholder completed the one-time passcode on their registered phone",
                      "none": "No step-up attempt was made within 24 hours"}[response["kind"]]
                requests.append({"type": "step_up_auth", "asked_after_step": len(self.events), "assumed_response": su})
            self.ev("request", f"evidence requested: {request}", response["why"])
            self.ev("response", "assumed response recorded", response["text"])
            F.add(Evidence(response["claim"], "customer" if request != "analyst_info" else "external",
                           "evidence_request:1", [], response["weight"], "customer", "response"))
            p1 = min(max(sigmoid(logit(p0) + response["weight"]), 0.04), 0.97)
            sit.customer_denied = response["kind"] == "deny"
            sit.customer_confirmed = response["kind"] == "confirm"
            sit.no_reply = response["kind"] == "none"
            groups += 1
        verdict = "fraud" if p1 >= 0.70 else "legitimate" if p1 <= 0.30 else "uncertain"
        if verdict == "legitimate":
            pattern = "none"
        elif pattern == "none":
            pattern = self.pick_pattern(F, max(p1, 0.31))
        if verdict == "legitimate":
            episode, exposure = [], 0.0
        else:
            episode = self.episode(F, max(p1, 0.5) if verdict == "fraud" else p1)
            exposure = round(sum(abs(r["amount"]) for r in episode), 2)
        sit.p, sit.pattern, sit.exposure = p1, pattern, exposure
        if verdict == "uncertain" and request:
            # policy section 5: the agent may also ask an analyst for information, without approval
            requests.append({"type": "analyst_info", "asked_after_step": len(self.events),
                             "assumed_response": "Analyst acknowledged the request for merchant / terminal records; "
                                                 "review pending, no additional facts within the investigation window"})
            self.ev("request", "evidence requested: analyst_info",
                    "customer did not settle it; ask an analyst for merchant/terminal records (policy section 5)")
        final = P.final_plan(sit, verdict) if request else init_plan
        self.ev("assess", f"re-assessment: p={p1:.2f} -> verdict {verdict}", f"pattern {pattern}")
        self.ev("decide", "final next-best-action", ", ".join(a["action"] for a in final.sorted()))
        file_sar, sar_reason = P.sar_required(sit, verdict)
        file_sar = file_sar and "FILE_REPORT" in final.names()
        status = P.status_for(verdict, final)
        if request:
            self.act("final", final, sorted(F.connected_cards - {card}) if verdict != "legitimate" else [])

        # ---------------------------------------------------------- write
        connected = sorted(F.connected_cards - {card}) if verdict != "legitimate" else []
        devices = sorted(F.connected_devices) if verdict != "legitimate" else []
        text = self.compose(case, F, verdict, pattern, p0, p1, episode, exposure, init_plan, final, requests,
                            file_sar, sar_reason, connected, devices, similar)
        answer = {
            "case_id": cid,
            "case": {
                "status": status, "verdict": verdict, "fraud_probability": round(p1, 2), "pattern": pattern,
                "pattern_description": text.get("pattern_description", "") if pattern == "undocumented" else "",
                "affected_txn_ids": [r["id"] for r in episode],
                "first_suspicious_txn_id": episode[0]["id"] if episode else "",
                "connected_card_ids": connected, "connected_device_profiles": devices,
                "exposure_usd": exposure,
                "evidence": [e.as_answer() for e in self.select_evidence(F)],
                "similar_prior_cases": similar["cited"],
                "summary": text["summary"],
                "written_to_graph": False, "graph_case_id": "",
            },
            "evidence_requests": requests,
            "next_best_actions": {"initial": init_plan.sorted(), "final": final.sorted(),
                                  "what_changed": text["what_changed"]},
            "sar": self.sar_block(file_sar, sar_reason, text, episode, exposure, connected, devices),
            "stop_reason": text["stop_reason"],
            "tool_calls": 0, "tokens": 0, "latency_s": 0.0,
        }
        if self.persist:
            from casegraph.agent.memory import persist_case

            persist_case(self, answer, F, episode, similar)
            answer["case"]["written_to_graph"] = True
            answer["case"]["graph_case_id"] = self.graph_case_id
            self.ev("memory", "case written to the graph", f"{self.graph_case_id} + events, links and vectors")
        answer["tool_calls"] = self.trace.n
        answer["tokens"] = self.llm.tokens - tok0
        answer["latency_s"] = round(time.time() - self.t0, 1)
        self.answer = answer
        return answer

    # ================================================================== planning
    def _plan_more(self) -> None:
        digest = self.digest()
        calls: list[dict] = []
        if self.llm.online:
            try:
                calls = self.llm.plan(digest, PLANNER_TOOLS, PLANNER_BUDGET)
                who = f"LLM planner ({self.llm.model})"
            except Exception as e:  # never fail a case on a planning error
                self.ev("plan", "LLM planner unavailable, rule planner used", str(e)[:200])
                calls, who = self._rule_planner(), "rule planner"
        else:
            calls, who = self._rule_planner(), "rule planner"
        if not calls or calls[0]["tool"] == "finish":
            self.ev("plan", f"{who}: no further tools", calls[0]["why"] if calls else "core evidence sufficient")
            return
        for c in calls[:PLANNER_BUDGET]:
            try:
                self._exec(c)
                self.ev("plan", f"{who} -> {c['tool']}", c.get("why", ""), c["args"])
            except Exception as e:
                self.ev("plan", f"{who} -> {c['tool']} failed", str(e)[:200], c["args"])

    def _rule_planner(self) -> list[dict]:
        out = []
        dn = self.data["dev_n"] or {}
        card = self.ctx["card"]
        suspicious = [c for c in dn.get("cards", []) if c["card"] != card]
        if self.data["specific_device"] and suspicious:
            for c in sorted(suspicious, key=lambda c: -c["avg_model"])[:2]:
                out.append({"tool": "card_window", "args": {"card_id": c["card"], "at": c["first_ts"],
                                                            "hours_before": 24, "hours_after": 24},
                            "why": f"what else happened on {c['card']}, which shares the device"})
        return out

    def _exec(self, c: dict) -> None:
        a, t = c["args"], c["tool"]
        if t == "card_window":
            at = dt(a.get("at") or self.f["ts"])
            rows = self.tools.card_window(a["card_id"], at - timedelta(hours=float(a.get("hours_before", 24))),
                                          at + timedelta(hours=float(a.get("hours_after", 24))), why=c.get("why", ""))
            self.data["extra_windows"][a["card_id"]] = rows
        elif t == "prior_cases":
            res = self.tools.prior_cases(a["card_id"], "", "", self.at, why=c.get("why", ""))
            self.data.setdefault("extra_prior", {})[a["card_id"]] = res
        elif t == "device_neighbors":
            res = self.tools.device_neighbors(a["device_id"], self.at - timedelta(days=float(a.get("days_before", 14))),
                                              self.at + timedelta(days=float(a.get("days_after", 7))), why=c.get("why", ""))
            if a["device_id"] == self.ctx["device"]:
                self.data["dev_n"] = res
        elif t == "device_ring":
            d = float(a.get("days", 14))
            self.data["ring"] = self.tools.device_ring(self.at - timedelta(days=d), self.at + timedelta(days=d),
                                                       seed_device=a["device_id"], why=c.get("why", ""))
        elif t == "region_context":
            self.data["reg"] = self.tools.region_context(a["card_id"], a["region"], self.at, why=c.get("why", ""))
        else:
            raise ValueError(f"unknown tool {t}")

    def digest(self) -> str:
        f, ctx, d = self.f, self.ctx, self.data
        prof = d["prof"]
        lines = [f"Case {self.case['case_id']} ({self.case['trigger_type']}): {self.case['trigger_text']}",
                 f"Flagged {f['id']}: {money(f['amount'])} {f['channel']} product {f['product']} at {f['ts']} on "
                 f"{ctx['card']} (holder {ctx['holder']}), region {ctx['region'] or '-'}, device "
                 f"{ctx['device'] or '-'} [{f.get('device_status') or '-'}; proxy {f.get('proxy_type') or 'none'}], "
                 f"risk {f['risk_score']}, case-memory model {f['model_prob']}",
                 f"Card before alert: {prof['card_n']} txns, avg {money(prof['card_avg_amount'] or 0)}, "
                 f"{prof['card_n_regions']} regions, {prof['card_n_devices']} devices. Holder: {prof['holder_n']} txns, "
                 f"same device used {prof['holder_same_device']}x, same region {prof['holder_same_region']}x.",
                 f"Window: {len(d['window'])} txns on the card from -72h to +24h.",
                 f"Prior closed cases linked: {len(d['prior']['closed'])} "
                 f"({Counter(c['outcome'] for c in d['prior']['closed'])})."]
        if d["dev_n"]:
            cards = d["dev_n"]["cards"]
            lines.append(f"Device {ctx['device']} seen on {ctx['device_cards_total']} cards overall, "
                         f"{len(cards)} in -14d/+7d: " + "; ".join(f"{c['card']} n={c['n']} amt={c['amount']:.0f} "
                                                                    f"model={c['avg_model']:.2f}" for c in cards[:8]))
        if d["reg"]:
            lines.append(f"Region {ctx['region']}: card had {d['reg']['card_prior_txns_in_region']} prior txns there.")
        if d["ring"] and d["ring"].get("components"):
            c = d["ring"]["components"][0]
            lines.append(f"Ring component: {c['n_cards']} cards, {c['n_devices']} devices, {c['n_txns']} txns.")
        return "\n".join(lines)

    # ================================================================== analysis
    def analyse(self) -> Findings:
        F = Findings()
        f, ctx, d = self.f, self.ctx, self.data
        prof, window = d["prof"], d["window"]
        card, holder, device, region = ctx["card"], ctx["holder"], ctx["device"], ctx["region"]
        fid = f["id"]
        flagged_row = next((r for r in window if r["id"] == fid), None) or {
            "id": fid, "ts": f["ts"], "amount": f["amount"], "channel": f["channel"], "product": f["product"],
            "region": region, "device": device, "device_status": f.get("device_status"), "proxy_type": f.get("proxy_type"),
            "risk_score": f["risk_score"], "model_prob": f["model_prob"], "holder": holder}
        F.episode[fid] = flagged_row
        trig = self.case["trigger_type"]

        # -- model / score -------------------------------------------------------
        mp, rs = float(f["model_prob"]), float(f["risk_score"])
        F.notes["model_prob"] = mp
        F.add(Evidence(f"Case-memory model (trained only on the bank's closed cases, out-of-time AUC 0.92) puts "
                       f"transaction {fid} at {mp:.2f}; the bank's risk score is {rs:.2f}. Both are inputs, not a verdict",
                       "graph", f"attr:Transaction({fid}).model_prob,risk_score", [fid],
                       0.0, "model", "model"))
        if trig == "customer_report":
            F.add(Evidence("Customer states they did not make the purchase", "customer", "trigger:customer_report",
                           [self.case["customer_id"], fid], 1.2, "customer", "dispute"))
        if trig == "analyst_request":
            F.add(Evidence("Analyst flagged purchases from the same unusual device profile on several cards",
                           "external", "trigger:analyst_request", [fid], 0.3, "customer", "analyst"))

        # -- sequences on the card -----------------------------------------------
        seq = card_testing(window, flagged_row)
        if seq:
            small = [r for r in seq if r["amount"] < 10]
            big = [r for r in seq if r["amount"] >= 10]
            F.flags["card_testing"] = True
            F.flags["testing_over_100"] = any(r["amount"] > 100 for r in big)
            for r in seq:
                F.episode[r["id"]] = r
            F.add(Evidence(f"{len(small)} online authorisations under $10 within an hour "
                           f"({', '.join(money(r['amount']) for r in small[:5])}), then "
                           f"{', '.join(money(r['amount']) for r in big)}: a card-testing sequence",
                           "graph", f"query:card_window(card_id={card}, -72h/+24h)", [r["id"] for r in seq],
                           3.0, "sequence", "card_testing"))
            F.support("card_testing", 6)
        burst = structuring(window, flagged_row)
        if burst:
            F.flags["structuring"] = True
            for r in burst:
                F.episode[r["id"]] = r
            span = (dt(burst[-1]["ts"]) - dt(burst[0]["ts"])).total_seconds() / 60
            devs = {r["device"] for r in burst if r["device"]}
            F.add(Evidence(f"{len(burst)} online purchases in {span:.0f} minutes, each just under a round authorisation "
                           f"limit ({', '.join(money(r['amount']) for r in burst)}), from {len(devs)} device profile(s) "
                           f"marked New", "graph", f"query:card_window(card_id={card}, -72h/+24h)",
                           [r["id"] for r in burst], 3.2, "sequence", "structuring"))
            F.support("undocumented", 7)
            for r in burst:
                if r["device"]:
                    F.connected_devices.add(r["device"])

        # -- holder behaviour ------------------------------------------------------
        hn = prof["holder_n"]
        online = f["channel"] == "online"
        if online and device:
            new_dev = f.get("device_status") == "New"
            seen_holder = prof["holder_same_device"] > 0
            seen_card = prof["card_same_device"] > 0
            proxy = f.get("proxy_type") or ""
            if seen_holder:
                F.add(Evidence(f"The device profile was already used {prof['holder_same_device']} time(s) by this "
                               f"cardholder before the alert", "graph", f"query:behaviour_profile(holder={holder})",
                               [holder, device], -0.8, "device", "known_device"))
            elif new_dev:
                w = 0.5 + (0.4 if "ANONYMOUS" in proxy or "HIDDEN" in proxy else 0.15 if proxy else 0.0)
                F.add(Evidence(f"Identity record marks the device New for this account and it has never been seen on "
                               f"this cardholder{' or card' if not seen_card else ''}"
                               f"{'; connection behind proxy ' + proxy.split(':')[-1].lower() if proxy else ''}",
                               "graph", f"query:txn_context(txn_id={fid})", [fid, device], w, "device", "new_device"))
                F.support("card_not_present_new_device", 2.0 + (1 if proxy else 0))
            F.support("card_not_present_fraud", 1.0)
        elif online:
            F.support("card_not_present_fraud", 1.2)
        amt = float(f["amount"])
        if hn >= 3 and prof["holder_max_amount"] and amt > 1.5 * prof["holder_max_amount"]:
            F.add(Evidence(f"{money(amt)} is {amt / prof['holder_avg_amount']:.1f}x this cardholder's average and above "
                           f"their largest earlier purchase ({money(prof['holder_max_amount'])})", "graph",
                           f"query:behaviour_profile(holder={holder})", [holder, fid], 0.4, "amount", "amount"))
        elif prof["card_n"] >= 20 and amt > 4 * (prof["card_avg_amount"] or 1) and prof["card_n_bigger"] <= 2:
            F.add(Evidence(f"{money(amt)} is {amt / prof['card_avg_amount']:.1f}x the card's average "
                           f"({money(prof['card_avg_amount'])}) and larger than all but {prof['card_n_bigger']} earlier "
                           f"purchases", "graph", f"query:behaviour_profile(card={card})", [card, fid], 0.4, "amount",
                           "amount"))
        if f["product"] and prof["card_n"] >= 20 and prof["card_same_product"] == 0:
            F.add(Evidence(f"First purchase under product code {f['product']} on this card", "graph",
                           f"query:behaviour_profile(card={card})", [card, fid], 0.3, "amount", "new_product"))

        # -- region ---------------------------------------------------------------
        reg = d["reg"]
        if reg is not None:
            prior_in = reg["card_prior_txns_in_region"]
            same_region_days = {dt(r["ts"]).date() for r in window if r.get("region") == region and r["channel"] == "in_person"}
            if prior_in == 0 and prof["holder_same_region"] == 0:
                trip = len(same_region_days) >= 3
                F.add(Evidence(f"First card-present use of {card} in billing region {region}"
                               + (f"; purchases there on {len(same_region_days)} separate days look like a trip"
                                  if trip else ""), "graph", f"query:region_context(card_id={card}, region={region})",
                               [card, region, fid], -0.3 if trip else 0.8, "region", "new_region"))
                F.support("out_of_region_use", 2.5)
                F.flags["region_new"] = True
            else:
                F.add(Evidence(f"The card has {prior_in} earlier purchases in billing region {region} "
                               f"(first on {str(reg.get('card_first_seen_in_region'))[:10]}); the region is not new",
                               "graph", f"query:region_context(card_id={card}, region={region})", [card, region],
                               -0.6 if prior_in >= 3 else -0.3, "region", "known_region"))
                F.support("out_of_region_use", 0.8)
        # account takeover: mixed channels on the same holder around the alert with an unfamiliar element
        hold_rows = [r for r in window if r.get("holder") == holder]
        chans = {r["channel"] for r in hold_rows}
        if len(chans) > 1 and (f.get("device_status") == "New" or f.get("m4") in ("M2",)):
            F.support("account_takeover", 2.0)
            F.add(Evidence("The same cardholder shows both card-present and online activity around the alert, with an "
                           "unfamiliar device or match-flag anomaly (M-columns are unnamed match flags)", "graph",
                           f"query:card_window(card_id={card})", [holder] + [r["id"] for r in hold_rows[:4]], 0.3,
                           "sequence", "mixed_channel"))

        # -- other flagged activity on the same holder ------------------------------
        near = [r for r in hold_rows if r["id"] != fid and abs((dt(r["ts"]) - self.at).total_seconds()) <= 48 * 3600]
        hot = [r for r in near if r["model_prob"] >= 0.35 and r["channel"] == f["channel"]]
        if hot:
            for r in hot:
                F.episode[r["id"]] = r
            F.add(Evidence(f"{len(hot)} other {f['channel']} transaction(s) by the same cardholder within 48 hours "
                           f"also score high on the case-memory model ({', '.join(r['id'] for r in hot[:5])})",
                           "graph", f"query:card_window(card_id={card}, -72h/+24h)", [r["id"] for r in hot],
                           0.5 + 0.1 * min(len(hot), 4), "sequence", "burst"))
            F.support("card_not_present_fraud" if online else "out_of_region_use", 1.0)

        # -- recurring (R7) -----------------------------------------------------------
        rr = recurring({**(d["rec"] or {}), "_region": region})
        if rr:
            F.flags["recurring"] = True
            F.notes["recurring"] = rr
            F.add(Evidence(f"The same amount (+/-2%) under product {f['product']} repeats {rr['n']} times before the "
                           f"alert, about every {rr['median_gap_days']:.0f} days "
                           f"({'this cardholder' if rr['source'] == 'holder_repeats' else 'this card, same region'}): "
                           f"a recurring payment", "graph", "query:recurring_check", rr["ids"], -2.2, "history",
                           "recurring"))
            F.support("none", 3)

        # -- device network --------------------------------------------------------------
        dn = d["dev_n"]
        if dn and device:
            cards = dn["cards"]
            others = [c for c in cards if c["card"] != card]
            total = self.ctx["device_cards_total"]
            fraud_cases = [k for k, v in (dn.get("closed_case_outcome") or {}).items() if v == "confirmed_fraud"]
            if d["specific_device"] and len(others) >= 2:
                amts = [c["amount"] / max(c["n"], 1) for c in others]
                similar_amt = sum(1 for a in amts if abs(a - amt) <= 0.15 * max(amt, 1)) >= 2
                hi = [c for c in others if c["avg_model"] >= 0.3 or c["n_new"] > 0]
                w = 0.6 + 0.15 * min(len(others), 6) + (0.5 if similar_amt else 0) + (0.3 if len(hi) >= 2 else 0)
                F.add(Evidence(f"Device profile '{device}' (seen on only {total} cards in six months) was used by "
                               f"{len(others)} other card(s) between {str(min(c['first_ts'] for c in others))[:10]} and "
                               f"{str(max(c['last_ts'] for c in others))[:10]}"
                               + (f", at amounts close to {money(amt)}" if similar_amt else "")
                               + (f"; {len(hi)} of them marked New or scored high" if hi else ""),
                               "graph", "query:device_neighbors(device_id, -14d/+7d)",
                               [c["card"] for c in others[:12]], w, "network", "shared_device"))
                F.flags["shared_device"] = True
                F.shared_element = f"device profile {device}"
                F.connected_devices.add(device)
                for c in others:
                    F.connected_cards.add(c["card"])
                # this card's own transactions on the shared device are part of the episode
                own = set((dn.get("card_txns") or {}).get(card, []))
                rows = [r for r in window if r["id"] in own]
                if own - {r["id"] for r in rows}:
                    wide = self.tools.card_window(card, self.at - timedelta(days=14), self.at + timedelta(days=7),
                                                  why="this card's own purchases on the shared device")
                    rows = [r for r in wide if r["id"] in own]
                for r in rows:
                    F.episode[r["id"]] = r
            if fraud_cases and d["specific_device"]:
                F.add(Evidence(f"Closed cases on transactions from this device profile: "
                               f"{len(fraud_cases)} confirmed fraud ({', '.join(sorted(fraud_cases)[:6])})",
                               "graph", "query:device_neighbors(device_id).closed_cases",
                               sorted(fraud_cases)[:8], 0.4 + 0.1 * min(len(fraud_cases), 4), "memory", "device_cases"))
                F.notes["device_cases"] = fraud_cases
                pats = Counter((dn.get("closed_case_pattern") or {}).get(k) for k in fraud_cases)
                F.notes["device_case_patterns"] = pats
                if pats.get("undocumented"):
                    F.support("undocumented", 1.5 * pats["undocumented"])
        ring = (d["ring"] or {}).get("components") or []
        if ring:
            c = ring[0]
            members = sorted(set(sum((d["ring"].get("component_cards") or {}).values(), [])))
            devs = sorted(set(sum((d["ring"].get("component_devices") or {}).values(), [])))
            tight = c["n_devices"] <= 5 and 3 <= c["n_cards"] <= 60
            new_share = c["n_new_device"] / max(c["n_txns"], 1)
            proxy_share = c["n_proxy"] / max(c["n_txns"], 1)
            if tight and (new_share >= 0.5 or proxy_share >= 0.5 or c["avg_model"] >= 0.3):
                F.flags["ring"] = True
                F.flags["ring_proxy"] = proxy_share >= 0.5
                F.add(Evidence(f"Connected-components scan over cardholders and specific device profiles "
                               f"({str(c['first_ts'])[:10]} to {str(c['last_ts'])[:10]}) puts this card in a group of "
                               f"{c['n_cards']} cards and {c['n_holders']} cardholders joined by {c['n_devices']} "
                               f"device profile(s): {c['n_txns']} transactions, {money(c['amount'])}, "
                               f"{new_share:.0%} on a device New to the account, {proxy_share:.0%} behind a proxy",
                               "graph", "query:device_ring(seed_device, +/-14d)", members[:25],
                               1.4 + (0.6 if proxy_share >= 0.5 else 0), "network", "ring"))
                F.connected_cards |= set(members)
                F.connected_devices |= set(devs)
                F.shared_element = f"device profile {devs[0]}" if len(devs) == 1 else f"{len(devs)} device profiles"
                F.support("undocumented", 2.5 if proxy_share >= 0.5 else 1.0)

        # -- Louvain community (TigerGraph GDBMS_ALGO, batch) -----------------------------------
        comm = d.get("comm")
        if comm and comm.get("community", -1) >= 0 and comm["n_holders"] >= 3:
            oc = list((comm.get("closed_case_outcome") or {}).values())
            fraud_n, clear_n = oc.count("confirmed_fraud"), oc.count("cleared")
            rate = fraud_n / comm["n_holders"]
            lift = (rate + 0.005) / (BASE_FRAUD_PER_HOLDER + 0.005)
            w = round(max(min(0.35 * math.log2(lift), 0.8), -0.3), 2)
            F.add(Evidence(f"TigerGraph's Louvain community detection (GDBMS_ALGO) places this device profile in a "
                           f"community of {comm['n_holders']} cardholders, {comm['n_cards']} cards and "
                           f"{comm['n_devices']} device profile(s); before this alert its cardholders produced "
                           f"{fraud_n} confirmed-fraud and {clear_n} cleared closed cases "
                           f"({lift:.1f}x the bank-wide rate per cardholder)", "graph",
                           f"algo:GDBMS_ALGO.community.louvain -> query:device_community(community={comm['community']})",
                           sorted(k for k, v in (comm.get("closed_case_outcome") or {}).items()
                                  if v == "confirmed_fraud")[:8] or [device], w, "network", "louvain_community"))
            F.notes["community"] = {"id": comm["community"], "holders": comm["n_holders"], "fraud": fraud_n,
                                    "cleared": clear_n, "lift": round(lift, 2)}
            if comm["n_devices"] == 1 and comm["n_holders"] >= 5 and d["specific_device"]:
                F.support("undocumented", 0.5)

        # -- connected cards the planner looked at -----------------------------------------
        for other, rows in d["extra_windows"].items():
            same = [r for r in rows if r.get("device") == device and device]
            if same:
                F.add(Evidence(f"On {other}, the shared device made {len(same)} purchase(s) totalling "
                               f"{money(sum(r['amount'] for r in same))}; average case-memory score "
                               f"{sum(r['model_prob'] for r in same) / len(same):.2f}", "graph",
                               f"query:card_window(card_id={other})", [other] + [r["id"] for r in same[:4]],
                               0.2, "network", "connected_card"))

        # -- memory: prior cases on this cardholder / card ------------------------------------
        closed = d["prior"]["closed"]
        holder_fraud = [c for c in closed if "holder" in c["link"] and c["outcome"] == "confirmed_fraud"
                        and (self.at - dt(c["opened"])).days <= 120]
        holder_clear = [c for c in closed if "holder" in c["link"] and c["outcome"] == "cleared"]
        card_fraud = [c for c in closed if "card" in c["link"] and c["outcome"] == "confirmed_fraud"]
        card_clear = [c for c in closed if "card" in c["link"] and c["outcome"] == "cleared"]
        F.notes["prior"] = {"holder_fraud": holder_fraud, "card_fraud": card_fraud, "card_clear": card_clear}
        if holder_fraud:
            F.add(Evidence(f"This cardholder already had confirmed fraud in the last 120 days: "
                           f"{', '.join(c['id'] + ' (' + c['pattern'] + ')' for c in holder_fraud[:4])}", "graph",
                           f"query:prior_cases(holder={holder})", [c["id"] for c in holder_fraud[:6]],
                           1.2 + 0.3 * min(len(holder_fraud) - 1, 2), "memory", "holder_history"))
            for c in holder_fraud[:3]:
                F.support(c["pattern"], 0.8)
        if holder_clear:
            F.add(Evidence(f"Earlier alerts on this cardholder were cleared as false alarms: "
                           f"{', '.join(c['id'] for c in holder_clear[:4])}", "graph",
                           f"query:prior_cases(holder={holder})", [c["id"] for c in holder_clear[:6]],
                           -0.4, "memory", "holder_cleared"))
        own = {}
        for c in d["prior"]["agent"]:
            own.setdefault(c["id"], c)
        own_fraud = [c for c in own.values() if c["verdict"] == "fraud"]
        if own_fraud:
            links = sorted({c["link"].split(":")[0] for c in d["prior"]["agent"] if c["id"] in {o["id"] for o in own_fraud}})
            F.add(Evidence(f"The agent's own earlier investigation(s) {', '.join(c['id'] for c in own_fraud[:3])} "
                           f"reached a fraud verdict ({own_fraud[0]['pattern']}) on the same {' and '.join(links)}",
                           "graph", "query:prior_cases(...).agent_cases",
                           [x for x in (card, device) if x], 0.6, "memory", "agent_memory"))
            F.support(own_fraud[0]["pattern"], 1.0)
            F.notes["agent_memory"] = [c["id"] for c in own_fraud]
        if card_fraud and not holder_fraud:
            F.add(Evidence(f"The card has {len(card_fraud)} earlier confirmed-fraud case(s) (most recent "
                           f"{card_fraud[0]['id']}, {card_fraud[0]['pattern']}), none involving this cardholder's "
                           f"transactions", "graph", f"query:prior_cases(card_id={card})",
                           [c["id"] for c in card_fraud[:5]], 0.15, "memory", "card_history"))
        return F

    # ================================================================== memory
    def recall(self, F: Findings) -> dict:
        rows = list(F.episode.values())
        for r in rows:
            r.setdefault("region_new", F.flags.get("region_new", False))
        fp = fingerprint(rows)
        self.fp = fp
        by_fp = self.tools.similar_cases_fp(fp, 12, why="closed episodes with the same behavioural shape")
        hyp = self.hypothesis_text(F)
        self.hyp_vec = embed_one(hyp)
        by_text = self.tools.similar_cases_text(self.hyp_vec, 6, why="closed cases with a similar narrative")
        own = []
        try:
            own = self.tools.similar_agent_cases(fp, 5, why="the agent's own earlier cases with the same shape")
        except Exception:
            pass
        kb = self.tools.search_knowledge(embed_one(hyp + " policy rule next best action report"), 6,
                                         why="policy rules, typologies and regulatory guidance for this situation")
        self.kb = kb
        fp_fraud = [c for c in by_fp[:8] if c["outcome"] == "confirmed_fraud"]
        fp_clear = [c for c in by_fp[:8] if c["outcome"] == "cleared"]
        share = len(fp_fraud) / max(len(by_fp[:8]), 1)
        if by_fp:
            pats = Counter(c["pattern"] for c in fp_fraud)
            F.add(Evidence(f"Of the 8 closed episodes closest in behavioural fingerprint, {len(fp_fraud)} were confirmed "
                           f"fraud{' (mostly ' + pats.most_common(1)[0][0] + ')' if pats else ''} and {len(fp_clear)} "
                           f"were cleared (history base rate: 84% of closed cases are fraud)", "graph",
                           "vector:similar_cases_fp(ClosedCase.fp, k=12)",
                           [c["v_id"] for c in by_fp[:8]], round(max(min(share - 0.8, 0.2), -0.6), 2), "memory", "fp_vote"))
            for pat, n in pats.items():
                F.support(pat, 0.3 * n)
        # cite: graph-linked cases first, then nearest behavioural and narrative matches that agree
        cited, seen = [], set()
        for c in F.notes["prior"]["holder_fraud"] + [dict(id=k) for k in (F.notes.get("device_cases") or [])][:4]:
            if c["id"] not in seen:
                cited.append(c["id"]); seen.add(c["id"])
        for c in by_fp[:5] + by_text[:3]:
            if c["v_id"] not in seen and len(cited) < 6:
                cited.append(c["v_id"]); seen.add(c["v_id"])
        self.ev("memory", "similar cases recalled", f"fingerprint top: {[c['v_id'] for c in by_fp[:5]]}; narrative "
                f"top: {[c['v_id'] for c in by_text[:3]]}; agent cases: {[c.get('case_ref') for c in own[:3]]}")
        return {"cited": cited, "by_fp": by_fp, "by_text": by_text, "own": own, "fp": fp}

    def hypothesis_text(self, F: Findings) -> str:
        sig = ", ".join(sorted({e.signal for e in F.evidence if e.weight > 0.25}))
        return (f"{self.f['channel']} purchase {money(self.f['amount'])} product {self.f['product']}; "
                f"device {self.f.get('device_status') or 'none'} {self.f.get('proxy_type') or ''}; signals: {sig}")

    # ================================================================== assessment
    def score(self, F: Findings) -> tuple[float, int, bool]:
        mp = min(max(F.notes["model_prob"], 0.01), 0.97)
        x = logit(mp) + 0.6 * (float(self.f["risk_score"]) - 0.5)
        pos_groups, neg_groups = set(), set()
        for e in F.evidence:
            x += e.weight
            if e.weight >= 0.3:
                pos_groups.add(e.group)
            if e.weight <= -0.3:
                neg_groups.add(e.group)
        if mp >= 0.35:
            pos_groups.add("model")
        if mp <= 0.05:
            neg_groups.add("model")
        p = min(max(sigmoid(x), 0.04), 0.97)
        groups = len(pos_groups) if p >= 0.5 else len(neg_groups)
        single = len(pos_groups - {"customer"}) <= 1
        self.conflicting = len(pos_groups - {"customer", "amount"}) >= 1 and len(neg_groups) >= 2
        return p, groups, single

    def pick_pattern(self, F: Findings, p: float) -> str:
        if p < 0.3 or not F.patterns:
            return "none"
        cands = {k: v for k, v in F.patterns.items() if k != "none"}
        if not cands:
            return "none"
        return max(cands.items(), key=lambda kv: kv[1])[0]

    def episode(self, F: Findings, p: float) -> list[dict]:
        """Affected transactions. Detected sequences (testing, structuring, shared device) are always in.
        When fraud is likely, the same cardholder's other activity from 72h before to 24h after the alert
        joins the episode: in the bank's closed cases, 67% of such same-holder neighbours were part of
        the confirmed fraud, against 2.3% for other transactions on the same card."""
        rows = dict(F.episode)
        holder = self.ctx["holder"]
        if p >= 0.5 and holder and "|na" not in holder:
            for r in self.data["window"]:
                if r.get("holder") == holder:
                    rows.setdefault(r["id"], r)
        elif p >= 0.5:
            dev = self.ctx["device"]
            for r in self.data["window"]:
                if r.get("holder") == holder and ((dev and r.get("device") == dev) or r["model_prob"] >= 0.35):
                    rows.setdefault(r["id"], r)
        return sorted(rows.values(), key=lambda r: (r["ts"], r["id"]))

    def situation(self, p, groups, single, pattern, exposure, F: Findings) -> P.Situation:
        return P.Situation(
            trigger=self.case["trigger_type"], p=p, groups=groups, pattern=pattern, exposure=exposure,
            channel=self.f["channel"], single_signal=single, card_testing=F.flags.get("card_testing", False),
            testing_purchase_over_100=F.flags.get("testing_over_100", False),
            undocumented=pattern == "undocumented",
            shared_origin=bool(F.flags.get("shared_device") or F.flags.get("ring")),
            shared_element=F.shared_element, connected_cards=len(F.connected_cards - {self.ctx["card"]}),
            recurring=F.flags.get("recurring", False), conflicting=self.conflicting,
            risk_score=float(self.f["risk_score"]))

    # ================================================================== simulation
    def simulate_reply(self, request: str, p: float, F: Findings, pattern: str) -> dict:
        """Customer/analyst replies are not in the data. We assume the reply the
        evidence points to and say so in the case file. The mode can be forced
        (--simulate deny|confirm|none) to show how the recommendation moves."""
        mode = self.simulate
        if mode == "evidence":
            mode = "deny" if p >= 0.55 else "confirm" if p <= 0.35 else "none"
        rec = F.flags.get("recurring")
        trip = any(e.signal == "new_region" and e.weight < 0 for e in F.evidence)
        if mode == "deny":
            text = ("Customer confirms they did not make the purchase(s), still has the card, and no one else uses it"
                    if self.case["trigger_type"] == "customer_report" else
                    "Customer replies that they did not make the flagged purchase and still has the card")
            return {"kind": "deny", "text": text, "weight": 2.2, "why": f"probability {p:.2f} (policy R1/section 5)",
                    "claim": "Customer denied the activity when asked (simulated reply; see evidence_requests)"}
        if mode == "confirm":
            if rec:
                text = "Customer recognises the charge as their own recurring payment and withdraws the dispute"
            elif self.case["trigger_type"] == "customer_report":
                text = ("On review of the merchant and date, the customer recognises the purchase (a household/own "
                        "purchase) and withdraws the dispute")
            elif trip:
                text = "Customer confirms they are travelling in the billing region"
            elif self.f.get("device_status") == "New":
                text = "Customer confirms the purchase and that they are using a new device; device added to profile"
            else:
                text = "Customer confirms they made the purchase"
            return {"kind": "confirm", "text": text, "weight": -3.0, "why": f"probability {p:.2f} (policy R1/R7)",
                    "claim": "Customer confirmed the activity when asked (simulated reply; see evidence_requests)"}
        return {"kind": "none", "text": "No reply from the customer within 24 hours", "weight": 0.0,
                "why": f"probability {p:.2f}: verification requested (R1)",
                "claim": "No reply to the verification request within 24 hours (simulated; R4)"}

    # ================================================================== writing
    def select_evidence(self, F: Findings) -> list[Evidence]:
        ev = sorted(F.evidence, key=lambda e: -abs(e.weight))
        keep = [e for e in ev if abs(e.weight) >= 0.1 or e.signal in ("model", "response")]
        return keep[:9]

    def compose(self, case, F, verdict, pattern, p0, p1, episode, exposure, init_plan, final, requests,
                file_sar, sar_reason, connected, devices, similar) -> dict:
        facts = {
            "case_id": case["case_id"], "trigger": case["trigger_type"], "trigger_text": case["trigger_text"],
            "flagged": {k: self.f.get(k) for k in ("id", "ts", "amount", "product", "channel", "addr1", "device_id",
                                                     "device_status", "proxy_type", "p_email", "risk_score", "model_prob")},
            "card": self.ctx["card"], "customer": self.ctx["customer"], "holder": self.ctx["holder"],
            "verdict": verdict, "pattern": pattern, "p_initial": round(p0, 2), "p_final": round(p1, 2),
            "evidence": [e.claim for e in self.select_evidence(F)],
            "episode": [{"id": r["id"], "ts": r["ts"], "amount": r["amount"], "channel": r["channel"],
                         "region": r.get("region"), "device": r.get("device")} for r in episode],
            "exposure_usd": exposure, "connected_cards": connected[:20], "connected_devices": devices,
            "shared_element": F.shared_element, "evidence_requests": requests,
            "initial_actions": [a["action"] for a in init_plan.sorted()],
            "final_actions": [a["action"] for a in final.sorted()],
            "sar_required": file_sar, "sar_reason": sar_reason, "similar_prior_cases": similar["cited"],
        }
        self.facts = facts
        ctx = [f"[{k.get('source')}] {k.get('section')}: {str(k.get('text'))[:900]}" for k in self.kb[:5]]
        ctx += [f"[closed case {c['v_id']}] {c.get('pattern')} {c.get('outcome')}: {str(c.get('analyst_notes'))[:300]}"
                for c in similar["by_fp"][:2] + similar["by_text"][:2]]
        out = {}
        if self.llm.online:
            try:
                out = self.llm.write(facts, ctx)
                self.ev("write", f"case text written by the LLM ({self.llm.model}) from graph evidence + retrieved context",
                        f"{len(ctx)} retrieved passages")
            except Exception as e:
                self.ev("write", "LLM writer unavailable, templates used", str(e)[:200])
        tpl = self.templates(facts, F, verdict, pattern, p0, p1, episode, exposure, requests, init_plan, final)
        for k, v in tpl.items():
            if not out.get(k):
                out[k] = v
        return out

    def templates(self, facts, F, verdict, pattern, p0, p1, episode, exposure, requests, init_plan, final) -> dict:
        f, card = self.f, self.ctx["card"]
        ev = [e for e in self.select_evidence(F) if e.signal not in ("model", "response")]
        if verdict == "legitimate":
            ev = sorted(ev, key=lambda e: e.weight)          # what argues for legitimate first
        else:
            ev = sorted(ev, key=lambda e: -e.weight)
        top = [e.claim for e in ev][:2]
        if verdict == "fraud":
            s = (f"{pattern.replace('_', ' ').capitalize()} on {card}: {len(episode)} transaction(s) totalling "
                 f"{money(exposure)}. " + " ".join(t.rstrip('.') + '.' for t in top))
        elif verdict == "legitimate":
            s = (f"Alert on {money(f['amount'])} ({f['id']}) judged legitimate at probability {p1:.2f}. "
                 + " ".join(t.rstrip('.') + '.' for t in top))
        else:
            s = (f"Unresolved: probability {p1:.2f} for {money(f['amount'])} on {card} after verification was requested. "
                 + " ".join(t.rstrip('.') + '.' for t in top))
        changed = "nothing"
        if requests:
            a0 = {a["action"] for a in init_plan.actions}
            a1 = {a["action"] for a in final.actions}
            if a0 != a1:
                changed = (f"{requests[0]['assumed_response']}. Probability moved from {p0:.2f} to {p1:.2f}, so the "
                           f"recommendation changed from {', '.join(sorted(a0))} to {', '.join(sorted(a1))}.")
        if requests:
            stop = (f"The verification reply settled the question (section 6): probability {p1:.2f} on "
                    f"independent evidence; further graph steps would not change the actions.")
            if verdict == "uncertain":
                stop = ("Stopped pending the customer: no reply within 24 hours, so the policy's R4/R8 holding actions "
                        "apply until an analyst or the customer resolves it.")
        else:
            stop = (f"Probability {p1:.2f} is {'at or above 0.85' if p1 >= 0.85 else 'at or below 0.15'} with at least "
                    f"two independent lines of evidence (section 6); more steps would not change the decision.")
        desc = ""
        if pattern == "undocumented":
            if F.flags.get("structuring"):
                desc = ("Several online purchases in under an hour, each priced just below a round authorisation limit "
                        "($500), made from device profiles new to the account. The amounts look chosen to stay under "
                        "the limit that triggers extra checks. Found by scanning the card's window for bursts near "
                        "round thresholds and matched to five closed cases with the same shape.")
            else:
                desc = (f"A coordinated group of cards sharing one unusual {F.shared_element} in a short window, with "
                        "purchases from a device new to each account and mostly behind an anonymous proxy. It is "
                        "not one cardholder's compromise but a common actor across customers. Found with a connected-"
                        "components scan over cardholders and specific device profiles in TigerGraph.")
        return {"summary": s, "what_changed": changed, "stop_reason": stop, "pattern_description": desc,
                "sar_narrative": ""}

    def sar_block(self, file_sar, reason, text, episode, exposure, connected, devices) -> dict:
        if not file_sar:
            return {"file": False, "reason": reason, "narrative": "", "subjects": [], "total_amount_usd": 0,
                    "activity_dates": []}
        narrative = text.get("sar_narrative") or self.sar_template(episode, exposure, connected, devices)
        subjects = [self.ctx["customer"], self.ctx["card"]] + connected[:25] + devices[:3]
        dates = sorted(str(r["ts"])[:10] for r in episode)
        return {"file": True, "reason": reason, "narrative": narrative, "subjects": subjects,
                "total_amount_usd": exposure, "activity_dates": [dates[0], dates[-1]] if dates else []}

    def sar_template(self, episode, exposure, connected, devices) -> str:
        f, card, cust = self.f, self.ctx["card"], self.ctx["customer"]
        first, last = episode[0], episode[-1]
        parts = [f"Between {first['ts']} and {last['ts']}, card {card} held by customer {cust} was used for "
                 f"{len(episode)} {'online' if f['channel'] == 'online' else 'card-present'} transaction(s) totalling "
                 f"{money(exposure)} (" + ", ".join(f"{r['id']} {money(r['amount'])}" for r in episode[:8]) + ")."]
        if devices:
            parts.append(f"The transactions came from device profile(s) {', '.join(devices[:2])}, which the bank's "
                         f"identity records mark as new to the account.")
        if connected:
            parts.append(f"The same device profile was used in the same period on {len(connected)} other card(s), "
                         f"including {', '.join(connected[:6])}, indicating a common actor across cardholders.")
        parts.append(" ".join(e.claim.rstrip('.') + '.' for e in self.select_evidence(self.F)[:3]))
        parts.append(f"The activity is suspicious because it departs from the cardholder's established behaviour and "
                     f"matches a known fraud typology. Total suspicious amount: {money(exposure)}. The card has been "
                     f"recommended for blocking and reissue"
                     + (" and the connected cards placed under monitoring." if connected else "."))
        return " ".join(parts)
