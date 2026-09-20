"""CaseGraph analyst console.

    streamlit run casegraph/ui/app.py

Left: the case queue (the 20 exam alerts). Right: one case, the way an analyst
reads it: trigger and verdict, the investigation timeline, the uncertainty and
how it moved, the next best action before and after the evidence request with
approval routes, the evidence table, the connected graph, the SAR, and the raw
tool calls the agent made through TigerGraph MCP.

"Investigate live" re-runs the agent on the selected alert against TigerGraph.
The reply selector forces the simulated customer response so you can watch the
recommendation change (policy section 3b).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from casegraph import config as C  # noqa: E402

st.set_page_config(page_title="CaseGraph · fraud investigation", page_icon="🕸️", layout="wide")

CSS = """
<style>
.block-container{padding-top:2.6rem;max-width:1500px}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.78rem;font-weight:600;margin-right:6px}
.fraud{background:#ffe1e1;color:#a40000}.legitimate{background:#dcf7e3;color:#0b6b2d}.uncertain{background:#fff2cc;color:#8a5a00}
.auto{background:#e6f0ff;color:#1846a3}.L1{background:#fff0e0;color:#9a4d00}.L2{background:#ffe1e1;color:#a40000}
.kpi{border:1px solid rgba(128,128,128,.25);border-radius:10px;padding:10px 14px}
.kpi b{font-size:1.35rem}
.ev{border-left:3px solid rgba(128,128,128,.35);padding:4px 10px;margin:4px 0;font-size:.9rem}
.small{font-size:.8rem;opacity:.75}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- data
@st.cache_data(ttl=5)
def load_answers(folder: str) -> dict[str, dict]:
    out = {}
    for f in sorted(Path(folder).glob("HHG-*.json")):
        out[f.stem] = json.loads(f.read_text())
    return out


@st.cache_data(ttl=5)
def load_trace(case_id: str) -> dict | None:
    f = C.RUNS_DIR / "traces" / f"{case_id}.json"
    return json.loads(f.read_text()) if f.exists() else None


@st.cache_data
def load_pack() -> pd.DataFrame:
    return pd.read_csv(C.RAW / "case_pack.csv", dtype=str)


def esc(text: str) -> str:
    """Streamlit markdown treats $...$ as LaTeX; amounts are full of dollars."""
    return str(text).replace("$", "\\$")


def pill(text: str, cls: str) -> str:
    return f"<span class='pill {cls}'>{text}</span>"


def actions_table(actions: list[dict]) -> str:
    rows = "".join(f"<tr><td><code>{a['action']}</code></td><td>{pill(a['route'], a['route'])}</td>"
                   f"<td class='small'>{a['reason']}</td></tr>" for a in actions)
    return f"<table style='width:100%'>{rows}</table>".replace("$", "&#36;")


def graph_html(ans: dict, trace: dict | None) -> str:
    from pyvis.network import Network

    k = ans["case"]
    facts = (trace or {}).get("facts", {})
    net = Network(height="460px", width="100%", bgcolor="#ffffff", font_color="#444", directed=False)
    net.set_options(json.dumps({
        "physics": {"solver": "forceAtlas2Based",
                    "forceAtlas2Based": {"gravitationalConstant": -60, "springLength": 90, "centralGravity": 0.02},
                    "stabilization": {"enabled": True, "iterations": 600, "fit": True}},
        "interaction": {"hover": True}, "nodes": {"font": {"size": 12}}, "edges": {"color": {"opacity": 0.5}}}))
    flagged = facts.get("flagged", {})
    card, cust, holder = facts.get("card"), facts.get("customer"), facts.get("holder")
    col = {"txn": "#ff6b6b", "card": "#4c8bf5", "cust": "#7b61ff", "holder": "#20c997", "dev": "#f59f00",
           "cc": "#adb5bd", "conn": "#91a7ff"}
    if cust:
        net.add_node(cust, label=cust, color=col["cust"], shape="dot", size=18, title="Customer")
    if card:
        net.add_node(card, label=card, color=col["card"], shape="dot", size=22, title="Card")
        if cust:
            net.add_edge(cust, card, title="OWNS")
    if holder:
        net.add_node(holder, label="holder\n" + holder.split("|", 1)[-1], color=col["holder"], size=14, title=holder)
        net.add_edge(card, holder, title="HAS_HOLDER")
    ep = facts.get("episode") or []
    ids = {r["id"] for r in ep} | ({flagged.get("id")} if flagged.get("id") else set())
    for r in ep or [flagged]:
        if not r:
            continue
        rid = r["id"]
        net.add_node(rid, label=f"{rid}\n${r.get('amount', 0):,.2f}", color=col["txn"], shape="box",
                     title=f"{r.get('ts', '')} {r.get('channel', '')}")
        net.add_edge(card, rid, title="MADE")
        dev = r.get("device") or (r.get("device_id") if rid == flagged.get("id") else None)
        if dev:
            net.add_node(dev, label=dev.split(" | ")[0][:22], color=col["dev"], shape="diamond", size=16, title=dev)
            net.add_edge(rid, dev, title="FROM_DEVICE")
    if flagged.get("id") and flagged["id"] not in ids and card:
        net.add_node(flagged["id"], label=flagged["id"], color=col["txn"], shape="box")
        net.add_edge(card, flagged["id"])
    for dv in k["connected_device_profiles"]:
        net.add_node(dv, label=dv.split(" | ")[0][:22], color=col["dev"], shape="diamond", size=18, title=dv)
    for c in k["connected_card_ids"][:30]:
        net.add_node(c, label=c, color=col["conn"], size=10, title="connected card")
        for dv in k["connected_device_profiles"][:1]:
            net.add_edge(c, dv, title="shares device")
    for cc in k["similar_prior_cases"][:6]:
        net.add_node(cc, label=cc, color=col["cc"], shape="square", size=10, title="closed case cited as memory")
        net.add_edge(card, cc, title="CITES", dashes=True)
    return net.generate_html()


# --------------------------------------------------------------------------- sidebar
pack = load_pack()
answers = load_answers(str(C.CASES_DIR))
st.sidebar.markdown("## 🕸️ CaseGraph")
st.sidebar.caption("Agentic fraud investigation on TigerGraph · MCP · GraphRAG")
rows = []
for _, r in pack.iterrows():
    a = answers.get(r.case_id, {})
    k = a.get("case", {})
    rows.append({"case": r.case_id, "trigger": r.trigger_type.replace("_", " "), "verdict": k.get("verdict", "-"),
                 "p": k.get("fraud_probability"), "pattern": k.get("pattern", "-")})
queue = pd.DataFrame(rows)
flt = st.sidebar.multiselect("Verdict", ["fraud", "legitimate", "uncertain"], default=[])
view = queue[queue.verdict.isin(flt)] if flt else queue
case_id = st.sidebar.radio("Case queue", view.case.tolist(),
                           format_func=lambda c: f"{c} · {queue.set_index('case').loc[c, 'verdict']}")
st.sidebar.divider()
st.sidebar.markdown("**Investigate live**")
mode = st.sidebar.selectbox("Assumed reply to evidence requests", ["evidence", "deny", "confirm", "none"],
                            help="'evidence' = the reply the evidence points to (what the answer files use). "
                                 "Force another reply to watch the next best action change.")
persist = st.sidebar.checkbox("Write case to the graph", value=False)
go = st.sidebar.button("▶ Run agent on this alert", use_container_width=True)

# --------------------------------------------------------------------------- live run
if go:
    from casegraph.agent.investigator import Investigator

    case_row = pack[pack.case_id == case_id].iloc[0].to_dict()
    with st.status(f"Investigating {case_id} …", expanded=True) as status:
        inv = Investigator(persist=persist, simulate=mode)
        st.write(f"graph transport **{inv.g.transport}** · LLM **{inv.llm.model if inv.llm.online else 'offline'}**")
        ans = inv.investigate(case_row)
        for e in inv.events:
            st.write(f"`{e['step']:02d}` **{e['kind']}** · {e['title']}")
        status.update(label=f"{case_id}: {ans['case']['verdict']} (p={ans['case']['fraud_probability']})",
                      state="complete")
        inv.g.close()
    st.session_state["live"] = {"answer": ans, "trace": {"events": inv.events, "facts": inv.facts, "actions": inv.actions_log,
                                                          "tool_calls": [vars(x) for x in inv.trace.calls],
                                                          "evidence_weights": [{"claim": e.claim, "weight": e.weight,
                                                                                "group": e.group, "signal": e.signal}
                                                                               for e in inv.F.evidence]},
                                "case_id": case_id, "mode": mode}

live = st.session_state.get("live")
if live and live["case_id"] == case_id:
    ans, trace = live["answer"], live["trace"]
    st.info(f"Showing the live run (assumed reply: {live['mode']}). The saved answer file is unchanged.")
else:
    ans, trace = answers.get(case_id), load_trace(case_id)
if not ans:
    st.warning("No answer file yet. Run `python -m casegraph run --all` or use the button.")
    st.stop()

k, nba, sar = ans["case"], ans["next_best_actions"], ans["sar"]
row = pack[pack.case_id == case_id].iloc[0]

# --------------------------------------------------------------------------- header
st.markdown(f"### {case_id} · {row.trigger_type.replace('_', ' ')} "
            f"{pill(k['verdict'], k['verdict'])}{pill(k['status'].replace('_', ' '), 'auto')}"
            f"{pill(k['pattern'].replace('_', ' '), 'L1' if k['pattern'] != 'none' else 'legitimate')}",
            unsafe_allow_html=True)
st.caption(row.trigger_text)
c1, c2, c3, c4, c5 = st.columns(5)
facts = (trace or {}).get("facts", {})
c1.markdown(f"<div class='kpi'>fraud probability<br><b>{facts.get('p_initial', '?')} → {k['fraud_probability']}</b></div>",
            unsafe_allow_html=True)
c2.markdown(f"<div class='kpi'>exposure<br><b>${k['exposure_usd']:,.2f}</b></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='kpi'>affected txns / connected cards<br><b>{len(k['affected_txn_ids'])} / "
            f"{len(k['connected_card_ids'])}</b></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='kpi'>SAR<br><b>{'file' if sar['file'] else 'no'}</b></div>", unsafe_allow_html=True)
c5.markdown(f"<div class='kpi'>tool calls · tokens · s<br><b>{ans['tool_calls']} · {ans['tokens']} · {ans['latency_s']}</b></div>",
            unsafe_allow_html=True)
st.markdown(f"> {esc(k['summary'])}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Next best action", "Evidence & uncertainty", "Graph", "Timeline & tools",
                                        "SAR & case file"])
with tab1:
    a, b = st.columns(2)
    a.markdown("#### Before evidence came back")
    a.markdown(actions_table(nba["initial"]), unsafe_allow_html=True)
    b.markdown("#### After")
    b.markdown(actions_table(nba["final"]), unsafe_allow_html=True)
    if ans["evidence_requests"]:
        st.markdown("#### Evidence requested (simulated replies)")
        for r in ans["evidence_requests"]:
            st.markdown(f"- **{r['type']}** after step {r['asked_after_step']}: _{esc(r['assumed_response'])}_")
    st.markdown(f"**What changed:** {esc(nba['what_changed'])}")
    st.markdown(f"**Stop reason:** {esc(ans['stop_reason'])}")
    acts = (trace or {}).get("actions") or []
    if acts:
        st.markdown("#### Action desk: what the agent executed and what waits for a human")
        st.dataframe(pd.DataFrame(acts)[["stage", "action", "route", "status", "system", "detail", "ref"]],
                     hide_index=True, use_container_width=True)
    st.caption("Only `auto` actions may be executed by the agent (mock bank systems). L1 = team lead, "
               "L2 = fraud manager: those are queued for approval, never executed.")

with tab2:
    w = pd.DataFrame((trace or {}).get("evidence_weights", []))
    if not w.empty:
        w = w.sort_values("weight", key=lambda s: -s.abs())
        st.markdown("#### How each finding moved the log-odds")
        st.bar_chart(w.set_index("signal")["weight"], horizontal=True, height=320)
    st.markdown("#### Evidence in the case file")
    for e in k["evidence"]:
        st.markdown(f"<div class='ev'>{esc(e['claim'])}<br><span class='small'>{e['source']} · <code>{e['ref']}</code> · "
                    f"{', '.join(e['entity_ids'][:8])}</span></div>", unsafe_allow_html=True)
    st.markdown(f"**Similar prior cases used as memory:** {', '.join(k['similar_prior_cases']) or '-'}")

with tab3:
    components.html(graph_html(ans, trace), height=480)
    if k["connected_device_profiles"]:
        st.markdown("**Shared device profile(s):** " + "; ".join(f"`{d}`" for d in k["connected_device_profiles"]))
    if k["affected_txn_ids"]:
        st.markdown("**Affected transactions:** " + ", ".join(k["affected_txn_ids"]))

with tab4:
    ev = pd.DataFrame((trace or {}).get("events", []))
    if not ev.empty:
        st.dataframe(ev[["step", "kind", "title", "detail", "t"]], hide_index=True, use_container_width=True)
    tc = pd.DataFrame((trace or {}).get("tool_calls", []))
    if not tc.empty:
        st.markdown("#### Tool calls (TigerGraph via MCP)")
        st.dataframe(tc[["step", "tool", "transport", "latency_s", "digest", "why"]], hide_index=True,
                     use_container_width=True)

with tab5:
    if sar["file"]:
        st.markdown("#### Suspicious activity report")
        st.markdown(f"**Reason:** {esc(sar['reason'])}")
        st.markdown(esc(sar["narrative"]))
        st.markdown(f"**Subjects:** {', '.join(sar['subjects'])}  \n**Total:** ${sar['total_amount_usd']:,.2f} · "
                    f"**Dates:** {' to '.join(sar['activity_dates'])}")
    else:
        st.markdown(f"**No SAR.** {esc(sar['reason'])}")
    if k["pattern_description"]:
        st.markdown(f"**Undocumented pattern:** {esc(k['pattern_description'])}")
    st.markdown(f"Written to graph: **{k['written_to_graph']}** as `{k['graph_case_id']}`")
    st.download_button("Download answer file", json.dumps(ans, indent=2), f"{case_id}.json", "application/json")
    with st.expander("Raw answer JSON"):
        st.json(ans)
