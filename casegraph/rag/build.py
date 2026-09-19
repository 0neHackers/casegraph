"""Build the GraphRAG layer inside TigerGraph.

    python -m casegraph.rag.build

* Pattern vertices for the five documented typologies, `undocumented` and `none`.
* DocChunk vertices (with 384-d embeddings) for: the bank's fraud policy, the
  pattern typologies, the answer-format rules the agent must respect, and the
  FinCEN regulatory guidance in knowledge/regulatory. Chunks are linked to the
  Pattern vertices they discuss (DOC_PATTERN), so retrieval returns connected
  context, not loose text.
* ClosedCase.emb  embedding of every analyst narrative (case memory by text).
* ClosedCase.fp   16-d behavioural fingerprint of every closed episode, computed
                  from its transactions in the graph data (case memory by shape).

Patterns, documents and links go through the TigerGraph MCP server; the one-time bulk
load of 11k closed-case vectors uses batched REST upserts.
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb

from casegraph import config as C
from casegraph.agent.mcp_client import MCPGraph
from casegraph.rag.embed import embed, fingerprint

PATTERNS = {
    "card_testing": ("Card testing", "R5"),
    "card_not_present_fraud": ("Card-not-present fraud", "R1 R2 R3 R4"),
    "card_not_present_new_device": ("Card-not-present fraud from a new device", "R1 R2 R3 R4"),
    "out_of_region_use": ("Out-of-region use", "R2 R3"),
    "account_takeover": ("Account takeover", "R2 R10"),
    "undocumented": ("Undocumented pattern", "R6 R9"),
    "none": ("No fraud", "R3 R7"),
}
KEYWORDS = {
    "card_testing": ["card testing", "small online authorizations", "testing a stolen"],
    "card_not_present_fraud": ["card-not-present", "card not present", "online without the card"],
    "card_not_present_new_device": ["new device", "device as `new`", "new phone"],
    "out_of_region_use": ["out-of-region", "billing region", "trip"],
    "account_takeover": ["account takeover", "credentials", "takeover"],
    "undocumented": ["undocumented", "coordinated", "shared origin", "r9", "r6"],
    "none": ["legitimate", "cleared", "recurring", "confirms the transaction"],
}


def _section(text: str, start: str, end: str | None) -> str:
    i = text.index(start)
    j = text.index(end, i + len(start)) if end else len(text)
    return text[i:j]


def readme_chunks() -> list[dict]:
    t = (C.RAW / "README.md").read_text()
    out = []
    pats = _section(t, "## The five known fraud patterns", "## Regulatory references")
    for m in re.finditer(r"\*\*(\d)\. ([^*]+)\*\*(.*?)(?=\n\*\*\d\.|\Z)", pats, re.S):
        out.append(dict(source="bank_readme", section=f"Known pattern {m.group(1)}: {m.group(2).strip()}",
                        text=f"{m.group(2).strip()}.{m.group(3).strip()}"))
    things = _section(t, "## Things to know", "## Rules")
    out.append(dict(source="bank_readme", section="Things to know", text=things.strip()))
    policy = _section(t, "# Fraud Policy", "# Answer Format")
    parts = re.split(r"\n(?=### |\*\*R\d+\.)", policy)
    for p in parts:
        p = p.strip()
        if len(p) < 40:
            continue
        head = p.splitlines()[0].strip("#* ").strip()
        out.append(dict(source="fraud_policy_v1", section=head[:120], text=p[:3500]))
    notes = _section(t, "### Notes", "# The 20 Cases")
    out.append(dict(source="answer_format", section="Answer notes", text=notes.strip()))
    fields = _section(t, "#### Part 2: `sar`", "#### Part 3")
    out.append(dict(source="answer_format", section="SAR fields", text=fields.strip()))
    return out


def discovered_chunks() -> list[dict]:
    """Typologies the analysts confirmed but could not name, lifted from the
    closed-case notes so the agent can recognise them again."""
    return [
        dict(source="case_memory", section="Undocumented: shared new device behind anonymous proxy",
             text="Several cardholders report online purchases they did not make from the same device profile "
                  "(a Samsung SM-G935F on Chrome for Android, behind an anonymous proxy), a device never seen on "
                  "their accounts, within the same month. Analysts could not match it to a documented typology, "
                  "blocked the cards and filed reports naming every connected card. Closed cases CC-2649, CC-2971, "
                  "CC-2985, CC-3035. Policy R6 (shared origin) and R9 (undocumented, coordinated abuse)."),
        dict(source="case_memory", section="Undocumented: purchases structured under an authorisation limit",
             text="A cardholder reports four online purchases within about forty minutes, each just under $500, "
                  "none of which they made. The amounts appear chosen to stay under a $500 authorisation threshold. "
                  "Not matched to a documented typology; card blocked and a report filed. Closed cases CC-3748, "
                  "CC-3841, CC-3907, CC-4086, CC-4124. Policy R9."),
        dict(source="case_memory", section="Cleared alerts: why high scores were false alarms",
             text="Cleared cases in the history were model alerts that turned out legitimate: the cardholder "
                  "confirmed travel to the billing region, confirmed a purchase from a new phone (device added to "
                  "profile), or confirmed an unusual amount that fit their stated intent. A high risk score alone "
                  "was never enough. Policy R1 and R3."),
    ]


def regulatory_chunks(max_chars: int = 1400) -> list[dict]:
    out = []
    names = {
        "sar_guidance_narrative": "FinCEN: Guidance on Preparing a Complete & Sufficient SAR Narrative",
        "SAR-FAQs-October-2025": "FinCEN: SAR Filing FAQs (October 2025)",
        "FTA_Identity_Final508": "FinCEN: Identity-Related Suspicious Activity (2021)",
        "fin-2007-g003": "FinCEN: SAR Supporting Documentation (FIN-2007-G003)",
        "fincen_ato_advisory_2011": "FinCEN Advisory FIN-2011-A016: Account Takeover Activity",
    }
    for f in sorted((C.KNOWLEDGE / "regulatory").glob("*.txt")):
        title = names.get(f.stem)
        if not title:
            continue
        text = re.sub(r"\s+", " ", f.read_text())
        if f.stem == "fincen_ato_advisory_2011":
            i = text.find("The Financial Crimes Enforcement Network (FinCEN) is issuing")
            text = text[i:] if i > 0 else text
        buf, n = [], 0
        for sent in re.split(r"(?<=[.!?]) ", text):
            buf.append(sent)
            if sum(len(s) for s in buf) > max_chars:
                out.append(dict(source="regulatory", section=f"{title} [{n}]", text=" ".join(buf)))
                buf, n = [], n + 1
        if buf:
            out.append(dict(source="regulatory", section=f"{title} [{n}]", text=" ".join(buf)))
    return out


def closed_case_fingerprints() -> dict[str, list[float]]:
    con = duckdb.connect()
    con.execute(f"create view t as select * from read_parquet('{C.PREP / 'txn.parquet'}')")
    con.execute(f"create view cc as select * from read_csv_auto('{C.RAW / 'closed_cases_history.csv'}', all_varchar=true)")
    # region novelty: has this card been billed in this region before this transaction?
    con.execute("""create table rn as select txn_id,
                   (count(*) over (partition by card_id, addr1 order by ts rows between unbounded preceding and 1 preceding)) = 0
                   and addr1 <> '' as region_new from t""")
    rows = con.execute("""select c.case_id, t.amount, t.channel, t.ts, t.device_status, t.proxy_type, t.device_id device,
                          t.addr1 region, t.model_prob, t.risk_score, rn.region_new
                          from (select case_id, cast(unnest(string_split(txn_ids,'|')) as bigint) tid from cc) c
                          join t on t.txn_id = c.tid join rn on rn.txn_id = t.txn_id""").df()
    out = {}
    for cid, g in rows.groupby("case_id"):
        out[cid] = fingerprint(g.to_dict("records"))
    return out


def bulk_vectors(vtype: str, attr: str, items: list, batch: int = 500) -> None:
    """One-time bulk load: batched REST upserts. (The MCP upsert_vectors tool makes one HTTP call per
    vertex, fine for the agent's per-case writes but ~11k round trips for the closed cases.)"""
    from casegraph.graph.conn import conn

    c = conn()
    for i in range(0, len(items), batch):
        c.upsertVertices(vtype, [(k, {attr: v}) for k, v in items[i:i + batch]])


def main(only: str = "all") -> None:
    g = MCPGraph()
    print("transport:", g.transport)
    if only == "cases":
        return closed_cases()
    # patterns
    for pid, (name, rules) in PATTERNS.items():
        g.upsert_vertex("Pattern", pid, {"name": name, "rules": rules, "description": name})
    # documents
    chunks = readme_chunks() + discovered_chunks() + regulatory_chunks()
    vecs = embed([f"{c['section']}\n{c['text']}" for c in chunks])
    edges = []
    for i, (c, v) in enumerate(zip(chunks, vecs)):
        cid = f"doc-{i:04d}"
        c["id"] = cid
        g.upsert_vertex("DocChunk", cid, {"source": c["source"], "section": c["section"], "text": c["text"]})
        low = (c["section"] + " " + c["text"]).lower()
        for pid, kws in KEYWORDS.items():
            if any(k in low for k in kws):
                edges.append((cid, pid, {}))
    for i in range(0, len(chunks), 100):
        g.upsert_vectors("DocChunk", "emb", [(c["id"], v) for c, v in zip(chunks[i:i + 100], vecs[i:i + 100])])
    g.upsert_edges("DOC_PATTERN", "DocChunk", "Pattern", edges)
    print(f"doc chunks: {len(chunks)}  pattern links: {len(edges)}")
    g.close()
    closed_cases()


def closed_cases() -> None:
    con = duckdb.connect()
    cc = con.execute(f"select case_id, pattern, outcome, analyst_notes from read_csv_auto('{C.RAW / 'closed_cases_history.csv'}', all_varchar=true)").fetchall()
    texts = [f"{p} {o}. {n}" for _, p, o, n in cc]
    ids = [r[0] for r in cc]
    print("embedding closed-case narratives ...")
    emb = embed(texts)
    bulk_vectors("ClosedCase", "emb", list(zip(ids, emb)))
    fps = closed_case_fingerprints()
    bulk_vectors("ClosedCase", "fp", list(fps.items()))
    print(f"closed cases: {len(ids)} text vectors, {len(fps)} fingerprints")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "all")
