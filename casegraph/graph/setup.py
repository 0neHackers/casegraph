"""Create the schema, load the prepared files, install the queries.

    python -m casegraph.graph.setup schema      # create graph + schema
    python -m casegraph.graph.setup load        # load data/prepared/*.csv over REST (works on Savanna)
    python -m casegraph.graph.setup queries     # install GSQL queries
    python -m casegraph.graph.setup all

Loading goes through the REST++ loading-job endpoint in chunks, so the same
command works against Savanna and a local Community Edition container.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

from casegraph import config as C
from casegraph.graph.conn import admin, conn

HERE = Path(__file__).parent

# job name -> (file, LOAD statements). $"col" refers to header names in the prepared CSVs.
JOBS: dict[str, tuple[str, list[str]]] = {
    "ld_customer": ("customer.csv", ['TO VERTEX Customer VALUES ($"customer_id")']),
    "ld_card": ("card.csv", ['TO VERTEX Card VALUES ($"card_id", $"network", $"card_type", $"n_txn", $"first_seen", $"last_seen")']),
    "ld_holder": ("holder.csv", ['TO VERTEX Holder VALUES ($"holder_id", $"n_txn", $"avg_amount", $"std_amount", $"first_seen", $"last_seen")']),
    "ld_txn": ("transaction.csv", ['TO VERTEX Transaction VALUES ($"txn_id", $"ts", $"amount", $"product", $"channel", $"risk_score", '
                                   '$"addr1", $"addr2", $"dist1", $"p_email", $"r_email", $"card_network", $"card_type", $"c1", $"c13", '
                                   '$"c14", $"d1", $"d15", $"m4", $"m5", $"m6", $"device_status", $"proxy_type", $"device_type", $"id_match", '
                                   '$"model_prob", $"holder_id", $"device_id")']),
    "ld_device": ("device.csv", ['TO VERTEX DeviceProfile VALUES ($"device_id", $"n_cards", $"n_txn")']),
    "ld_email": ("email.csv", ['TO VERTEX EmailDomain VALUES ($"d", $"n_txn")']),
    "ld_region": ("region.csv", ['TO VERTEX BillingRegion VALUES ($"region_id", $"country", $"n_txn", $"n_cards")']),
    "ld_owns": ("e_owns.csv", ['TO EDGE OWNS VALUES ($"customer_id", $"card_id")']),
    "ld_made": ("e_made.csv", ['TO EDGE MADE VALUES ($"card_id", $"txn_id")']),
    "ld_has_holder": ("e_has_holder.csv", ['TO EDGE HAS_HOLDER VALUES ($"card_id", $"holder_id")']),
    "ld_by_holder": ("e_by_holder.csv", ['TO EDGE BY_HOLDER VALUES ($"txn_id", $"holder_id")']),
    "ld_from_device": ("e_from_device.csv", ['TO EDGE FROM_DEVICE VALUES ($"txn_id", $"device_id")']),
    "ld_p_email": ("e_purchaser_email.csv", ['TO EDGE PURCHASER_EMAIL VALUES ($"txn_id", $"p_email")']),
    "ld_r_email": ("e_recipient_email.csv", ['TO EDGE RECIPIENT_EMAIL VALUES ($"txn_id", $"r_email")']),
    "ld_billed": ("e_billed_in.csv", ['TO EDGE BILLED_IN VALUES ($"txn_id", $"addr1")']),
    "ld_next": ("e_next.csv", ['TO EDGE NEXT VALUES ($"txn_id", $"nxt", $"gap_s")']),
    "ld_cc": ("closed_case.csv", ['TO VERTEX ClosedCase VALUES ($"case_id", $"customer_id", $"card_id", $"opened_at", $"closed_at", '
                                  '$"outcome", $"pattern", $"first_fraud_txn_id", $"n_txns", $"exposure_usd", $"actions_taken", '
                                  '$"report_filed", $"analyst_notes")',
                                  'TO EDGE CC_PATTERN VALUES ($"case_id", $"pattern")']),
    "ld_cc_inv": ("e_cc_involves.csv", ['TO EDGE CC_INVOLVES VALUES ($"case_id", $"txn_id")']),
    "ld_cc_card": ("e_cc_on_card.csv", ['TO EDGE CC_ON_CARD VALUES ($"case_id", $"card_id")']),
    "ld_cc_conn": ("e_cc_connected.csv", ['TO EDGE CC_CONNECTED VALUES ($"case_id", $"card_id")']),
}


def gsql(cmd: str, c=None) -> str:
    c = c or conn()
    out = c.gsql(cmd)
    return out if isinstance(out, str) else str(out)


def create_schema() -> None:
    c = admin()
    text = (HERE / "schema.gsql").read_text().replace("@GRAPH@", C.TG_GRAPH)
    existing = gsql("SHOW GRAPH *", c)
    if f"Graph {C.TG_GRAPH}(" in existing:
        print(f"graph {C.TG_GRAPH} already exists; skipping schema")
        return
    print(gsql(text, c)[-1500:])


def create_jobs() -> None:
    import re

    stmts = []
    for job, (fname, loads) in JOBS.items():
        # REST-uploaded files have no server-side path, so header names cannot be
        # resolved at job-creation time: translate $"col" into positional $n.
        header = (C.PREP / fname).open().readline().strip().split(",")
        pos = lambda m: f"${header.index(m.group(1))}"  # noqa: E731
        loads = [re.sub(r'\$"(\w+)"', pos, ld) for ld in loads]
        body = "\n".join(f'  LOAD f {ld} USING header="false", separator=",", QUOTE="double";' for ld in loads)
        stmts.append(f"CREATE LOADING JOB {job} FOR GRAPH {C.TG_GRAPH} {{\n  DEFINE FILENAME f;\n{body}\n}}")
    existing = gsql(f"USE GRAPH {C.TG_GRAPH}\nSHOW JOB *")
    todo = [s for s, j in zip(stmts, JOBS) if f"CREATE LOADING JOB {j} " not in existing and f"- {j}" not in existing]
    if todo:
        print(gsql(f"USE GRAPH {C.TG_GRAPH}\n" + "\n".join(todo))[-800:])


def _chunks(path: Path, max_bytes: int):
    with open(path, "rb") as fh:
        fh.readline()  # header is dropped: jobs use positional columns
        buf, size = [], 0
        for line in fh:
            if size + len(line) > max_bytes and buf:
                yield b"".join(buf)
                buf, size = [], 0
            buf.append(line)
            size += len(line)
        if buf:
            yield b"".join(buf)


def load(only: list[str] | None = None, chunk_mb: int = 16) -> None:
    create_jobs()
    c = conn()
    for job, (fname, _) in JOBS.items():
        if only and job not in only:
            continue
        path = C.PREP / fname
        t0, ok, bad = time.time(), 0, 0
        for chunk in _chunks(path, chunk_mb * 1024 * 1024):
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(chunk)
            res = c.runLoadingJobWithFile(tmp.name, "f", job, sep=",", timeout=600000, sizeLimit=chunk_mb * 2 * 1024 * 1024)
            Path(tmp.name).unlink()
            for r in res or []:
                fl = r.get("statistics", {}).get("parsingStatistics", {}).get("fileLevel", {})
                ok += int(fl.get("validLine", 0))
                bad += sum(int(v) for k, v in fl.items() if k != "validLine" and str(v).isdigit())
        print(f"  {job:16s} {fname:26s} valid={ok:>9,} rejected={bad:>5,}  {time.time() - t0:5.1f}s")


def install_queries() -> None:
    qdir = HERE / "queries"
    files = sorted(qdir.glob("*.gsql"))
    names = []
    for f in files:
        text = f.read_text().replace("@GRAPH@", C.TG_GRAPH)
        out = gsql(f"USE GRAPH {C.TG_GRAPH}\n{text}")
        name = f.stem.split("_", 1)[1] if f.stem[:2].isdigit() else f.stem
        names.append(name)
        if "error" in out.lower() and "successfully" not in out.lower():
            print(f"!! {f.name}\n{out[-1500:]}")
    print(gsql(f"USE GRAPH {C.TG_GRAPH}\nINSTALL QUERY ALL")[-1500:])


def counts() -> None:
    c = conn()
    print("vertices:", c.getVertexCount("*"))
    print("edges:", c.getEdgeCount("*"))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("schema", "all"):
        create_schema()
    if step in ("load", "all"):
        load(sys.argv[2:] or None)
    if step in ("queries", "all"):
        install_queries()
    if step in ("counts", "all"):
        counts()
