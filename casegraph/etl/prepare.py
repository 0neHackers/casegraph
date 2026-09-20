"""Turn the raw HHGOA_IEEE CSVs into vertex / edge files for TigerGraph.

Run once:  python -m casegraph.etl.prepare

What it derives (everything else is copied through):

* card_id      customer_id + "-K" + dense rank of card6 (card type) within the
               customer, NULL first. This reproduces the card IDs used in the
               closed cases and the case pack exactly (verified on all 14,955
               closed-case transactions).
* holder_id    card_id | addr1 | anchor day. A "customer" in this data is an
               issuer bucket that can hold thousands of transactions across
               dozens of billing regions, so it is not one person. The anchor
               day (transaction day minus D1, i.e. the day the card relationship
               started) plus billing region isolates a single cardholder far
               better. We call that inferred person a Holder.
* device_id    DeviceInfo | OS | browser | screen, the device profile the task
               defines, for online transactions that have an identity record.
* prev/next    per-card time ordering for the NEXT edge.

No outcome label from the public Kaggle files is used anywhere.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import duckdb

from casegraph.config import RAW, PREP

TXN_ATTRS = [
    # name in graph, SQL expression
    ("txn_id", "t.TransactionID"),
    ("ts", "strftime(t.ts, '%Y-%m-%d %H:%M:%S')"),
    ("amount", "round(t.TransactionAmt, 2)"),
    ("product", "t.ProductCD"),
    ("channel", "t.channel"),
    ("risk_score", "round(t.risk_score, 4)"),
    ("addr1", "coalesce(cast(cast(t.addr1 as int) as varchar), '')"),
    ("addr2", "coalesce(cast(cast(t.addr2 as int) as varchar), '')"),
    ("dist1", "coalesce(t.dist1, -1)"),
    ("p_email", "coalesce(t.P_emaildomain, '')"),
    ("r_email", "coalesce(t.R_emaildomain, '')"),
    ("card_network", "coalesce(t.card4, '')"),
    ("card_type", "coalesce(t.card6, '')"),
    ("c1", "coalesce(t.C1, -1)"),
    ("c13", "coalesce(t.C13, -1)"),
    ("c14", "coalesce(t.C14, -1)"),
    ("d1", "coalesce(t.D1, -1)"),
    ("d15", "coalesce(t.D15, -1)"),
    ("m4", "coalesce(t.M4, '')"),
    ("m5", "coalesce(cast(t.M5 as varchar), '')"),
    ("m6", "coalesce(cast(t.M6 as varchar), '')"),
    ("device_status", "coalesce(i.id_15, '')"),      # New / Found / Unknown
    ("proxy_type", "coalesce(i.id_23, '')"),              # IP_PROXY:TRANSPARENT / ANONYMOUS / HIDDEN
    ("device_type", "coalesce(i.DeviceType, '')"),
    ("id_match", "coalesce(i.id_34, '')"),
    ("model_prob", "round(coalesce(ms.model_prob, 0), 4)"),
    ("holder_id", "h.holder_id"),
    ("device_id", "coalesce(dv.device_id, '')"),
]


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    return con


def build(model_scores: Path | None = None) -> None:
    PREP.mkdir(parents=True, exist_ok=True)
    con = _con()
    tx = RAW / "transactions.csv"
    idt = RAW / "identity.csv"
    print("reading transactions + identity ...")
    con.execute(f"create table tx as select * from read_csv_auto('{tx}', sample_size=-1)")
    con.execute(f"create table idt as select * from read_csv_auto('{idt}', sample_size=-1)")
    con.execute(f"create table cc as select * from read_csv_auto('{RAW / 'closed_cases_history.csv'}', all_varchar=true)")
    con.execute(f"create table cp as select * from read_csv_auto('{RAW / 'case_pack.csv'}', all_varchar=true)")

    # card ids ---------------------------------------------------------------
    con.execute("""
        create table cardmap as
        select customer_id, card6,
               customer_id || '-K' || dense_rank() over (partition by customer_id order by card6 nulls first) as card_id
        from (select distinct customer_id, card6 from tx)""")
    con.execute("""
        create table t as
        select tx.*, m.card_id,
               floor(tx.TransactionDT / 86400) - tx.D1 as anchor_day
        from tx join cardmap m on m.customer_id = tx.customer_id and m.card6 is not distinct from tx.card6""")

    # holders ------------------------------------------------------------------
    con.execute("""
        create table h as
        select TransactionID, card_id || '|' || coalesce(cast(cast(addr1 as int) as varchar), 'na') || '|' ||
               coalesce(cast(cast(anchor_day as int) as varchar), 'na') as holder_id
        from t""")

    # device profiles ------------------------------------------------------------
    con.execute("""
        create table dv as
        select TransactionID,
               coalesce(DeviceInfo, '?') || ' | ' || coalesce(id_30, '?') || ' | ' || coalesce(id_31, '?') || ' | ' || coalesce(id_33, '?') as device_id
        from idt
        where not (DeviceInfo is null and id_30 is null and id_31 is null and id_33 is null)""")

    if model_scores and Path(model_scores).exists():
        con.execute(f"create table ms as select TransactionID, model_prob from read_parquet('{model_scores}')")
    else:
        print("  (no model scores yet: model_prob will be 0)")
        con.execute("create table ms as select TransactionID, 0.0::double as model_prob from tx limit 0")

    sel = ", ".join(f"{expr} as {name}" for name, expr in TXN_ATTRS)
    con.execute(f"""
        create table txn as
        select {sel}, t.card_id, t.customer_id
        from t left join idt i using (TransactionID)
               left join ms using (TransactionID)
               join h using (TransactionID)
               left join dv using (TransactionID)""")

    out = lambda name: str(PREP / name)  # noqa: E731
    attr_cols = ", ".join(name for name, _ in TXN_ATTRS)
    print("writing vertices ...")
    con.execute(f"copy (select {attr_cols} from txn order by txn_id) to '{out('transaction.csv')}' (header, delimiter ',')")
    con.execute(f"copy (select distinct customer_id from txn) to '{out('customer.csv')}' (header)")
    con.execute(f"""copy (select card_id, any_value(customer_id) customer_id, any_value(card_network) network, any_value(card_type) card_type,
                          count(*) n_txn, strftime(min(ts::timestamp), '%Y-%m-%d %H:%M:%S') first_seen, strftime(max(ts::timestamp), '%Y-%m-%d %H:%M:%S') last_seen
                          from txn group by card_id) to '{out('card.csv')}' (header)""")
    con.execute(f"""copy (select holder_id, any_value(card_id) card_id, count(*) n_txn, round(avg(amount),2) avg_amount,
                          round(coalesce(stddev_samp(amount),0),2) std_amount,
                          strftime(min(ts::timestamp), '%Y-%m-%d %H:%M:%S') first_seen, strftime(max(ts::timestamp), '%Y-%m-%d %H:%M:%S') last_seen
                          from txn group by holder_id) to '{out('holder.csv')}' (header)""")
    con.execute(f"""copy (select device_id, count(distinct card_id) n_cards, count(*) n_txn
                          from txn where device_id <> '' group by 1) to '{out('device.csv')}' (header)""")
    con.execute(f"""copy (select d, count(*) n_txn from (select p_email d from txn union all select r_email from txn) where d <> '' group by 1)
                    to '{out('email.csv')}' (header)""")
    con.execute(f"""copy (select addr1 region_id, any_value(addr2) country, count(*) n_txn, count(distinct card_id) n_cards
                          from txn where addr1 <> '' group by 1) to '{out('region.csv')}' (header)""")

    print("writing edges ...")
    con.execute(f"copy (select distinct customer_id, card_id from txn) to '{out('e_owns.csv')}' (header)")
    con.execute(f"copy (select card_id, txn_id from txn) to '{out('e_made.csv')}' (header)")
    con.execute(f"copy (select distinct card_id, holder_id from txn) to '{out('e_has_holder.csv')}' (header)")
    con.execute(f"copy (select txn_id, holder_id from txn) to '{out('e_by_holder.csv')}' (header)")
    con.execute(f"copy (select txn_id, device_id from txn where device_id <> '') to '{out('e_from_device.csv')}' (header)")
    con.execute(f"copy (select txn_id, p_email from txn where p_email <> '') to '{out('e_purchaser_email.csv')}' (header)")
    con.execute(f"copy (select txn_id, r_email from txn where r_email <> '') to '{out('e_recipient_email.csv')}' (header)")
    con.execute(f"copy (select txn_id, addr1 from txn where addr1 <> '') to '{out('e_billed_in.csv')}' (header)")
    con.execute(f"""copy (select txn_id, nxt, gap_s from (
                          select txn_id, lead(txn_id) over w nxt,
                                 date_diff('second', ts::timestamp, lead(ts::timestamp) over w) gap_s
                          from txn window w as (partition by card_id order by ts, txn_id)) where nxt is not null)
                    to '{out('e_next.csv')}' (header)""")

    # holder -> device projection for the graph-algorithm library ----------------------
    # Only devices that link 2..250 cards: a device seen once links nobody, and a generic
    # profile ("Windows | chrome") would glue thousands of cardholders into one blob.
    con.execute(f"""copy (select t.holder_id, t.device_id, count(*) n_txn
                          from txn t join (select device_id from txn where device_id <> '' group by 1
                                           having count(distinct card_id) between 2 and 250) d using (device_id)
                          group by 1, 2) to '{out('e_used_device.csv')}' (header)""")

    # closed cases ---------------------------------------------------------------
    con.execute(f"""copy (select case_id, customer_id, card_id, opened_at, closed_at, outcome, pattern, first_fraud_txn_id,
                          n_txns, exposure_usd, actions_taken, report_filed, analyst_notes from cc)
                    to '{out('closed_case.csv')}' (header)""")
    con.execute(f"""copy (select case_id, unnest(string_split(txn_ids, '|')) txn_id from cc) to '{out('e_cc_involves.csv')}' (header)""")
    con.execute(f"copy (select case_id, card_id from cc) to '{out('e_cc_on_card.csv')}' (header)")
    con.execute(f"""copy (select case_id, unnest(string_split(connected_card_ids, '|')) card_id from cc where connected_card_ids is not null and connected_card_ids <> '')
                    to '{out('e_cc_connected.csv')}' (header)""")
    con.execute(f"copy (select * from cp) to '{out('case_pack.csv')}' (header)")

    # a parquet copy of the slim transaction table for the offline tools
    con.execute(f"copy (select * from txn) to '{out('txn.parquet')}' (format parquet)")
    for f in sorted(PREP.glob("*.csv")):
        n = sum(1 for _ in open(f)) - 1
        print(f"  {f.name:28s} {n:>9,}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(PREP / "model_scores.parquet"))
    a = ap.parse_args()
    build(Path(a.scores))
