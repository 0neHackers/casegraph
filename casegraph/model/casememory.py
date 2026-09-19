"""Case-memory model: learn from the bank's own closed investigations.

The only confirmed outcomes in the dataset are the closed cases (July to
October). Those cases cover the confirmed fraud of that period, so every
July–October transaction that is not in a confirmed case is, as far as the bank
knows, legitimate. That gives a labelled history the agent can learn from
without touching the public Kaggle labels (which the rules forbid).

The model is a gradient-boosted classifier over the original Vesta features
(C, D, M, V, id columns, used as unnamed signals) plus per-card and
per-holder behaviour aggregates. It is validated out-of-time (train Jul–Sep,
test Oct), calibrated with isotonic regression on that October hold-out, then
refit on Jul–Oct and used to score Nov–Dec.

It is one piece of evidence the agent weighs, never the verdict. The bank's own
`risk_score` is another.

    python -m casegraph.model.casememory
"""
from __future__ import annotations

import json
import pickle
import warnings

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pandas.api.types as pt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from casegraph.config import PREP, RAW

warnings.filterwarnings("ignore")

# A de-correlated subset of the 339 V columns (they come in highly redundant blocks).
V_SUBSET = [1, 3, 4, 6, 8, 11, 13, 14, 17, 20, 23, 26, 27, 30, 36, 37, 40, 41, 44, 47, 48, 54, 56, 59, 62, 65, 67,
            68, 70, 76, 78, 80, 82, 86, 88, 89, 91, 107, 108, 111, 115, 117, 120, 121, 123, 124, 127, 129, 130, 136,
            138, 139, 142, 147, 156, 160, 162, 165, 166, 169, 171, 173, 175, 176, 178, 180, 182, 185, 187, 188, 198,
            203, 205, 207, 209, 210, 215, 218, 220, 221, 223, 224, 226, 228, 229, 235, 240, 252, 253, 257, 258, 260,
            261, 264, 266, 267, 274, 277, 281, 283, 285, 289, 291, 294, 296, 297, 301, 303, 305, 307, 309, 310, 314, 320]
BASE = (["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "card2", "card3", "card4", "card5", "card6",
         "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"]
        + [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)] + [f"M{i}" for i in range(1, 10)]
        + [f"V{i}" for i in V_SUBSET])
ID_COLS = [f"id_{i:02d}" for i in range(1, 39)] + ["DeviceType", "DeviceInfo"]
DROP = {"TransactionID", "TransactionDT", "ts", "y", "customer_id", "card_id", "risk_score", "uid", "day"}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=50, feature_fraction=0.4,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, max_cat_to_onehot=4, cat_smooth=20, num_threads=2)


def load_frame() -> pd.DataFrame:
    con = duckdb.connect()
    cols = ", ".join(f't."{c}"' for c in BASE) + ", " + ", ".join(f'i."{c}"' for c in ID_COLS)
    con.execute(f"create view tx as select * from read_csv_auto('{RAW / 'transactions.csv'}', sample_size=-1)")
    con.execute(f"create view idt as select * from read_csv_auto('{RAW / 'identity.csv'}', sample_size=-1)")
    con.execute(f"create view cc as select * from read_csv_auto('{RAW / 'closed_cases_history.csv'}', all_varchar=true)")
    con.execute("""create table cardmap as select customer_id, card6,
                   customer_id || '-K' || dense_rank() over (partition by customer_id order by card6 nulls first) card_id
                   from (select distinct customer_id, card6 from tx)""")
    con.execute("""create table fraud as select distinct cast(unnest(string_split(txn_ids, '|')) as bigint) tid
                   from cc where outcome = 'confirmed_fraud'""")
    df = con.execute(f"""
        select {cols}, t.customer_id, m.card_id, t.ts, t.channel, t.risk_score,
               case when f.tid is null then 0 else 1 end y
        from tx t join cardmap m on m.customer_id = t.customer_id and m.card6 is not distinct from t.card6
        left join idt i using (TransactionID) left join fraud f on f.tid = t.TransactionID""").df()
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    return df


def engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df["hour"] = df.ts.dt.hour.astype("int8")
    df["amt_cents"] = ((df.TransactionAmt * 100) % 100).astype("float32")
    df["day"] = df.TransactionDT // 86400
    df["D1n"] = (df.day - df.D1).astype("float32")               # card-relationship anchor day
    df["uid"] = df.card_id + "_" + df.addr1.astype(str) + "_" + df.D1n.astype(str)   # the Holder
    for k in ["card_id", "uid"]:
        g = df.groupby(k).TransactionAmt
        df[k + "_amt_mean"] = g.transform("mean").astype("float32")
        df[k + "_amt_std"] = g.transform("std").astype("float32")
        df[k + "_n"] = g.transform("count").astype("float32")
        df[k + "_amt_z"] = ((df.TransactionAmt - df[k + "_amt_mean"]) / (df[k + "_amt_std"] + 1)).astype("float32")
    for c in ["P_emaildomain", "id_30", "id_31", "DeviceInfo", "id_33"]:
        df[c + "_uid_nu"] = df.groupby("uid")[c].transform("nunique").astype("float32")
    df["uid_dev_n"] = df.groupby(["uid", "DeviceInfo"]).TransactionAmt.transform("count").astype("float32")
    cats = [c for c in df.columns if c not in DROP and not pt.is_numeric_dtype(df[c])
            and not pt.is_datetime64_any_dtype(df[c]) and not pt.is_bool_dtype(df[c])]
    for c in cats:
        df[c] = df[c].astype("category")
    return df, [c for c in df.columns if c not in DROP]


def main() -> None:
    PREP.mkdir(parents=True, exist_ok=True)
    df, feats = engineer(load_frame())
    month = df.ts.dt.month
    tr, va = month <= 9, month == 10
    print(f"train Jul-Sep: {tr.sum():,} txns, {int(df.y[tr].sum()):,} confirmed fraud | validate Oct: {va.sum():,}")
    b = lgb.train(PARAMS, lgb.Dataset(df.loc[tr, feats], df.y[tr]), 2000,
                  valid_sets=[lgb.Dataset(df.loc[va, feats], df.y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    p = b.predict(df.loc[va, feats])
    metrics = {
        "validation": "out-of-time, train Jul-Sep, test Oct",
        "model_auc": round(float(roc_auc_score(df.y[va], p)), 4),
        "model_ap": round(float(average_precision_score(df.y[va], p)), 4),
        "risk_score_auc": round(float(roc_auc_score(df.y[va], df.risk_score[va])), 4),
        "risk_score_ap": round(float(average_precision_score(df.y[va], df.risk_score[va])), 4),
        "best_iteration": int(b.best_iteration),
    }
    print(json.dumps(metrics, indent=2))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(p, df.y[va])
    final = lgb.train(PARAMS, lgb.Dataset(df.loc[month <= 10, feats], df.y[month <= 10]), int(b.best_iteration * 1.15))
    raw = final.predict(df[feats])
    pd.DataFrame({"TransactionID": df.TransactionID, "model_prob": iso.predict(raw).astype("float32")}) \
        .to_parquet(PREP / "model_scores.parquet")
    final.save_model(str(PREP / "casememory_lgb.txt"))
    pickle.dump(iso, open(PREP / "casememory_iso.pkl", "wb"))
    imp = pd.Series(final.feature_importance("gain"), index=feats).sort_values(ascending=False)
    metrics["top_features"] = [f for f in imp.index[:15]]
    json.dump(metrics, open(PREP / "casememory_metrics.json", "w"), indent=2)
    print("wrote", PREP / "model_scores.parquet")


if __name__ == "__main__":
    main()
