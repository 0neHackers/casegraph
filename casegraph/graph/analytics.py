"""Batch graph analytics with TigerGraph's own algorithm library (GDBMS_ALGO).

    python -m casegraph.graph.analytics

Runs over the Holder <-DEVICE_LINK-> DeviceProfile projection (undirected, weighted by transactions; cardholders and the device profiles that link 2..250 cards). Needs the package once:
`IMPORT PACKAGE GDBMS_ALGO` (already present on Savanna 4.2).

* GDBMS_ALGO.community.wcc      -> wcc_id      connected components
* GDBMS_ALGO.community.louvain  -> louvain_id  modularity communities

The results are written onto the vertices, like a nightly feature job, so the agent's
`device_community` query can read "which community is this device in, and how much
confirmed fraud has that community already produced" in milliseconds during an
investigation. The windowed `device_ring` query (hand-written GSQL label propagation)
stays: it answers the time-bounded question "who shared this device *this fortnight*".
"""
from __future__ import annotations

import time

from casegraph import config as C
from casegraph.graph.setup import gsql

V = '["Holder", "DeviceProfile"]'
E = '["DEVICE_LINK"]'           # undirected, weight = number of transactions
CALLS = {
    "wcc": f'CALL GDBMS_ALGO.community.wcc({V}, {E}, 0, FALSE, "wcc_id", "")',
    "louvain": f'CALL GDBMS_ALGO.community.louvain({V}, {E}, "weight", 10, 10, 1, "louvain_id", "")',
}


def main() -> None:
    for name, call in CALLS.items():
        t = time.time()
        out = gsql(f"USE GRAPH {C.TG_GRAPH}\nSET QUERY_TIMEOUT = 900000\n{call}")
        ok = '"error": false' in out and "Failed to create" not in out and "Syntax Error" not in out
        print(f"{name:8s} {'ok' if ok else 'FAILED'}  {time.time() - t:5.1f}s")
        if not ok:
            print(out[-1500:])


if __name__ == "__main__":
    main()
