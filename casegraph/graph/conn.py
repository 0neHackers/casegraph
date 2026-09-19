"""One place that knows how to reach TigerGraph (Savanna or Community Edition)."""
from __future__ import annotations

from functools import lru_cache

from pyTigerGraph import TigerGraphConnection

from casegraph import config as C


def _base(graph: str | None = None) -> TigerGraphConnection:
    kw = dict(host=C.TG_HOST, graphname=graph or C.TG_GRAPH, username=C.TG_USERNAME, password=C.TG_PASSWORD,
              restppPort=C.TG_RESTPP_PORT, gsPort=C.TG_GS_PORT, tgCloud=C.TG_TGCLOUD)
    if C.TG_SECRET:
        kw["gsqlSecret"] = C.TG_SECRET
    if C.TG_API_TOKEN:
        kw["apiToken"] = C.TG_API_TOKEN
    return TigerGraphConnection(**kw)


@lru_cache(maxsize=1)
def conn() -> TigerGraphConnection:
    c = _base()
    if C.TG_SECRET and not C.TG_API_TOKEN:
        try:
            c.getToken(C.TG_SECRET)
        except Exception:  # CE without auth enabled does not need a token
            pass
    return c


def admin() -> TigerGraphConnection:
    """Connection for DDL before the graph exists."""
    return _base()


def mcp_env() -> dict[str, str]:
    """Environment for the tigergraph-mcp stdio server (it does not inherit ours)."""
    env = {"TG_HOST": C.TG_HOST, "TG_GRAPHNAME": C.TG_GRAPH, "TG_USERNAME": C.TG_USERNAME,
           "TG_PASSWORD": C.TG_PASSWORD, "TG_RESTPP_PORT": str(C.TG_RESTPP_PORT), "TG_GS_PORT": str(C.TG_GS_PORT),
           "TG_TGCLOUD": "true" if C.TG_TGCLOUD else "false"}
    if C.TG_SECRET:
        env["TG_SECRET"] = C.TG_SECRET
    if C.TG_API_TOKEN:
        env["TG_API_TOKEN"] = C.TG_API_TOKEN
    return env
