"""A small synchronous facade over the TigerGraph MCP server.

The agent talks to TigerGraph through the official `tigergraph-mcp` server
(stdio transport): installed GSQL queries, vector search/upsert, and node/edge
writes for case memory are all MCP tool calls. A background thread owns the
asyncio loop and the MCP session so the agent code can stay synchronous.

If the MCP server cannot start (CASEGRAPH_USE_MCP=false, or no binary), the
same calls fall back to pyTigerGraph REST so a run never silently stops; the
transport actually used is recorded in every tool-call log line.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
from concurrent.futures import Future
from typing import Any

from casegraph import config as C
from casegraph.graph.conn import conn, mcp_env


def _parse(text: str):
    """tigergraph-mcp answers in markdown: the structured payload is the first ```json block."""
    import re

    m = re.search(r"```json\s*(.*?)```", text, re.S)
    try:
        return json.loads(m.group(1) if m else text)
    except (json.JSONDecodeError, AttributeError):
        return None


class MCPGraph:
    def __init__(self, use_mcp: bool | None = None):
        self.use_mcp = C.USE_MCP if use_mcp is None else use_mcp
        self.transport = "rest"
        self.tools: list[str] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._err: Exception | None = None
        if self.use_mcp and shutil.which("tigergraph-mcp"):
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self._ready.wait(90)
            if self._session is not None:
                self.transport = "mcp"

    # -- lifecycle -------------------------------------------------------------
    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:  # pragma: no cover
            self._err = e
            self._ready.set()

    async def _main(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import get_default_environment, stdio_client

        params = StdioServerParameters(command="tigergraph-mcp", args=[],
                                       env={**get_default_environment(), **mcp_env()})
        self._stop = asyncio.Event()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.tools = [t.name for t in (await session.list_tools()).tools]
                self._session = session
                self._ready.set()
                await self._stop.wait()

    def close(self) -> None:
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)

    # -- calls -------------------------------------------------------------------
    def call_tool(self, name: str, args: dict[str, Any], timeout: float = 180) -> Any:
        fut: Future = asyncio.run_coroutine_threadsafe(self._session.call_tool(name, arguments=args), self._loop)
        res = fut.result(timeout)
        text = "".join(getattr(c, "text", "") for c in res.content)
        if getattr(res, "isError", False):
            raise RuntimeError(f"{name}: {text[:500]}")
        data = _parse(text)
        if data is None:
            return text
        # tigergraph-mcp wraps results as {"success":..,"data":..} (structured responses)
        if isinstance(data, dict) and data.get("success") is False:
            raise RuntimeError(f"{name}: {data.get('error') or data.get('summary') or data.get('message')}")
        if isinstance(data, dict) and "data" in data and ("success" in data or "status" in data):
            return data["data"]
        return data

    def run_query(self, name: str, params: dict[str, Any]) -> list[dict]:
        t0 = time.time()
        if self.transport == "mcp":
            out = self.call_tool("tigergraph__run_installed_query",
                                 {"graph_name": C.TG_GRAPH, "query_name": name, "params": params})
            if isinstance(out, dict):
                out = out.get("result", out.get("results", out))
        else:
            out = conn().runInstalledQuery(name, params, timeout=180000)
        self.last_latency = time.time() - t0
        return out

    def upsert_vertex(self, vtype: str, vid: str, attrs: dict[str, Any]) -> None:
        if self.transport == "mcp":
            self.call_tool("tigergraph__add_node", {"graph_name": C.TG_GRAPH, "vertex_type": vtype,
                                                    "vertex_id": vid, "attributes": attrs})
        else:
            conn().upsertVertex(vtype, vid, attrs)

    def upsert_edges(self, etype: str, src_type: str, tgt_type: str, pairs: list[tuple[str, str, dict]]) -> None:
        if not pairs:
            return
        if self.transport == "mcp":
            edges = [{"source_type": src_type, "source_id": s, "target_type": tgt_type, "target_id": t, **a}
                     for s, t, a in pairs]
            self.call_tool("tigergraph__add_edges", {"graph_name": C.TG_GRAPH, "edge_type": etype, "edges": edges})
        else:
            conn().upsertEdges(src_type, etype, tgt_type, [(s, t, a) for s, t, a in pairs])

    def upsert_vectors(self, vtype: str, attr: str, rows: list[tuple[str, list[float]]]) -> None:
        if self.transport == "mcp":
            self.call_tool("tigergraph__upsert_vectors", {
                "graph_name": C.TG_GRAPH, "vertex_type": vtype, "vector_attribute": attr,
                "vectors": [{"vertex_id": i, "vector": v} for i, v in rows]})
        else:
            conn().upsertVertices(vtype, [(i, {attr: v}) for i, v in rows])
