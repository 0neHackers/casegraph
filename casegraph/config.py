"""Paths and settings. Everything configurable comes from the environment / .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

RAW = Path(os.getenv("CASEGRAPH_RAW", ROOT / "data" / "raw"))
PREP = Path(os.getenv("CASEGRAPH_PREP", ROOT / "data" / "prepared"))
CASES_DIR = ROOT / "cases"
RUNS_DIR = ROOT / "runs"
KNOWLEDGE = ROOT / "knowledge"

# TigerGraph ------------------------------------------------------------------
TG_HOST = os.getenv("TG_HOST", "http://127.0.0.1")
TG_GRAPH = os.getenv("TG_GRAPHNAME", "CaseGraph")
TG_USERNAME = os.getenv("TG_USERNAME", "tigergraph")
TG_PASSWORD = os.getenv("TG_PASSWORD", "tigergraph")
TG_SECRET = os.getenv("TG_SECRET", "")
TG_API_TOKEN = os.getenv("TG_API_TOKEN", "")
TG_RESTPP_PORT = os.getenv("TG_RESTPP_PORT", "9000")
TG_GS_PORT = os.getenv("TG_GS_PORT", "14240")
TG_TGCLOUD = os.getenv("TG_TGCLOUD", "false").lower() == "true"

# LLM ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACKS = [m.strip() for m in os.getenv(
    "GEMINI_FALLBACKS", "gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-flash-latest,gemini-3-flash-preview"
).split(",")]

# Embeddings run locally (no key, deterministic) ---------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

# Agent ---------------------------------------------------------------------------
USE_MCP = os.getenv("CASEGRAPH_USE_MCP", "true").lower() == "true"
