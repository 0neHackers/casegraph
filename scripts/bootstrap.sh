#!/usr/bin/env bash
# One-shot setup: model -> ETL -> TigerGraph schema/load/queries/algorithms -> GraphRAG vectors -> 20 cases -> checks.
# Needs: data/raw/{transactions,identity,closed_cases_history,case_pack}.csv + README.md, and a filled .env
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
[ -f data/prepared/model_scores.parquet ] || python -m casegraph.model.casememory   # ~3 min, optional: scores ship in the repo
python -m casegraph.etl.prepare                  # vertex/edge CSVs (~40 s)
python -m casegraph.graph.setup schema
python -m casegraph.graph.setup load             # REST, chunked; works on Savanna
python -m casegraph.graph.setup queries          # installs 13 GSQL queries (~5 min)
python -m casegraph.graph.analytics              # GDBMS_ALGO Louvain + WCC -> louvain_id / wcc_id (~1 min)
python -m casegraph.rag.build                    # patterns, policy/regulatory chunks, closed-case vectors (~6 min)
python -m casegraph reset-memory
python -m casegraph run --all                    # the 20 exam cases -> cases/*.json, written to the graph
python -m casegraph check
