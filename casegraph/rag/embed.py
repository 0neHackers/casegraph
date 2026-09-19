"""Embeddings (local, deterministic) and the behavioural episode fingerprint."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from casegraph import config as C

THRESHOLDS = (100, 200, 250, 300, 500, 1000, 1500, 2000, 2500, 3000, 5000, 10000)


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(C.EMBED_MODEL, cache_dir=str(C.ROOT / "data" / "model_cache"))


def embed(texts: list[str]) -> list[list[float]]:
    return [list(map(float, v)) for v in _model().embed(texts, batch_size=64)]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def near_threshold(amount: float) -> bool:
    """Just under a round authorisation limit (within 10%)."""
    return any(t * 0.90 <= amount < t for t in THRESHOLDS)


def fingerprint(txns: list[dict]) -> list[float]:
    """16-d shape of an episode. txns: dicts with amount, channel, ts (datetime or str),
    device_status, proxy_type, device, region, model_prob, risk_score, region_new (bool, optional)."""
    if not txns:
        return [0.0] * 15 + [1.0]
    amts = np.array([abs(float(t.get("amount", 0))) for t in txns])
    n = len(txns)
    online = sum(1 for t in txns if t.get("channel") == "online")
    ts = sorted(np.datetime64(str(t.get("ts"))[:19].replace(" ", "T")) for t in txns)
    span_h = float((ts[-1] - ts[0]) / np.timedelta64(1, "h")) if n > 1 else 0.0
    devices = {t.get("device") for t in txns if t.get("device")}
    regions = {t.get("region") for t in txns if t.get("region")}
    fp = [
        min(math.log1p(n) / 4.0, 1.0),
        min(math.log1p(amts.sum()) / 9.0, 1.0),
        online / n,
        (n - online) / n,
        sum(1 for t in txns if t.get("device_status") == "New") / n,
        sum(1 for t in txns if t.get("proxy_type")) / n,
        min(math.log1p(span_h) / 7.0, 1.0),
        min(math.log1p(float(amts.mean())) / 8.0, 1.0),
        float((amts < 5).mean()),
        min(float(amts.std() / (amts.mean() + 1e-9)), 2.0) / 2.0,
        float(np.mean([near_threshold(a) for a in amts])),
        min(math.log1p(len(devices)) / 3.0, 1.0),
        min(math.log1p(len(regions)) / 3.0, 1.0),
        float(np.mean([1.0 if t.get("region_new") else 0.0 for t in txns])),
        float(np.mean([float(t.get("model_prob") or 0) for t in txns])),
        0.5 + 0.5 * float(np.mean([float(t.get("risk_score") or 0) for t in txns])),
    ]
    return [round(x, 5) for x in fp]
