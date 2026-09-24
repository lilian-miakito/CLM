"""End-to-end diagnostic for Qwen3-8B on MPS and the released CLM head.

Start clm-mps-embed and clm-serve first. The cases are small, hand-written
checks of the serving path, not a benchmark of CLM answer quality.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import time
from pathlib import Path

import numpy as np
import requests

from mps_small_smoke import CASES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8700")
    parser.add_argument("--emb-url", default="http://127.0.0.1:8090")
    parser.add_argument("--output", default="data/mps-8b-smoke.json")
    args = parser.parse_args()

    api = args.api_url.rstrip("/")
    emb = args.emb_url.rstrip("/")
    health = requests.get(f"{api}/health", timeout=10).json()
    assert health["ok"] and health["embedder"]
    assert "clm-latest" in health["models"]
    encoder_health = requests.get(f"{emb}/health", timeout=10).json()
    assert encoder_health["ok"] and encoder_health["device"] == "mps"

    response = requests.post(
        f"{emb}/v1/embeddings",
        json={"model": "qwen3-8b", "input": ["Embedding shape check."], "encoding_format": "base64"},
        timeout=300,
    )
    response.raise_for_status()
    vector = np.frombuffer(base64.b64decode(response.json()["data"][0]["embedding"]), dtype="<f4")
    assert vector.shape == (4096,) and np.isfinite(vector).all()

    rows = []
    for case in CASES:
        start = time.perf_counter()
        response = requests.post(
            f"{api}/v1/rank",
            json={"context": case["state"], "question": case["question"],
                  "answers": case["candidates"], "model": "clm-latest"},
            timeout=300,
        )
        response.raise_for_status()
        ranked = response.json()["ranked"]
        assert {row["candidate"] for row in ranked} == set(case["candidates"])
        assert all(math.isfinite(row["prob"]) for row in ranked)
        assert abs(sum(row["prob"] for row in ranked) - 1.0) < 1e-5
        rows.append({"id": case["id"], "expected": case["expected"],
                     "top": ranked[0]["candidate"], "match": ranked[0]["candidate"] == case["expected"],
                     "ranked": ranked, "seconds": round(time.perf_counter() - start, 3)})
        print(f"{case['id']}: {rows[-1]['top']} ({rows[-1]['seconds']} s)", flush=True)

    typed = requests.post(
        f"{api}/v1/systemone",
        json={"state": "A customer reports a duplicate credit-card charge and asks for a refund.",
              "questions": {"team": {"type": "choice", "instructions": "Who should handle this?",
                                     "criteria": {"billing": "Investigate charges and refunds",
                                                  "engineering": "Fix technical application errors"}}},
              "model": "clm-latest"},
        timeout=300,
    )
    typed.raise_for_status()
    typed_answer = typed.json()["answers"]["team"]
    assert typed_answer["type"] == "choice"
    print(f"typed choice: {typed_answer['choice']}", flush=True)

    report = {"model": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218",
              "head": "Contrastive-LM/CLM-v0.1-8B", "device": "mps",
              "dtype": encoder_health.get("dtype", "unknown"), "embedding_width": len(vector),
              "method": "clm-latest trained projection head", "cases": rows,
              "matches": sum(row["match"] for row in rows), "total": len(rows),
              "typed_choice": typed_answer}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output}; {report['matches']}/{report['total']} expected tops", flush=True)


if __name__ == "__main__":
    main()
