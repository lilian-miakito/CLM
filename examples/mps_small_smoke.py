"""Diagnostic examples with a small pretrained Qwen3 encoder and clm-raw.

This does not use the Qwen3-8B projection heads, so its rankings are not CLM
quality measurements. Run after starting clm-mps-embed with Qwen3-0.6B.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from clm import Engine
from clm.embedder import Embedder


CASES = [
    {
        "id": "duplicate_charge",
        "state": "A customer says their card was charged twice for the same invoice and asks for a refund.",
        "question": "Which team should handle this?",
        "candidates": [
            "Billing: investigate duplicate charges and refunds.",
            "Technical support: fix login and application errors.",
            "Sales: discuss a new subscription.",
        ],
        "expected": "Billing: investigate duplicate charges and refunds.",
    },
    {
        "id": "http_500",
        "state": "The web app returns HTTP 500 immediately after a deployment.",
        "question": "Which team should take the next action?",
        "candidates": [
            "Engineering: investigate production errors and roll back the deployment.",
            "Billing: review invoices and payments.",
            "Sales: prepare a customer proposal.",
        ],
        "expected": "Engineering: investigate production errors and roll back the deployment.",
    },
    {
        "id": "tides",
        "state": "What is the main cause of ocean tides on Earth?",
        "question": "Choose the most accurate answer.",
        "candidates": [
            "The Moon's gravitational pull.",
            "Photosynthesis in marine plants.",
            "Earth's spherical shape.",
        ],
        "expected": "The Moon's gravitational pull.",
    },
    {
        "id": "translation",
        "state": "Translate the French sentence « Merci pour votre aide » into English.",
        "question": "Which translation is correct?",
        "candidates": [
            "Thank you for your help.",
            "Please send the invoice.",
            "The server is offline.",
        ],
        "expected": "Thank you for your help.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emb-url", default="http://127.0.0.1:8090/v1/embeddings")
    parser.add_argument("--emb-model", default="qwen3-0.6b")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", default="data/mps-small-smoke.json")
    args = parser.parse_args()

    embedder = Embedder(url=args.emb_url, model=args.emb_model, max_tokens=2048)
    if not embedder.healthy():
        parser.error(f"embedding server is not reachable at {args.emb_url}")
    sample, _ = embedder.embed(["Embedding shape check."])
    assert sample.ndim == 2 and sample.shape[0] == 1 and np.isfinite(sample).all()
    assert np.allclose(np.linalg.norm(sample, axis=1), 1.0, atol=1e-4)

    engine = Engine(embedder=embedder, device="mps", action_cache=0)
    rows = []
    for case in CASES:
        start = time.perf_counter()
        ranked = engine.rank(case["state"], case["candidates"], case["question"], model="clm-raw")
        rows.append({"id": case["id"], "expected": case["expected"],
                     "top": ranked[0]["candidate"], "match": ranked[0]["candidate"] == case["expected"],
                     "ranked": ranked, "seconds": round(time.perf_counter() - start, 3)})
        print(f"{case['id']}: {rows[-1]['top']} ({rows[-1]['seconds']} s)", flush=True)

    report = {"model": "Qwen/Qwen3-0.6B", "revision": args.model_revision,
              "device": "mps", "embedding_width": int(sample.shape[1]),
              "method": "clm-raw, no trained projection head", "cases": rows,
              "matches": sum(row["match"] for row in rows), "total": len(rows)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"saved {args.output}; {report['matches']}/{report['total']} expected tops", flush=True)


if __name__ == "__main__":
    main()
