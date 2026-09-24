"""Replay the exact CLM README examples against a running local API.

The README numbers come from a CUDA/vLLM run. Deltas here are diagnostics,
not an equivalence test across hardware and numerical formats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from clm import CLMClient, Choice, Noul, Score


REFERENCE = {"urgency": 0.41022, "billing": 0.93878, "frustration": 1.98386, "tides": 0.997}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8700")
    parser.add_argument("--output", default="data/mps-8b-reference.json")
    args = parser.parse_args()

    with CLMClient(base_url=args.api_url, timeout=300) as client:
        result = client.system_one(
            state="Customer: my invoice was charged twice and nobody answers the phone!",
            questions={
                "urgency": Noul(instructions="Is this urgent?"),
                "department": Choice(instructions="Which team should handle this?",
                                     criteria={"billing": "Charges, invoices, refunds",
                                               "technical": "Bugs and outages"}),
                "frustration": Score(instructions="How frustrated is the customer?",
                                     criteria=["Calm", "Frustrated", "Very angry"]),
            },
        )
        tides = client.rank(
            "What causes tides on Earth?", None,
            ["The Moon's gravitational pull.", "Photosynthesis in plants.", "Because the Earth is round."],
        )

    observed = {"urgency": result.answers["urgency"].noul,
                "department": result.answers["department"].choice,
                "billing": result.answers["department"].probabilities["billing"],
                "frustration": result.answers["frustration"].score,
                "tides_top": tides[0]["candidate"], "tides": tides[0]["prob"]}
    deltas = {key: observed[key] - ref for key, ref in REFERENCE.items()}
    report = {"observed": observed, "readme_cuda_reference": REFERENCE,
              "delta": deltas, "typed_request_latency_ms": result.latency_ms,
              "typed_request_input_tokens": result.usage.input_tokens,
              "tides_ranking": tides}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
