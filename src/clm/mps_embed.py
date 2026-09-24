"""OpenAI-compatible Qwen3 last-token embeddings on Apple Silicon.

This endpoint supplies the existing CLM Embedder with the unprojected final
hidden state of Qwen3-8B. The CLM client normalizes it before scoring. It is a
functional MPS alternative to the vLLM pooling service, not a speed benchmark.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import threading

import numpy as np
from fastapi import FastAPI, HTTPException, Request

MODEL = "Qwen/Qwen3-8B"
SERVED_MODEL = "qwen3-8b"


class MpsEncoder:
    def __init__(self, model: str = MODEL, device: str = "mps", max_tokens: int = 2048,
                 batch_size: int = 1):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS is unavailable; check Apple Silicon and the torch installation")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_name, self.device, self.max_tokens, self.batch_size = model, device, max_tokens, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        dtype = torch.float16 if device == "mps" else torch.float32
        self.model = AutoModel.from_pretrained(model, torch_dtype=dtype, device_map=device,
                                               low_cpu_mem_usage=True).eval()
        self._lock = threading.Lock()

    def embed(self, inputs: list[str] | list[list[int]], max_tokens: int | None = None) -> tuple[np.ndarray, int]:
        import torch

        limit = min(max_tokens or self.max_tokens, self.max_tokens)
        out, count = [], 0
        with self._lock, torch.inference_mode():
            for start in range(0, len(inputs), self.batch_size):
                chunk = inputs[start:start + self.batch_size]
                if isinstance(chunk[0], str):
                    batch = self.tokenizer([s or " " for s in chunk], padding=True, truncation=True,
                                           max_length=limit, return_tensors="pt")
                else:
                    rows = [ids[-limit:] for ids in chunk]
                    batch = self.tokenizer.pad({"input_ids": rows}, padding=True, return_tensors="pt")
                count += int(batch["attention_mask"].sum())
                indices = batch["attention_mask"].sum(dim=1) - 1
                batch = {k: v.to(self.device) for k, v in batch.items()}
                hidden = self.model(**batch, use_cache=False).last_hidden_state
                vectors = hidden[torch.arange(len(chunk), device=self.device), indices.to(self.device)]
                out.extend(vectors.float().cpu().numpy())
        return np.stack(out), count


def create_app(encoder: MpsEncoder, served_model: str = SERVED_MODEL) -> FastAPI:
    app = FastAPI(title="CLM MPS embedding server")

    @app.get("/health")
    def health():
        return {"ok": True, "model": served_model, "device": encoder.device}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": served_model, "object": "model"}]}

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(422, "body must be JSON") from exc
        if not isinstance(body, dict) or body.get("model") != served_model:
            raise HTTPException(422, f"model must be {served_model!r}")
        inputs = body.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, list) or not inputs or not all(
            isinstance(x, str) or (isinstance(x, list) and bool(x) and all(isinstance(t, int) and t >= 0 for t in x))
            for x in inputs
        ) or any(isinstance(x, str) != isinstance(inputs[0], str) for x in inputs):
            raise HTTPException(422, "input must be text or a non-empty list of texts or token ID lists")
        encoding = body.get("encoding_format", "float")
        if encoding not in ("float", "base64"):
            raise HTTPException(422, "encoding_format must be float or base64")
        limit = body.get("truncate_prompt_tokens")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
            raise HTTPException(422, "truncate_prompt_tokens must be a positive integer")
        try:
            vectors, tokens = await asyncio.get_running_loop().run_in_executor(None, encoder.embed, inputs, limit)
        except (ValueError, IndexError) as exc:
            raise HTTPException(422, str(exc)) from exc
        data = []
        for i, vector in enumerate(vectors):
            arr = np.asarray(vector, dtype="<f4")
            value = base64.b64encode(arr.tobytes()).decode("ascii") if encoding == "base64" else arr.tolist()
            data.append({"object": "embedding", "index": i, "embedding": value})
        return {"object": "list", "data": data, "model": served_model,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL, help="local path or Hugging Face model ID")
    parser.add_argument("--served-model-name", default=SERVED_MODEL)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="encoder microbatch size (default 1 to limit unified-memory use)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    if args.max_model_len < 1:
        parser.error("--max-model-len must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    encoder = MpsEncoder(args.model, args.device, args.max_model_len, args.batch_size)
    print(f"[clm] {args.model} embeddings on {args.device}, port {args.port}", flush=True)
    import uvicorn
    uvicorn.run(create_app(encoder, args.served_model_name), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
