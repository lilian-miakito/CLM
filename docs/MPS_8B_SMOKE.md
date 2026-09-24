# Qwen3-8B MPS functional test

This is a local end-to-end check with the released CLM projection head and the
original Qwen3-8B safetensors. It is not a quality benchmark or a numerical
equivalence claim against the CUDA/vLLM reference.

## Setup

Observed on one Apple Silicon Mac with 24 GB unified memory, macOS 26.6.2,
Python 3.12.14, PyTorch 2.14.0, and Transformers 4.57.6. The model snapshot was
`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`; the head was
`Contrastive-LM/CLM-v0.1-8B/CLM_v0.1-8B.pt` at revision
`87655cb835bd76fd66c2da78e1e3709f7fa11a94` (SHA-256
`b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5`).
The first full smoke used MPS float16,
last non-padding token pooling, one text at a time, and a 512-token limit. The
head ran on MPS with its action cache disabled.

On this Transformers version, the first 8B load failed because the allocator
warmup tried to create a single 14.10 GiB MPS buffer. The MPS loader now skips
that warmup, while still loading the model shards directly onto MPS. All five
shards then loaded and both HTTP services became healthy.

## Reproduce

From a checkout with `requirements-macos.txt` installed:

```bash
clm-download --dest data/checkpoints
```

Start the encoder and CLM API in separate terminals:

```bash
MODEL_DIR=$(hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218)
clm-mps-embed --model "$MODEL_DIR" \
  --device mps --dtype float16 --max-model-len 512 --port 8090
```

```bash
clm-serve --host 127.0.0.1 --port 8700 --device mps --action-cache 0 \
  --ckpt data/checkpoints/CLM_v0.1-8B.pt \
  --emb-url http://127.0.0.1:8090/v1/embeddings --max-tokens 512
```

```bash
python examples/mps_8b_smoke.py
python examples/mps_8b_reference.py
```

To replay the native-weight test, restart the encoder with `--dtype bfloat16`
(`--dtype auto` selects it on this Mac), then run
`python examples/mps_8b_reference.py --output data/mps-8b-reference-bf16.json`.

## Observed

The embedding endpoint returned a finite 4096-element vector. All four rank
requests and a typed-choice request completed through the API. The typed
choice selected `billing` for a duplicate card charge.

| Case | Expected top | Observed top | Match | Wall time |
| --- | --- | --- | :---: | ---: |
| Duplicate charge | Billing | Billing | Yes | 77.799 s |
| HTTP 500 after deploy | Engineering | Engineering | Yes | 56.283 s |
| Ocean tides | Moon's gravity | Moon's gravity | Yes | 34.867 s |
| French translation | “Thank you for your help” | “Please send the invoice” | No | 73.396 s |

The top choice matched the hand-written expectation in **3/4** cases. These
prompts are too few and too broad to estimate accuracy. The translation miss
could reflect CLM's task behaviour or numerical differences; this test does
not distinguish them. macOS reported about 14 GB of swap in use during the
test, but no pre-run swap baseline was recorded. These timings should not be
generalized to other Macs or workloads.

The exact ranking details and probabilities are in `data/mps-8b-smoke.json`
(ignored by Git in the test checkout).

## README reference replay in float16 and bfloat16

The exact typed question and tides ranking shown in the upstream README were
also replayed through `examples/mps_8b_reference.py`. The README values were
reported for CUDA/vLLM, so the differences below should not be attributed to
MPS alone without a matched encoder and revision comparison.

| Output | README CUDA/vLLM | MPS float16 | MPS bfloat16 |
| --- | ---: | ---: | ---: |
| Urgency yes probability | 0.41022 | 0.84162 | 0.84301 |
| Billing probability | 0.93878 | 0.98781 | 0.98839 |
| Frustration score | 1.98386 | 1.99998 | 1.99998 |
| Tides: Moon probability | 0.99700 | 0.99350 | 0.99321 |

Both discrete choices were the same. The urgency probability differed from the
README by 0.43140 in float16 and 0.43279 in bfloat16; using the checkpoint's
native bfloat16 format did not close that gap. The typed request took 165.9 s
in float16 and 185.7 s in bfloat16 in these single runs. Numerical equivalence
to the reference is **not** established. Full outputs are in
`data/mps-8b-reference.json` and `data/mps-8b-reference-bf16.json` in the test
checkout (both ignored by Git).
