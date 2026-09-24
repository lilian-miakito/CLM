# Qwen3-0.6B MPS smoke test

This is a functional test of the Apple Silicon embedding server and CLM's
unprojected `clm-raw` ranking path. It is **not** a test of the released CLM-8B
head: that head expects 4096-dimensional Qwen3-8B embeddings, while the small
model returns 1024 dimensions.

## Reproduce

Run from a clone with `requirements-macos.txt` installed. Download the pinned
pretrained model, then run the encoder and example script in separate terminals:

```bash
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir data/qwen3-0.6b
clm-mps-embed --model data/qwen3-0.6b --served-model-name qwen3-0.6b \
  --device mps --max-model-len 512 --port 8090
```

```bash
python examples/mps_small_smoke.py \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca
```

The inputs and expected top choices are frozen in the script. It checks that
the HTTP endpoint returns finite, L2-normalized embeddings, then ranks each
set of candidate actions using the normal `Embedder` and `Engine` paths.

## Observed on one Apple Silicon Mac

Python 3.12.14, PyTorch 2.14.0, Transformers 4.57.6, MPS, 24 GB unified
memory. Times below include HTTP and ranking but exclude model startup and the
initial embedding shape check.

| Case | Expected top | Observed top | Time |
| --- | --- | --- | ---: |
| Duplicate charge | Billing | Billing | 0.481 s |
| HTTP 500 after deployment | Engineering | Sales | 0.153 s |
| Ocean tides | Moon's gravitational pull | Earth's spherical shape | 0.055 s |
| French translation | “Thank you for your help” | “Please send the invoice” | 0.054 s |

The pipeline completed all four cases; the raw top choice matched the expected
one in **1/4**. These four hand-written cases are a smoke test, not a benchmark.
The raw probabilities are uncalibrated. The poor ranking on three examples is
a reason to avoid treating this small untrained pairing as a replacement for
the published CLM-8B model.
