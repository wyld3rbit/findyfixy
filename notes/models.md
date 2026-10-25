# Models

Recorded so a silent swap is visible. If a hash or revision below stops matching, stop and investigate.

## Embedding model (Day 3)

- Name: `BAAI/bge-small-en-v1.5`
- License: MIT
- Pinned revision: `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` (last changed upstream 2024-02-22)
- Weights file: `model.safetensors`, 133466304 bytes
- SHA-256 of the weights: `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad`
- Dimensions: 384, vectors normalised
- Downloaded: 2026-10-21, first run of `uv run python -m app.ingest`
- Where it is pinned in code: `MODEL_NAME` and `MODEL_REVISION` in `code/app/ingest.py`

Check it any time:

```bash
shasum -a 256 < ~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/*/model.safetensors
```

## Reranker (Day 4)

- Name: `BAAI/bge-reranker-base` (cross-encoder)
- License: MIT (model card)
- Pinned revision: `2cfc18c9415c912f9d8155881c133215df768a70` (last changed upstream 2024-06-24). The downloaded snapshot folder carries the same hash.
- Weights file: `model.safetensors`, 1112206140 bytes
- SHA-256 of the weights: `ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd`
- Downloaded: first run of `uv run python -m app.eval_retrieval`
- Where it is pinned in code: `RERANKER_NAME` and `RERANKER_REVISION` in `code/app/retrieve.py`

Check it any time:

```bash
shasum -a 256 < ~/.cache/huggingface/hub/models--BAAI--bge-reranker-base/snapshots/*/model.safetensors
```
