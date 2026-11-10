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

## Qwen3 candidates (pulled 2026-10-26 for the model comparison)

The project rule "Qwen2.5 1.5B and 7B only" was lifted by the learner on 2026-10-26 for this comparison. Limits that still hold: open source (OSI license), at most 8B parameters, 16 GB machine.

| Ollama tag | Parameters, quantisation | Download | Ollama digest (sha256) | License (official model card) |
|---|---|---|---|---|
| `qwen3:4b-instruct` | 4.0B, Q4_K_M | 2.5 GB | `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0` | Apache-2.0, `Qwen/Qwen3-4B-Instruct-2507`. Non-thinking only. |
| `qwen3:8b` | 8.2B, Q4_K_M | 5.2 GB | `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41` | Apache-2.0, `Qwen/Qwen3-8B`. Thinking on by default; the code switches it off. |

- Existing models, for comparison: `qwen2.5:1.5b` (`65ec06548149`), `qwen2.5:7b` (`845dbda0ea48`).
- The model card for Qwen3 8B warns against greedy decoding in thinking mode and recommends temperature 0.7 in non-thinking mode. This project uses temperature 0 with thinking off. Watch for repetition or empty answers; the 120 s read timeout bounds a loop.
- Thinking cost on the same alert, cold load included: 17.6 s on, 6.9 s off (`qwen3:8b`).
