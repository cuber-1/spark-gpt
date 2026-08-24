# Seven-day training plan

## Decision

Train the 304M model from scratch. Treat 8B as a pretrained-model adaptation target. A model fitting
in 128GB and a model receiving enough training compute in seven days are different questions.

## Measured budget

Measured on 2026-08-24 on the local NVIDIA GB10 with PyTorch 2.13.0+cu130, BF16, batch 8, sequence
2,048, and AdamW:

| Metric | Result |
| --- | ---: |
| Parameters | 304,137,216 |
| Tokens/second | 9,666 |
| Estimated training TFLOP/s (`6ND/time`) | 17.64 |
| Peak PyTorch allocation | 34.32 GB |
| Target tokens | 4,499,963,904 |
| Optimizer steps | 17,166 |

The token target is `steps × batch × accumulation × context`. The checked-in global batch is
262,144 tokens/step. A raw 4.5B-token pass is approximately 5.4 days at measured throughput; the
remaining time is reserved for evaluation, checkpoints, data stalls, and recovery. Re-run the
benchmark with and without `--compile`, then change `max_steps` only if the sustained result and
remaining wall time justify it.

## Schedule

1. Day 0: validate CUDA/BF16, license/provenance, tokenizer, packed shards, disk, and power.
2. Day 1: run tests, tiny overfit, 300M throughput benchmark, and a 10M-token loss sanity run.
3. Days 1–6: train continuously; evaluate and atomically checkpoint every 250 optimizer steps.
4. Day 6: stop expansion experiments, preserve the strongest checkpoint, and run held-out evals.
5. Day 7: generate a fixed prompt suite, document limitations, checksum artifacts, and write the
   model/data cards.

## Go/no-go gates

- Training and validation losses are finite and trend down in the 10M-token sanity run.
- Resume produces the same next sampled batch and preserves optimizer/RNG state.
- Tokenized train/validation sets are split by document, not by token window.
- At least 100GB of disk remains free after packed data and retained checkpoints are budgeted.
- The dataset revision, source, license, filters, token counts, and tokenizer hash are recorded.
- The run remains below 110GB peak coherent memory to leave failure headroom.

The checked-in `configs/spark_300m_sanity.toml` processes about 8.4M non-repeated training tokens
from the local 10.2M-token gate corpus, evaluating every 2.1M tokens.

## Acceptance criteria

- Process at least 95% of the benchmark-derived target without NaN/Inf.
- Preserve `best.pt`, `last.pt`, run config, metrics, data manifest, tokenizer, and checksums.
- Report held-out loss/perplexity and the exact number of training tokens.
- Evaluate a fixed qualitative/safety prompt set without claiming ChatGPT-level capability.
- Publish weights only after a separate dataset and weight-license review.

## 8B decision

Dense training compute is approximately `6 × parameters × tokens`. Compute-optimal 8B pretraining
is roughly 160B tokens, far beyond one week on this box. The practical alternative is LoRA/SFT or
domain continued pretraining of an openly licensed base, with a preregistered before/after metric.
