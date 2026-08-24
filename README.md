# SparkGPT

SparkGPT is an original, readable GPT training stack built for one NVIDIA DGX Spark. The primary
goal is to train a useful **304M-parameter decoder-only model from scratch** while exposing every
important part of the pipeline: tokenization, packed data, attention, optimization, evaluation,
checkpointing, and generation.

This is a serious learning and engineering project, not a claim that one workstation can reproduce
a frontier model. NanoGPT inspired the learn-by-building approach; this implementation is written
from scratch and uses a modern RMSNorm + RoPE + GQA + SwiGLU architecture.

## What is realistic in one week?

| Lane | Parameters | One-week outcome |
| --- | ---: | --- |
| From scratch | 304,137,216 | Primary target: roughly 4.5B well-documented tokens |
| From scratch | 7,986,253,824 | Architecture sizing only; badly undertrained in one week |
| Pretrained adaptation | 8B class | Practical LoRA/SFT or domain continued-pretraining lane |

On this repository's NVIDIA GB10, a real BF16 forward/backward/AdamW benchmark at batch 8 and
context 2,048 measured **9,666 tokens/s**, **17.64 estimated training TFLOP/s**, and **34.32 GB**
peak PyTorch allocation. At that rate, 4.5B tokens takes about 5.4 uninterrupted days before data,
evaluation, checkpoint, and failure-recovery overhead. The budget is deliberately based on a
measured training step, not the Spark's advertised low-precision inference peak.

The first full 304M sanity run then processed **8,388,608 non-repeated FineWeb-Edu tokens** at an
average **9,839 tokens/s**. Training loss fell from 10.60 to 7.49 and held-out validation loss fell
from 8.56 at the first gate to 7.47 at completion. The raw record is in
[`benchmarks/gb10_300m_sanity_2026-08-24.json`](benchmarks/gb10_300m_sanity_2026-08-24.json).

## Architecture

The main config in [`configs/spark_300m.toml`](configs/spark_300m.toml) uses:

- 24 decoder blocks, width 1,024, and SwiGLU width 2,816
- 16 query heads and 4 key/value heads (grouped-query attention)
- rotary position embeddings and a 2,048-token context
- RMSNorm, no linear biases, and tied token/output embeddings
- PyTorch scaled-dot-product causal attention
- a 32,768-token SentencePiece BPE vocabulary

Run parameter accounting without allocating the model:

```bash
spark-gpt inspect --config configs/spark_300m.toml
spark-gpt inspect --config configs/spark_8b_sanity.toml
```

## Quick start

Python 3.11+ and a CUDA-enabled PyTorch build are required for serious training. On DGX Spark,
start from NVIDIA's supported PyTorch environment, then install this package:

```bash
git clone https://github.com/cuber-1/spark-gpt.git
cd spark-gpt
python -m pip install -e ".[data,dev]"
spark-gpt doctor
python -m unittest discover -s tests -v
```

Run the tiny end-to-end GPU exercise:

```bash
spark-gpt prepare \
  --input examples/tiny_corpus.txt \
  --output data/demo \
  --tokenizer byte \
  --val-fraction 0.25
spark-gpt train --config configs/smoke.toml
spark-gpt generate \
  --config configs/smoke.toml \
  --checkpoint runs/smoke/best.pt \
  --prompt "Spark is" \
  --max-new-tokens 80
```

The included smoke run has already been exercised on the GB10: training loss moved from 5.48 to
2.71 in 40 steps, with held-out validation loss 3.05.

## Preparing real data

SparkGPT accepts UTF-8 text (one document per non-empty line) or JSONL (one document per row). The
split is performed at document level using a stable hash so a document cannot leak between train
and validation.

```bash
# Train a 32K BPE on a representative, licensed sample.
spark-gpt train-tokenizer \
  --input data/tokenizer_sample.txt \
  --output-prefix tokenizers/spark-32k \
  --vocab-size 32768

# Pack local text or JSONL into memory-mapped uint16 token IDs.
spark-gpt prepare \
  --input data/corpus/*.jsonl \
  --output data/fineweb_edu \
  --tokenizer tokenizers/spark-32k.model \
  --jsonl-field text \
  --val-fraction 0.002
```

Do not treat a dataset name as a license. Pin the dataset revision, save its manifest, record every
filter, and review terms before publishing weights. See [`docs/data.md`](docs/data.md).

## Train, benchmark, and resume

```bash
# Re-measure this exact machine before locking the token budget.
spark-gpt benchmark \
  --config configs/spark_300m.toml \
  --batch-size 8 \
  --sequence-length 2048

# Main run (about 4.5B tokens with the checked-in config).
spark-gpt train --config configs/spark_300m.toml

# Checkpoints include model, optimizer, RNG, config, and data-sampler state.
spark-gpt train \
  --config configs/spark_300m.toml \
  --resume runs/spark-300m/last.pt
```

Metrics are appended to `runs/<name>/metrics.jsonl`. Checkpoints are written to a temporary file
and atomically renamed, so an interruption cannot leave a half-written `last.pt`.

## The 8B lane

[`configs/spark_8b_sanity.toml`](configs/spark_8b_sanity.toml) proves the architecture math without
allocating anything via `spark-gpt inspect`. It is not a training recommendation: compute-optimal
8B pretraining is on the order of 160B tokens, and full AdamW state plus activations creates a poor
fit for the 128GB coherent-memory budget.

For a useful one-week 8B result, adapt a compatible, openly licensed pretrained base with LoRA,
targeting 10–50M curated instruction tokens or 100–500M domain continued-pretraining tokens. The
base revision, data terms, adapter configuration, and before/after evaluation must all be recorded.

## Correctness and reproducibility

The test suite covers:

- exact analytical versus instantiated parameter counts
- finite forward/backward gradients and tied embeddings
- no future-token leakage through causal attention
- deterministic document splitting and next-token alignment
- checkpoint restoration of weights, optimizer, training RNGs, and both sampler positions

Each run writes the full config, parameter count, Git commit, PyTorch version, and device name.
See [`docs/training-plan.md`](docs/training-plan.md) for the seven-day gate and acceptance criteria.

## References

- [NVIDIA DGX Spark specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [PyTorch scaled dot product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [The FineWeb Datasets](https://arxiv.org/abs/2406.17557)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [NanoGPT](https://github.com/karpathy/nanoGPT)

## License

Code is released under the [MIT License](LICENSE). Dataset licenses, tokenizer training material,
and model weights are separate artifacts and are not covered by the code license.
