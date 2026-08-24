# SparkGPT

A learning project to build and train a small decoder-only language model on one NVIDIA
DGX Spark.

This repository intentionally starts without an implementation. The owner is building each
component to understand the complete language-model pipeline rather than beginning with a
finished training framework.

## Initial target

- Approximately 250–300 million parameters
- Decoder-only Transformer
- 32K-token vocabulary
- 2,048-token context length
- BF16 training on one NVIDIA GB10
- Approximately 5 billion pretraining tokens
- FineWeb-Edu for initial pretraining experiments
- A small supervised fine-tuning stage after pretraining

The exact architecture and training budget will be selected after small-scale throughput and
memory benchmarks.

## Planned milestones

1. Define configuration objects and project layout.
2. Train or select a tokenizer.
3. Build deterministic token shards and a resumable data loader.
4. Implement embeddings, normalization, rotary positions, attention, MLP, and Transformer blocks.
5. Assemble the causal language model and verify its loss on a tiny batch.
6. Implement optimization, logging, evaluation, and checkpoint/resume.
7. Overfit a tiny dataset as an end-to-end correctness test.
8. Benchmark several model sizes on the DGX Spark.
9. Pretrain the selected model.
10. Add supervised fine-tuning and local inference.

## Likely model design

The first serious configuration will probably use RMSNorm, rotary position embeddings,
grouped-query causal attention, SwiGLU feed-forward layers, tied input/output embeddings, and
PyTorch scaled-dot-product attention. These are design intentions, not committed decisions.

## Data and evaluation principles

- Keep training data, checkpoints, credentials, and generated artifacts out of Git.
- Record dataset sources, versions, licenses, filters, and token counts.
- Create validation and test sets before training.
- Check for evaluation contamination and duplicate documents.
- Begin with tiny fixtures before downloading or processing a large corpus.
- Treat a model fitting in memory and a model being trainable in the available time as separate
  questions.

## Development rule

Implementation is added incrementally by the repository owner. AI assistance may explain,
review, debug, or implement a component only when explicitly requested.

## References

- [NVIDIA DGX Spark documentation](https://docs.nvidia.com/dgx/dgx-spark/)
- [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
