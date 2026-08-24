# Data and tokenizer contract

SparkGPT stores contiguous token IDs as raw `train.bin` and `val.bin` files plus `meta.json`.
Vocabularies up to 65,535 use `uint16`; larger vocabularies use `uint32`. Files can therefore be
opened as NumPy memory maps without loading the corpus into RAM.

## Required provenance

Record these fields before a serious run:

- dataset name, URL, immutable revision, config, and split
- original license/terms and the date they were reviewed
- document filters, language filters, deduplication, and PII/toxicity handling
- document and token counts before and after each filter
- train/validation split seed and method
- tokenizer input sample, SentencePiece command, model hash, vocabulary size, and special IDs
- hashes of the final manifest and all packed shards

The built-in preparer uses a seeded BLAKE2 hash of the complete document for the split. Repeated
identical documents therefore land in the same split, but this is not a substitute for corpus-level
near-deduplication.

Training consumes a shuffled permutation of aligned, non-overlapping context windows before starting
a new epoch. The permutation and cursor are checkpointed, avoiding the large amount of repeated-token
sampling caused by random windows with replacement.

## Format assumptions

- Plain text: one non-empty line is one document.
- JSONL: one object is one document; `--jsonl-field` selects the text field.
- SentencePiece: EOS is appended when the tokenizer defines one.
- Byte tokenizer: intended only for smoke tests; it has no EOS token.

The code never downloads a corpus implicitly. Downloading is kept separate so users must make an
explicit data-source and licensing decision.

## FineWeb-Edu helpers

For a serious run, download the pinned 10BT snapshot first. Hugging Face's snapshot downloader
resumes interrupted files and verifies its content-addressed artifacts:

```bash
python scripts/download_fineweb.py
```

Then pack 4.6B tokens locally with deterministic source ordering and a document-level validation
split:

```bash
python scripts/prepare_fineweb.py \
  --tokenizer tokenizers/spark-32k.model \
  --local-parquet-root data/source/fineweb-edu-10BT/sample/10BT \
  --output data/fineweb_edu \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --max-tokens 4600000000
```

The local preparer records every Parquet SHA-256 and saves an atomic `resume.json` at each fsynced
shard boundary. After an interruption, rerun the identical command; extra partial output is
truncated to the verified boundary before packing resumes. Completed token files are never
overwritten.

Use a new output directory for every distinct preparation. Resume identity binds the resolved
source root, ordered source paths and hashes, tokenizer metadata, revision, token limit, validation
fraction, and seed. Moving or changing any of them requires another output directory. Treat
`resume.json`, `train.bin.partial`, and `val.bin.partial` as one recovery unit and never hand-edit or
delete only part of it; the current shard may safely replay after recovery. A complete output has
`meta.json`, `train.bin`, and `val.bin` and no `resume.json`. If the snapshot ends below the requested
token limit, preparation deliberately fails while retaining recovery state.

The streaming helpers remain useful for small tokenizer and pipeline samples:

```bash
python scripts/sample_fineweb.py --max-characters 50000000
spark-gpt train-tokenizer --input data/tokenizer_sample.txt \
  --output-prefix tokenizers/spark-32k --vocab-size 32768
python scripts/prepare_fineweb.py --tokenizer tokenizers/spark-32k.model \
  --output data/fineweb_stream_sample --max-tokens 10000000
```

Remote streaming retains `.partial` files for diagnosis but is intentionally not resumable. Use the
local snapshot path for multi-billion-token preparation.

Some `datasets`/PyArrow builds hit the open
[`huggingface/datasets#7566`](https://github.com/huggingface/datasets/issues/7566) native-thread crash
only during successful interpreter shutdown. If that exact post-success crash occurs, set
`SPARKGPT_EXIT_WITHOUT_FINALIZE=1`; the helpers explicitly close and rename outputs and flush stdout
before using the same opt-in clean-exit workaround documented by NVIDIA's Cosmos framework.
