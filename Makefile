PYTHON ?= python3

.PHONY: prepare-demo test doctor smoke inspect

prepare-demo:
	$(PYTHON) -m sparkgpt prepare --input examples/tiny_corpus.txt --output data/demo --tokenizer byte --val-fraction 0.25

test:
	$(PYTHON) -m unittest discover -s tests -v

doctor:
	$(PYTHON) -m sparkgpt doctor

smoke: prepare-demo
	$(PYTHON) -m sparkgpt train --config configs/smoke.toml

inspect:
	$(PYTHON) -m sparkgpt inspect --config configs/spark_300m.toml
