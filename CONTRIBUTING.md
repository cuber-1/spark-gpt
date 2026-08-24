# Contributing

Keep the training path readable, tested, and measurable. Before opening a change:

1. Run `python -m unittest discover -s tests -v`.
2. Add a correctness test for changes to masking, positions, data, or checkpoint state.
3. Include the exact command, environment, and raw result for performance claims.
4. Document the source and license of any data, weights, or adapted code.

Generated datasets, checkpoints, credentials, and run artifacts must not be committed.
