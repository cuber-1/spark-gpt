import json
import tempfile
import unittest
from pathlib import Path

import torch

from sparkgpt.data import ByteTokenizer, TokenBatcher, prepare_token_files


class DataTests(unittest.TestCase):
    def test_prepare_and_deterministic_batch_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "documents.txt"
            source.write_text(
                "".join(
                    f"document {index} has enough repeated training text\n" for index in range(200)
                )
            )
            output = root / "packed"
            manifest = prepare_token_files(
                [source], output, ByteTokenizer(), val_fraction=0.25, seed=11
            )
            self.assertGreater(manifest["train"], 0)
            self.assertGreater(manifest["val"], 0)
            self.assertEqual(json.loads((output / "meta.json").read_text())["dtype"], "uint16")
            batcher = TokenBatcher(output / "train.bin", 16, 4, 99, torch.device("cpu"))
            state = batcher.state_dict()
            expected_x, expected_y = batcher.next()
            batcher.next()
            batcher.load_state_dict(state)
            actual_x, actual_y = batcher.next()
            torch.testing.assert_close(expected_x, actual_x)
            torch.testing.assert_close(expected_y, actual_y)
            torch.testing.assert_close(expected_x[:, 1:], expected_y[:, :-1])

    def test_jsonl_field_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "documents.jsonl"
            source.write_text('{"wrong": "text"}\n')
            with self.assertRaises(TypeError):
                prepare_token_files([source], root / "packed", ByteTokenizer(), 0.2)


if __name__ == "__main__":
    unittest.main()
