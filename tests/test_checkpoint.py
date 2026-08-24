import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from sparkgpt.config import ModelConfig, TrainConfig
from sparkgpt.data import ByteTokenizer, TokenBatcher, prepare_token_files
from sparkgpt.model import GPT
from sparkgpt.training import load_checkpoint, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_restores_model_optimizer_rng_and_data_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text(
                "".join(f"checkpoint document {i} with content\n" for i in range(200))
            )
            data_dir = root / "data"
            prepare_token_files([source], data_dir, ByteTokenizer(), 0.2, seed=3)
            model_config = ModelConfig(
                vocab_size=256,
                context_length=8,
                n_layer=1,
                n_head=2,
                n_kv_head=1,
                n_embd=16,
                intermediate_size=48,
            )
            config = TrainConfig(model=model_config, data_dir=str(data_dir), device="cpu")
            model = GPT(model_config)
            optimizer = model.configure_optimizer(1e-3, 0.1, (0.9, 0.95), "cpu")
            initial_x = torch.randint(256, (2, 8))
            _, initial_loss = model(initial_x, initial_x)
            assert initial_loss is not None
            initial_loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            batcher = TokenBatcher(data_dir / "train.bin", 8, 2, 17, torch.device("cpu"))
            val_batcher = TokenBatcher(data_dir / "val.bin", 8, 2, 18, torch.device("cpu"))
            data_manifest = {
                "contents": json.loads((data_dir / "meta.json").read_text()),
                "sha256": hashlib.sha256((data_dir / "meta.json").read_bytes()).hexdigest(),
            }
            save_checkpoint(
                root / "state.pt",
                model,
                optimizer,
                config,
                12,
                2.5,
                batcher,
                val_batcher,
                data_manifest,
                None,
            )
            expected_optimizer_steps = [
                state["step"].clone() for state in optimizer.state.values() if "step" in state
            ]
            expected_batch = batcher.next()
            expected_val_batch = val_batcher.next()
            expected_random = torch.rand(4)
            with torch.no_grad():
                next(model.parameters()).zero_()
            batcher.next()
            val_batcher.next()
            torch.rand(20)
            step, loss, _ = load_checkpoint(
                root / "state.pt",
                model,
                optimizer,
                batcher,
                val_batcher,
                config,
                data_manifest,
            )
            actual_batch = batcher.next()
            actual_val_batch = val_batcher.next()
            actual_random = torch.rand(4)
            self.assertEqual(step, 12)
            self.assertEqual(loss, 2.5)
            torch.testing.assert_close(expected_batch[0], actual_batch[0])
            torch.testing.assert_close(expected_batch[1], actual_batch[1])
            torch.testing.assert_close(expected_val_batch[0], actual_val_batch[0])
            torch.testing.assert_close(expected_val_batch[1], actual_val_batch[1])
            torch.testing.assert_close(expected_random, actual_random)
            self.assertFalse(torch.count_nonzero(next(model.parameters())) == 0)
            actual_optimizer_steps = [
                state["step"] for state in optimizer.state.values() if "step" in state
            ]
            self.assertEqual(len(expected_optimizer_steps), len(actual_optimizer_steps))
            for expected, actual in zip(expected_optimizer_steps, actual_optimizer_steps):
                torch.testing.assert_close(expected, actual)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_resume_keeps_sampler_state_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text(
                "".join(f"gpu resume document {i} with content\n" for i in range(200))
            )
            data_dir = root / "data"
            prepare_token_files([source], data_dir, ByteTokenizer(), 0.2, seed=9)
            model_config = ModelConfig(
                vocab_size=256,
                context_length=8,
                n_layer=1,
                n_head=2,
                n_kv_head=1,
                n_embd=16,
                intermediate_size=48,
            )
            config = TrainConfig(model=model_config, data_dir=str(data_dir), device="cuda")
            device = torch.device("cuda")
            model = GPT(model_config).to(device)
            optimizer = model.configure_optimizer(1e-3, 0.1, (0.9, 0.95), "cuda")
            train_batcher = TokenBatcher(data_dir / "train.bin", 8, 2, 21, device)
            val_batcher = TokenBatcher(data_dir / "val.bin", 8, 2, 22, device)
            manifest_path = data_dir / "meta.json"
            data_manifest = {
                "contents": json.loads(manifest_path.read_text()),
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            }
            save_checkpoint(
                root / "cuda.pt",
                model,
                optimizer,
                config,
                1,
                3.0,
                train_batcher,
                val_batcher,
                data_manifest,
                None,
            )
            expected_x, _ = train_batcher.next()
            train_batcher.next()
            load_checkpoint(
                root / "cuda.pt",
                model,
                optimizer,
                train_batcher,
                val_batcher,
                config,
                data_manifest,
            )
            actual_x, _ = train_batcher.next()
            self.assertEqual(train_batcher.permutation.device.type, "cpu")
            torch.testing.assert_close(expected_x, actual_x)


if __name__ == "__main__":
    unittest.main()
