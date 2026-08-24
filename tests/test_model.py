import unittest

import torch

from sparkgpt.config import ModelConfig
from sparkgpt.model import GPT


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        context_length=16,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        intermediate_size=80,
    )


class ModelTests(unittest.TestCase):
    def test_forward_backward_and_tied_embeddings(self) -> None:
        model = GPT(tiny_config())
        tokens = torch.randint(64, (2, 12))
        logits, loss = model(tokens, tokens)
        self.assertEqual(logits.shape, (2, 12, 64))
        self.assertIsNotNone(loss)
        assert loss is not None
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)
        self.assertTrue(
            all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        )

    def test_causal_mask_blocks_future_information(self) -> None:
        torch.manual_seed(7)
        model = GPT(tiny_config()).eval()
        first = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        second = torch.tensor([[1, 2, 3, 4, 40, 41, 42, 43]])
        with torch.inference_mode():
            first_logits, _ = model(first)
            second_logits, _ = model(second)
        torch.testing.assert_close(first_logits[:, :4], second_logits[:, :4])

    def test_context_limit_is_enforced(self) -> None:
        model = GPT(tiny_config())
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 17), dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
